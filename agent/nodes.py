from __future__ import annotations

from dataclasses import asdict

from langchain_core.messages import AIMessage
from langgraph.types import interrupt

from adapters.registry import get_adapter
from agent.state import AgentState
import log.store as log_store


# ---------------------------------------------------------------------------
# Helper: get adapter from state
# ---------------------------------------------------------------------------

def _adapter(state: AgentState):
    return get_adapter(state.platform)


def _option_dict(opt) -> dict:
    return {
        "option_id": opt.option_id,
        "product_name": opt.product_name,
        "price_low": opt.price_low,
        "price_high": opt.price_high,
        "currency": opt.currency,
        "eta_minutes": opt.eta_minutes,
        "duration_minutes": opt.duration_minutes,
        "surge_multiplier": opt.surge_multiplier,
    }


# ---------------------------------------------------------------------------
# Node: discover_options
# ---------------------------------------------------------------------------

def discover_options(state: AgentState) -> dict:
    log_store.append(log_store.ActionLogEntry(
        action_type="REQUESTED",
        node="discover_options",
        platform=state.platform,
        payload={"pickup": state.pickup, "dropoff": state.dropoff},
        response={},
        success=True,
    ))

    try:
        adapter = _adapter(state)
        options = adapter.get_price_estimates(state.pickup, state.dropoff)

        log_store.append(log_store.ActionLogEntry(
            action_type="RESULT",
            node="discover_options",
            platform=state.platform,
            payload={"pickup": state.pickup, "dropoff": state.dropoff},
            response={"options_count": len(options), "options": [_option_dict(o) for o in options]},
            success=True,
        ))

        summary = "\n".join(
            f"  {i+1}. {o.product_name}: {o.price_display}, ~{o.eta_minutes} min away"
            for i, o in enumerate(options)
        )
        return {
            "options": options,
            "error": None,
            "messages": [AIMessage(content=f"Found {len(options)} ride options:\n{summary}")],
        }
    except Exception as exc:
        log_store.append(log_store.ActionLogEntry(
            action_type="ERROR",
            node="discover_options",
            platform=state.platform,
            payload={"pickup": state.pickup, "dropoff": state.dropoff},
            response={"error": str(exc)},
            success=False,
        ))
        return {"error": str(exc), "options": []}


# ---------------------------------------------------------------------------
# Node: compare_prices
# ---------------------------------------------------------------------------

def compare_prices(state: AgentState) -> dict:
    """
    Formats the options list for display. In the graph this is a pass-through
    that sets awaiting_confirmation=True so the UI shows the confirmation gate.
    """
    log_store.append(log_store.ActionLogEntry(
        action_type="REQUESTED",
        node="compare_prices",
        platform=state.platform,
        payload={"options_count": len(state.options)},
        response={},
        success=True,
    ))

    if not state.options:
        return {"error": "No options available to compare.", "awaiting_confirmation": False}

    lines = []
    for i, o in enumerate(state.options):
        surge = f" (surge {o.surge_multiplier}x)" if o.surge_multiplier > 1 else ""
        lines.append(
            f"{i+1}. {o.product_name}: {o.price_display}{surge} · "
            f"ETA {o.eta_minutes} min · Trip ~{o.duration_minutes} min"
        )
    comparison = "\n".join(lines)

    log_store.append(log_store.ActionLogEntry(
        action_type="RESULT",
        node="compare_prices",
        platform=state.platform,
        payload={"options_count": len(state.options)},
        response={"comparison": comparison},
        success=True,
    ))

    return {
        "awaiting_confirmation": True,
        "messages": [AIMessage(content=f"Price comparison:\n{comparison}\n\nPlease select a ride and confirm.")],
    }


# ---------------------------------------------------------------------------
# Node: request_confirmation  (hard interrupt — graph suspends here)
# ---------------------------------------------------------------------------

def request_confirmation(state: AgentState) -> dict:
    """
    Suspends the graph until the UI resumes it with the user's decision.
    `interrupt()` raises a special LangGraph exception; the graph checkpoints
    state and pauses. The UI resumes by calling graph.invoke(Command(resume=...)).
    """
    if state.selected_option_index is None:
        return {"error": "No ride option selected."}

    option = state.options[state.selected_option_index]

    log_store.append(log_store.ActionLogEntry(
        action_type="REQUESTED",
        node="request_confirmation",
        platform=state.platform,
        payload={
            "option": _option_dict(option),
            "pickup": state.pickup,
            "dropoff": state.dropoff,
        },
        response={},
        success=True,
    ))

    # Hard suspension — the UI must call Command(resume={"confirmed": True/False})
    user_response = interrupt({
        "question": (
            f"Book {option.product_name} for {option.price_display} "
            f"from '{state.pickup}' to '{state.dropoff}'?"
        ),
        "option": _option_dict(option),
    })

    confirmed: bool = user_response.get("confirmed", False)

    log_store.log_verification(
        node="request_confirmation",
        platform=state.platform,
        details={"confirmed": confirmed, "option": _option_dict(option)},
    )

    return {"user_confirmed": confirmed, "awaiting_confirmation": False}


# ---------------------------------------------------------------------------
# Node: book_ride
# ---------------------------------------------------------------------------

def book_ride(state: AgentState) -> dict:
    option = state.options[state.selected_option_index]

    log_store.append(log_store.ActionLogEntry(
        action_type="REQUESTED",
        node="book_ride",
        platform=state.platform,
        payload={"option": _option_dict(option), "pickup": state.pickup, "dropoff": state.dropoff},
        response={},
        success=True,
    ))

    try:
        adapter = _adapter(state)
        booking = adapter.book_ride(option, state.pickup, state.dropoff)

        log_store.log_executed(
            node="book_ride",
            platform=state.platform,
            payload={"option": _option_dict(option)},
            response={
                "booking_id": booking.booking_id,
                "driver": booking.driver_name,
                "vehicle": f"{booking.vehicle_make} {booking.vehicle_model}",
                "plate": booking.license_plate,
                "status": booking.status,
            },
        )

        return {
            "booking": booking,
            "ride_status": booking.status,
            "error": None,
            "messages": [AIMessage(
                content=(
                    f"Booked! Your {booking.option.product_name} is on its way.\n"
                    f"Driver: {booking.driver_name} · {booking.vehicle_make} {booking.vehicle_model} · {booking.license_plate}"
                )
            )],
        }
    except Exception as exc:
        log_store.append(log_store.ActionLogEntry(
            action_type="ERROR",
            node="book_ride",
            platform=state.platform,
            payload={"option": _option_dict(option)},
            response={"error": str(exc)},
            success=False,
        ))
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Node: track_ride
# ---------------------------------------------------------------------------

def track_ride(state: AgentState) -> dict:
    if not state.booking:
        return {"error": "No active booking to track."}

    log_store.append(log_store.ActionLogEntry(
        action_type="REQUESTED",
        node="track_ride",
        platform=state.platform,
        payload={"booking_id": state.booking.booking_id},
        response={},
        success=True,
    ))

    try:
        adapter = _adapter(state)
        status = adapter.get_ride_status(state.booking.booking_id)

        log_store.append(log_store.ActionLogEntry(
            action_type="RESULT",
            node="track_ride",
            platform=state.platform,
            payload={"booking_id": state.booking.booking_id},
            response={"status": status},
            success=True,
        ))

        return {"ride_status": status}
    except Exception as exc:
        log_store.append(log_store.ActionLogEntry(
            action_type="ERROR",
            node="track_ride",
            platform=state.platform,
            payload={"booking_id": state.booking.booking_id},
            response={"error": str(exc)},
            success=False,
        ))
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Node: cancel_ride
# ---------------------------------------------------------------------------

def cancel_ride(state: AgentState) -> dict:
    if not state.booking:
        return {"error": "No active booking to cancel."}

    log_store.append(log_store.ActionLogEntry(
        action_type="REQUESTED",
        node="cancel_ride",
        platform=state.platform,
        payload={"booking_id": state.booking.booking_id},
        response={},
        success=True,
    ))

    try:
        adapter = _adapter(state)
        result = adapter.cancel_ride(state.booking.booking_id)

        log_store.log_executed(
            node="cancel_ride",
            platform=state.platform,
            payload={"booking_id": state.booking.booking_id},
            response={"success": result.success, "message": result.message, "fee": result.cancellation_fee},
        )

        return {
            "ride_status": "CANCELLED",
            "cancel_requested": False,
            "error": None,
            "messages": [AIMessage(content=f"Ride cancelled. {result.message}")],
        }
    except Exception as exc:
        log_store.append(log_store.ActionLogEntry(
            action_type="ERROR",
            node="cancel_ride",
            platform=state.platform,
            payload={"booking_id": state.booking.booking_id},
            response={"error": str(exc)},
            success=False,
        ))
        return {"error": str(exc)}
