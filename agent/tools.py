"""
Agent tools exposed to the LLM.

Tool call flow:
  1. set_platform  — pick the ride platform (default: uber)
  2. search_rides  — discover options and prices
  3. suggest_ride  — recommend one option; INTERRUPTS for user confirmation
  4. book_ride     — executes only when state.ride_confirmed is True

Each tool returns a Command that both updates AgentState fields and injects
the ToolMessage (keyed by tool_call_id) that the LLM reads as the result.

Adding a new platform (e.g. Lyft):
  1. Subclass RidePlatformAdapter in adapters/
  2. Add the class to PLATFORM_REGISTRY below
  — no other changes required.
"""

from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command, interrupt

from adapters.uber_adapter import UberRideAdapter
from agent.action_log import make_log_entry
from agent.state import AgentState

PLATFORM_REGISTRY = {
    "uber": UberRideAdapter,
    # "lyft": LyftRideAdapter,   ← add here
    # "zocdoc": ZocdocAdapter,   ← different domain, subclass PlatformAdapter
}


def _get_adapter(platform_name: str):
    cls = PLATFORM_REGISTRY.get(platform_name)
    if cls is None:
        raise ValueError(
            f"Unknown platform '{platform_name}'. "
            f"Supported: {list(PLATFORM_REGISTRY)}"
        )
    return cls()


# ---------------------------------------------------------------------------
# Tool 1 — set_platform
# ---------------------------------------------------------------------------


@tool
def set_platform(
    platform: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """
    Set the ride-hailing platform to use for this session.

    Args:
        platform: Platform name, e.g. 'uber'.
    """
    supported = list(PLATFORM_REGISTRY)

    if platform not in supported:
        msg = (
            f"Unsupported platform '{platform}'. "
            f"Supported platforms: {supported}."
        )
        log = make_log_entry(
            "set_platform",
            requested={"platform": platform},
            verified={"supported": False},
            executed={},
            outcome=f"error: {msg}",
        )
        return Command(
            update={
                "action_log": [log],
                "messages": [ToolMessage(content=msg, tool_call_id=tool_call_id)],
            }
        )

    log = make_log_entry(
        "set_platform",
        requested={"platform": platform},
        verified={"supported": True},
        executed={"adapter_class": PLATFORM_REGISTRY[platform].__name__},
        outcome="success",
    )
    return Command(
        update={
            "platform_adapter": platform,
            "action_log": [log],
            "messages": [
                ToolMessage(
                    content=f"Platform set to '{platform}'. Ready to search for rides.",
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )


# ---------------------------------------------------------------------------
# Tool 2 — search_rides
# ---------------------------------------------------------------------------


@tool
def search_rides(
    pickup: str,
    dropoff: str,
    state: Annotated[AgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """
    Search for available rides between two locations.

    Args:
        pickup:  Starting address or landmark (e.g. '123 Main St, New York').
        dropoff: Destination address or landmark (e.g. 'JFK Airport').
    """
    platform = state.get("platform_adapter") or "uber"

    try:
        adapter = _get_adapter(platform)
        results = adapter.search_rides(pickup=pickup, dropoff=dropoff)

        summary_lines = [
            f"{i}. **{r['display_name']}** — {r['price_estimate']} "
            f"(~{r['duration_estimate_minutes']} min, up to {r['capacity']} passengers)"
            for i, r in enumerate(results)
        ]
        content = f"Found {len(results)} ride options:\n" + "\n".join(summary_lines)

        log = make_log_entry(
            "search_rides",
            requested={"pickup": pickup, "dropoff": dropoff, "platform": platform},
            verified={"adapter_available": True},
            executed={"method": "search_rides", "pickup": pickup, "dropoff": dropoff},
            outcome=f"found {len(results)} options",
        )
        return Command(
            update={
                "search_results": results,
                "action_log": [log],
                "messages": [ToolMessage(content=content, tool_call_id=tool_call_id)],
            }
        )

    except Exception as exc:
        log = make_log_entry(
            "search_rides",
            requested={"pickup": pickup, "dropoff": dropoff},
            verified={},
            executed={},
            outcome=f"error: {exc}",
        )
        return Command(
            update={
                "action_log": [log],
                "messages": [
                    ToolMessage(
                        content=f"Error searching rides: {exc}",
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )


# ---------------------------------------------------------------------------
# Tool 3 — suggest_ride  (triggers UI confirmation via interrupt)
# ---------------------------------------------------------------------------


@tool
def suggest_ride(
    ride_index: int,
    reasoning: str,
    state: Annotated[AgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """
    Suggest a specific ride option to the user and wait for their confirmation.
    The graph will PAUSE here until the user clicks Confirm or Reject in the UI.
    Do NOT call book_ride until this tool has returned.

    Args:
        ride_index: Zero-based index into the list returned by search_rides.
        reasoning:  Short explanation of why this option was chosen.
    """
    results = state.get("search_results") or []

    if not results:
        msg = "No search results available. Call search_rides first."
        log = make_log_entry(
            "suggest_ride",
            requested={"ride_index": ride_index},
            verified={"has_results": False},
            executed={},
            outcome="error: no search results",
        )
        return Command(
            update={
                "action_log": [log],
                "messages": [ToolMessage(content=msg, tool_call_id=tool_call_id)],
            }
        )

    if ride_index < 0 or ride_index >= len(results):
        msg = f"Invalid ride_index {ride_index}. Valid range: 0–{len(results) - 1}."
        log = make_log_entry(
            "suggest_ride",
            requested={"ride_index": ride_index},
            verified={"valid_index": False},
            executed={},
            outcome="error: invalid index",
        )
        return Command(
            update={
                "action_log": [log],
                "messages": [ToolMessage(content=msg, tool_call_id=tool_call_id)],
            }
        )

    ride = results[ride_index]

    # ----------------------------------------------------------------
    # INTERRUPT — graph pauses here; Streamlit shows confirm/reject UI.
    # On resume, `confirmed` is the bool the user chose.
    # Code before interrupt() may replay on resume; keep it side-effect-free.
    # ----------------------------------------------------------------
    confirmed: bool = interrupt(
        {
            "suggested_ride": ride,
            "reasoning": reasoning,
        }
    )

    # Everything below runs exactly once — after the user responds.
    outcome = "confirmed" if confirmed else "rejected"
    log = make_log_entry(
        "suggest_ride",
        requested={"ride_index": ride_index, "reasoning": reasoning},
        verified={"ride": ride["display_name"]},
        executed={"user_decision": outcome},
        outcome=outcome,
    )

    if confirmed:
        content = (
            f"User confirmed **{ride['display_name']}**. "
            "Proceeding to book the ride."
        )
    else:
        content = (
            f"User rejected **{ride['display_name']}**. "
            "Ask the user what they'd prefer or suggest an alternative."
        )

    return Command(
        update={
            "suggested_ride": ride,
            "ride_confirmed": confirmed,
            "action_log": [log],
            "messages": [ToolMessage(content=content, tool_call_id=tool_call_id)],
        }
    )


# ---------------------------------------------------------------------------
# Tool 4 — book_ride
# ---------------------------------------------------------------------------


@tool
def book_ride(
    state: Annotated[AgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """
    Book the ride that the user has already confirmed via suggest_ride.
    Will refuse to proceed if the confirmation gate has not been passed.
    """
    confirmed = state.get("ride_confirmed", False)
    ride = state.get("suggested_ride")
    platform = state.get("platform_adapter") or "uber"

    # Safety gate — cannot be bypassed by the LLM
    if not confirmed or not ride:
        msg = (
            "Cannot book: the user has not confirmed a ride yet. "
            "Call suggest_ride first and wait for user confirmation."
        )
        log = make_log_entry(
            "book_ride",
            requested={"platform": platform},
            verified={"ride_confirmed": confirmed, "ride_available": bool(ride)},
            executed={},
            outcome="blocked: confirmation gate not satisfied",
        )
        return Command(
            update={
                "action_log": [log],
                "messages": [ToolMessage(content=msg, tool_call_id=tool_call_id)],
            }
        )

    try:
        adapter = _get_adapter(platform)
        booking = adapter.book_ride(ride)

        driver = booking.get("driver", {})
        content = (
            f"Ride booked successfully!\n\n"
            f"**Ride ID:** `{booking['ride_id']}`\n"
            f"**Driver:** {driver.get('name')} "
            f"({driver.get('vehicle')}, plate {driver.get('license_plate')})\n"
            f"**Driver rating:** ⭐ {driver.get('rating')}\n"
            f"**Pickup ETA:** {booking['pickup_eta_minutes']} minutes\n"
            f"**Price estimate:** {booking['price_estimate']}\n"
        )

        log = make_log_entry(
            "book_ride",
            requested={"ride": ride["display_name"], "platform": platform},
            verified={"ride_confirmed": True, "ride_available": True},
            executed={
                "method": "book_ride",
                "product": ride["display_name"],
                "ride_id": booking["ride_id"],
            },
            outcome=f"success: ride_id={booking['ride_id']}",
        )
        return Command(
            update={
                "booked_ride": booking,
                "action_log": [log],
                "messages": [ToolMessage(content=content, tool_call_id=tool_call_id)],
            }
        )

    except Exception as exc:
        log = make_log_entry(
            "book_ride",
            requested={"ride": ride.get("display_name"), "platform": platform},
            verified={"ride_confirmed": True},
            executed={},
            outcome=f"error: {exc}",
        )
        return Command(
            update={
                "action_log": [log],
                "messages": [
                    ToolMessage(
                        content=f"Booking failed: {exc}",
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )


ALL_TOOLS = [set_platform, search_rides, suggest_ride, book_ride]
