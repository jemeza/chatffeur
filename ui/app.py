from __future__ import annotations

"""
ChatFFeur — AI Ride Booking Agent
Run with: streamlit run ui/app.py
"""

import sys
import os
import uuid
import time

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langgraph.types import Command

import log.store as log_store
from adapters.registry import available_platforms
from adapters.base import RideOption
from agent.graph import graph
from agent.state import AgentState

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

log_store.init_db()

st.set_page_config(
    page_title="ChatFFeur",
    page_icon="🚗",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

def _init_session():
    defaults = {
        "thread_id": str(uuid.uuid4()),
        "phase": "input",       # input | searching | confirming | booked | tracking | done
        "options": [],
        "booking": None,
        "ride_status": None,
        "error": None,
        "platform": "mock",
        "pickup": "",
        "dropoff": "",
        "selected_index": 0,
        "graph_state": None,    # last AgentState snapshot
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_session()


def _thread_config() -> dict:
    return {"configurable": {"thread_id": st.session_state.thread_id}}


def _reset():
    keys_to_clear = [k for k in st.session_state if k != "_streamlit_internal"]
    for k in keys_to_clear:
        del st.session_state[k]
    _init_session()
    st.rerun()


# ---------------------------------------------------------------------------
# Graph interaction helpers
# ---------------------------------------------------------------------------

def _run_graph(input_state: dict | Command):
    """Invoke the graph and sync results into session_state."""
    try:
        result = graph.invoke(input_state, config=_thread_config())
        _sync_from_graph(result)
    except Exception as exc:
        st.session_state.error = str(exc)


def _sync_from_graph(result: dict | AgentState | None):
    if result is None:
        return
    if isinstance(result, dict):
        state = result
    else:
        state = result.model_dump() if hasattr(result, "model_dump") else vars(result)

    if state.get("options"):
        st.session_state.options = state["options"]
    if state.get("booking") is not None:
        st.session_state.booking = state["booking"]
    if state.get("ride_status"):
        st.session_state.ride_status = state["ride_status"]
    if state.get("error"):
        st.session_state.error = state["error"]


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

STATUS_EMOJI = {
    "PROCESSING":      "⏳",
    "DRIVER_ASSIGNED": "✅",
    "EN_ROUTE":        "🚗",
    "ARRIVED":         "📍",
    "IN_PROGRESS":     "🛣️",
    "COMPLETED":       "🏁",
    "CANCELLED":       "❌",
    "NO_DRIVERS":      "😔",
}

STATUS_LABEL = {
    "PROCESSING":      "Processing…",
    "DRIVER_ASSIGNED": "Driver assigned",
    "EN_ROUTE":        "Driver en route",
    "ARRIVED":         "Driver arrived",
    "IN_PROGRESS":     "Trip in progress",
    "COMPLETED":       "Trip completed",
    "CANCELLED":       "Cancelled",
    "NO_DRIVERS":      "No drivers available",
}


def _render_option_card(opt: RideOption, index: int, selected: bool) -> None:
    border_color = "#1DB954" if selected else "#2d2d2d"
    surge_badge = (
        f'<span style="background:#ff4b4b;color:white;padding:2px 6px;border-radius:4px;font-size:0.75rem;">⚡ {opt.surge_multiplier}x surge</span>'
        if opt.surge_multiplier > 1 else ""
    )
    st.markdown(
        f"""
        <div style="border:2px solid {border_color};border-radius:8px;padding:12px;margin-bottom:8px;">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <div>
                    <strong style="font-size:1.1rem;">{opt.product_name}</strong>
                    &nbsp;{surge_badge}
                </div>
                <div style="text-align:right;">
                    <strong style="font-size:1.2rem;">{opt.price_display}</strong><br/>
                    <small style="color:#aaa;">~{opt.duration_minutes} min trip</small>
                </div>
            </div>
            <div style="margin-top:6px;color:#ccc;font-size:0.9rem;">
                🕐 Driver ~{opt.eta_minutes} min away
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_log_panel():
    st.subheader("Action Log", anchor=False)
    entries = log_store.fetch_all(limit=100)
    if not entries:
        st.caption("No actions recorded yet.")
        return

    # Colour coding
    _colour = {
        "REQUESTED": "#4a9eff",
        "VERIFIED":  "#f0a500",
        "EXECUTED":  "#1DB954",
        "RESULT":    "#888",
        "ERROR":     "#ff4b4b",
    }

    rows = []
    for e in entries:
        ts = e.timestamp.strftime("%H:%M:%S")
        colour = _colour.get(e.action_type, "#ccc")
        badge = f'<span style="color:{colour};font-weight:bold;">{e.action_type}</span>'
        icon = "✓" if e.success else "✗"
        rows.append(
            f'<div style="font-family:monospace;font-size:0.8rem;padding:3px 0;border-bottom:1px solid #333;">'
            f'{ts} {badge} <span style="color:#aaa;">{e.node}</span>'
            f' <span style="color:#666;">[{e.platform}]</span> {icon}'
            f'</div>'
        )

    st.markdown(
        f'<div style="max-height:300px;overflow-y:auto;padding:8px;background:#1a1a1a;border-radius:6px;">'
        + "".join(rows)
        + "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Page sections
# ---------------------------------------------------------------------------

def _section_input():
    st.markdown("### 1. Where are you going?")
    col1, col2 = st.columns(2)
    with col1:
        pickup = st.text_input("Pickup location", value=st.session_state.pickup, placeholder="e.g. 123 Main St, NYC")
    with col2:
        dropoff = st.text_input("Dropoff location", value=st.session_state.dropoff, placeholder="e.g. JFK Airport")

    platform = st.selectbox(
        "Platform",
        options=available_platforms(),
        index=available_platforms().index(st.session_state.platform),
    )

    if st.button("Find Rides 🔍", type="primary", disabled=not (pickup and dropoff)):
        st.session_state.pickup = pickup
        st.session_state.dropoff = dropoff
        st.session_state.platform = platform
        st.session_state.phase = "searching"
        st.session_state.error = None
        st.session_state.thread_id = str(uuid.uuid4())  # fresh thread per booking
        st.rerun()


def _section_searching():
    with st.spinner("Fetching ride options…"):
        _run_graph(AgentState(
            pickup=st.session_state.pickup,
            dropoff=st.session_state.dropoff,
            platform=st.session_state.platform,
        ).model_dump())

    if st.session_state.error:
        st.error(f"Error: {st.session_state.error}")
        if st.button("Try again"):
            st.session_state.phase = "input"
            st.session_state.error = None
            st.rerun()
        return

    st.session_state.phase = "confirming"
    st.rerun()


def _section_confirming():
    options: list[RideOption] = st.session_state.options
    if not options:
        st.warning("No ride options found.")
        if st.button("Back"):
            st.session_state.phase = "input"
            st.rerun()
        return

    st.markdown("### 2. Choose your ride")
    st.caption(f"From **{st.session_state.pickup}** → **{st.session_state.dropoff}**")

    selected = st.session_state.selected_index
    for i, opt in enumerate(options):
        col_card, col_sel = st.columns([5, 1])
        with col_card:
            _render_option_card(opt, i, selected == i)
        with col_sel:
            st.write("")  # vertical spacing
            if st.button("Select", key=f"sel_{i}"):
                st.session_state.selected_index = i
                st.rerun()

    st.divider()

    chosen = options[st.session_state.selected_index]
    st.markdown(
        f"#### Confirm booking: **{chosen.product_name}** for **{chosen.price_display}**?"
    )
    st.caption(f"Driver ETA: ~{chosen.eta_minutes} min · Trip: ~{chosen.duration_minutes} min")

    col_yes, col_no, _ = st.columns([1, 1, 4])
    with col_yes:
        if st.button("✓ Confirm", type="primary"):
            st.session_state.phase = "booking"
            st.session_state.error = None
            st.rerun()
    with col_no:
        if st.button("✗ Cancel"):
            st.session_state.phase = "done"
            log_store.log_verification(
                node="request_confirmation",
                platform=st.session_state.platform,
                details={"confirmed": False, "reason": "user declined"},
            )
            st.rerun()


def _section_booking():
    """Resume graph after the interrupt with user confirmation."""
    with st.spinner("Booking your ride…"):
        # Resume graph — interrupt was before request_confirmation node
        # First let request_confirmation run (it will interrupt again internally
        # if needed, but we compiled with interrupt_before so we feed confirmation
        # via Command resume after the node is invoked once)
        _run_graph(Command(resume={"confirmed": True, "selected_index": st.session_state.selected_index}))

    if st.session_state.error:
        st.error(f"Booking failed: {st.session_state.error}")
        if st.button("Back to options"):
            st.session_state.phase = "confirming"
            st.session_state.error = None
            st.rerun()
        return

    st.session_state.phase = "tracking"
    st.rerun()


def _section_tracking():
    booking = st.session_state.booking
    if not booking:
        st.error("No active booking found.")
        return

    status = st.session_state.ride_status or booking.status
    emoji = STATUS_EMOJI.get(status, "🚗")
    label = STATUS_LABEL.get(status, status)

    st.markdown(f"### 3. Your ride  {emoji} {label}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Driver:** {booking.driver_name}")
        st.markdown(f"**Phone:** {booking.driver_phone}")
        st.markdown(f"**Vehicle:** {booking.vehicle_make} {booking.vehicle_model}")
        st.markdown(f"**Plate:** `{booking.license_plate}`")
    with col2:
        st.markdown(f"**Ride:** {booking.option.product_name}")
        st.markdown(f"**Estimated price:** {booking.option.price_display}")
        st.markdown(f"**From:** {booking.pickup}")
        st.markdown(f"**To:** {booking.dropoff}")

    # Progress bar for visual status
    _progress = {
        "PROCESSING": 0.05, "DRIVER_ASSIGNED": 0.15, "EN_ROUTE": 0.35,
        "ARRIVED": 0.5, "IN_PROGRESS": 0.75, "COMPLETED": 1.0,
        "CANCELLED": 0.0, "NO_DRIVERS": 0.0,
    }
    st.progress(_progress.get(status, 0.1))

    # Terminal states
    if status in ("COMPLETED", "CANCELLED", "NO_DRIVERS"):
        st.success("Trip complete!" if status == "COMPLETED" else label)
        if st.button("Book another ride"):
            _reset()
        return

    # Actions
    col_refresh, col_cancel, _ = st.columns([1, 1, 4])
    with col_refresh:
        if st.button("Refresh status 🔄"):
            _refresh_status()
    with col_cancel:
        if st.button("Cancel ride ❌"):
            st.session_state.phase = "cancelling"
            st.rerun()

    # Auto-refresh hint
    st.caption("Status refreshes on each page interaction. Click 'Refresh status' to poll now.")


def _section_cancelling():
    with st.spinner("Cancelling ride…"):
        booking = st.session_state.booking
        if booking:
            from adapters.registry import get_adapter
            adapter = get_adapter(st.session_state.platform)
            result = adapter.cancel_ride(booking.booking_id)
            log_store.log_executed(
                node="cancel_ride",
                platform=st.session_state.platform,
                payload={"booking_id": booking.booking_id},
                response={"success": result.success, "message": result.message, "fee": result.cancellation_fee},
            )
            st.session_state.ride_status = "CANCELLED"
            st.session_state.error = None

    st.session_state.phase = "tracking"
    st.rerun()


def _section_done():
    st.info("Booking cancelled — no ride was booked.")
    if st.button("Start over"):
        _reset()


def _refresh_status():
    booking = st.session_state.booking
    if not booking:
        return
    from adapters.registry import get_adapter
    adapter = get_adapter(st.session_state.platform)
    try:
        status = adapter.get_ride_status(booking.booking_id)
        st.session_state.ride_status = status
        log_store.append(log_store.ActionLogEntry(
            action_type="RESULT",
            node="track_ride",
            platform=st.session_state.platform,
            payload={"booking_id": booking.booking_id},
            response={"status": status},
            success=True,
        ))
    except Exception as exc:
        st.session_state.error = str(exc)


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

st.title("🚗 ChatFFeur")
st.caption("AI-powered ride booking · confirm before booking · full action log")

main_col, log_col = st.columns([3, 2])

with main_col:
    phase = st.session_state.phase

    if phase == "input":
        _section_input()
    elif phase == "searching":
        _section_searching()
    elif phase == "confirming":
        _section_confirming()
    elif phase == "booking":
        _section_booking()
    elif phase == "tracking":
        _section_tracking()
    elif phase == "cancelling":
        _section_cancelling()
    elif phase == "done":
        _section_done()

    if st.session_state.error and phase not in ("searching", "booking", "confirming"):
        st.error(st.session_state.error)

with log_col:
    _render_log_panel()
    if st.session_state.phase not in ("input", "searching"):
        if st.button("New booking", key="new_booking_btn"):
            _reset()
