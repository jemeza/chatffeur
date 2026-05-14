"""
Chatffeur — Streamlit UI

Layout
------
  Left (main):  Chat messages + ride-confirmation card (when agent interrupts)
  Right (sidebar): Live action log

Session state keys
------------------
  thread_id          str   — unique LangGraph thread identifier
  graph              —     — compiled LangGraph graph (built once per session)
  display_messages   list  — [{role, content}] shown in the chat pane
  graph_msg_idx      int   — how many graph messages we've already surfaced
  awaiting_confirm   bool  — True while the agent is waiting for confirm/reject
  interrupt_data     dict  — {suggested_ride, reasoning} from the interrupt call
  action_log         list  — mirrors AgentState.action_log for the sidebar
"""

import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from agent.graph import build_graph

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Chatffeur 🚗",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session initialisation
# ---------------------------------------------------------------------------


def _init():
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "graph" not in st.session_state:
        st.session_state.graph, _ = build_graph()
    if "display_messages" not in st.session_state:
        st.session_state.display_messages = []
    if "graph_msg_idx" not in st.session_state:
        st.session_state.graph_msg_idx = 0
    if "awaiting_confirm" not in st.session_state:
        st.session_state.awaiting_confirm = False
    if "interrupt_data" not in st.session_state:
        st.session_state.interrupt_data = None
    if "action_log" not in st.session_state:
        st.session_state.action_log = []


_init()


def _config():
    return {"configurable": {"thread_id": st.session_state.thread_id}}


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------


def _sync_from_state():
    """Pull new AI messages + action_log + interrupt status from graph state."""
    state = st.session_state.graph.get_state(_config())
    graph_messages = state.values.get("messages", [])

    # Surface new AI text messages (skip pure tool-call placeholders)
    for msg in graph_messages[st.session_state.graph_msg_idx:]:
        if isinstance(msg, AIMessage) and msg.content:
            st.session_state.display_messages.append(
                {"role": "assistant", "content": msg.content}
            )
    st.session_state.graph_msg_idx = len(graph_messages)

    # Sync action log
    if state.values.get("action_log"):
        st.session_state.action_log = state.values["action_log"]

    # Detect interrupt
    if state.next:
        for task in state.tasks:
            interrupts = getattr(task, "interrupts", ())
            if interrupts:
                st.session_state.awaiting_confirm = True
                st.session_state.interrupt_data = interrupts[0].value
                return

    st.session_state.awaiting_confirm = False
    st.session_state.interrupt_data = None


def _run(input_data):
    """Stream the graph with the given input, then sync state."""
    try:
        for _ in st.session_state.graph.stream(
            input_data, _config(), stream_mode="values"
        ):
            pass
    except Exception as exc:
        st.session_state.display_messages.append(
            {"role": "assistant", "content": f"⚠️ An error occurred: {exc}"}
        )
        return
    _sync_from_state()


def _reset():
    for key in [
        "thread_id",
        "graph",
        "display_messages",
        "graph_msg_idx",
        "awaiting_confirm",
        "interrupt_data",
        "action_log",
    ]:
        st.session_state.pop(key, None)
    st.rerun()


# ---------------------------------------------------------------------------
# Sidebar — Action Log
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("📋 Action Log")

    if st.button("🔄 New Conversation", use_container_width=True):
        _reset()

    st.divider()

    if st.session_state.action_log:
        for entry in reversed(st.session_state.action_log):
            action_type = entry.get("action_type", "action")
            outcome = entry.get("outcome", "")

            if "error" in outcome or "blocked" in outcome:
                icon = "🔴"
            elif any(k in outcome for k in ("success", "confirmed", "found")):
                icon = "🟢"
            elif "awaiting" in outcome or "rejected" in outcome:
                icon = "🟡"
            else:
                icon = "⚪"

            with st.expander(f"{icon} {action_type}", expanded=False):
                ts = entry.get("timestamp", "")[:19].replace("T", " ")
                st.caption(ts + " UTC")
                st.write(f"**Outcome:** `{outcome}`")

                cols = st.columns(3)
                with cols[0]:
                    st.write("**Requested**")
                    st.json(entry.get("requested", {}), expanded=False)
                with cols[1]:
                    st.write("**Verified**")
                    st.json(entry.get("verified", {}), expanded=False)
                with cols[2]:
                    st.write("**Executed**")
                    st.json(entry.get("executed", {}), expanded=False)
    else:
        st.caption("No actions recorded yet.")
        st.caption("Start by asking to book a ride!")

# ---------------------------------------------------------------------------
# Main — Chat
# ---------------------------------------------------------------------------

st.title("🚗 Chatffeur")
st.caption("Your AI-powered ride-booking assistant — powered by Claude + LangGraph")

# Welcome hint (shown only before the first message)
if not st.session_state.display_messages:
    st.info(
        '👋 Try: **"Book an Uber from Times Square to JFK airport"**\n\n'
        "The agent will search for options, recommend one, and ask for your "
        "confirmation before booking anything."
    )

# Render chat history
for msg in st.session_state.display_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------------------------------------------------------
# Ride confirmation card (shown when suggest_ride interrupted the graph)
# ---------------------------------------------------------------------------

if st.session_state.awaiting_confirm and st.session_state.interrupt_data:
    data = st.session_state.interrupt_data
    ride = data.get("suggested_ride", {})
    reasoning = data.get("reasoning", "")

    st.divider()

    with st.container(border=True):
        st.subheader("🚕 Confirm Your Ride")

        m1, m2, m3 = st.columns(3)
        m1.metric("Ride type", ride.get("display_name", "—"))
        m2.metric("Price estimate", ride.get("price_estimate", "—"))
        m3.metric(
            "ETA",
            f"~{ride.get('duration_estimate_minutes', '?')} min",
        )

        st.info(f"**Why this ride?** {reasoning}")

        st.caption(
            f"📍 **From:** {ride.get('pickup', '—')}  \n"
            f"🏁 **To:** {ride.get('dropoff', '—')}  \n"
            f"👥 **Capacity:** up to {ride.get('capacity', '?')} passengers  \n"
            f"💳 **Surge:** {ride.get('surge_multiplier', 1.0)}×"
        )

        st.divider()

        btn_confirm, btn_reject, _ = st.columns([1, 1, 2])

        if btn_confirm.button("✅ Confirm booking", type="primary", use_container_width=True):
            st.session_state.display_messages.append(
                {
                    "role": "user",
                    "content": f"✅ I confirm the **{ride.get('display_name')}** booking.",
                }
            )
            st.session_state.awaiting_confirm = False
            st.session_state.interrupt_data = None
            _run(Command(resume=True))
            st.rerun()

        if btn_reject.button("❌ Reject", type="secondary", use_container_width=True):
            st.session_state.display_messages.append(
                {
                    "role": "user",
                    "content": f"❌ I rejected the **{ride.get('display_name')}** suggestion.",
                }
            )
            st.session_state.awaiting_confirm = False
            st.session_state.interrupt_data = None
            _run(Command(resume=False))
            st.rerun()

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

if not st.session_state.awaiting_confirm:
    if prompt := st.chat_input("e.g. 'Book an Uber from Main St to JFK airport'"):
        st.session_state.display_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                _run({"messages": [HumanMessage(content=prompt)]})

        st.rerun()
else:
    st.chat_input(
        "Please confirm or reject the ride suggestion above ☝️",
        disabled=True,
    )
