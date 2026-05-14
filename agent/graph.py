from __future__ import annotations

import os
os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "false")

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from agent.state import AgentState
from agent.nodes import (
    discover_options,
    compare_prices,
    request_confirmation,
    book_ride,
    track_ride,
    cancel_ride,
)


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def _route_after_confirmation(state: AgentState) -> str:
    if state.user_confirmed is True:
        return "book_ride"
    return END  # user declined — done


def _route_after_booking(state: AgentState) -> str:
    if state.error:
        return END
    return "track_ride"


def _route_after_tracking(state: AgentState) -> str:
    terminal = {"COMPLETED", "CANCELLED", "NO_DRIVERS"}
    if state.ride_status in terminal or state.error:
        return END
    if state.cancel_requested:
        return "cancel_ride"
    # UI will re-invoke graph to poll again; return END to yield control
    return END


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    builder = StateGraph(AgentState)

    builder.add_node("discover_options",     discover_options)
    builder.add_node("compare_prices",       compare_prices)
    builder.add_node("request_confirmation", request_confirmation)
    builder.add_node("book_ride",            book_ride)
    builder.add_node("track_ride",           track_ride)
    builder.add_node("cancel_ride",          cancel_ride)

    builder.set_entry_point("discover_options")

    builder.add_edge("discover_options",     "compare_prices")
    builder.add_edge("compare_prices",       "request_confirmation")

    builder.add_conditional_edges(
        "request_confirmation",
        _route_after_confirmation,
        {"book_ride": "book_ride", END: END},
    )

    builder.add_conditional_edges(
        "book_ride",
        _route_after_booking,
        {"track_ride": "track_ride", END: END},
    )

    builder.add_conditional_edges(
        "track_ride",
        _route_after_tracking,
        {"cancel_ride": "cancel_ride", END: END},
    )

    builder.add_edge("cancel_ride", END)

    return builder


# Compile with MemorySaver so the graph can checkpoint for interrupt() support
_checkpointer = MemorySaver()
graph = build_graph().compile(checkpointer=_checkpointer, interrupt_before=["request_confirmation"])
