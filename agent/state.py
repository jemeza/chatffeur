from typing import Annotated, Optional

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


def _append_logs(existing: list, new: list) -> list:
    return existing + new


class AgentState(TypedDict):
    # Conversation history — add_messages merges new messages by id.
    messages: Annotated[list, add_messages]

    # Name of the active platform adapter, e.g. "uber".
    platform_adapter: str

    # Raw results returned by the last search_rides call.
    search_results: list[dict]

    # Ride the agent selected from search_results and presented to the user.
    suggested_ride: Optional[dict]

    # Set to True only after the user explicitly confirms via the UI interrupt.
    # book_ride checks this before proceeding.
    ride_confirmed: bool

    # Confirmation payload returned by adapter.book_ride() on success.
    booked_ride: Optional[dict]

    # Append-only log of every tool action.  Uses a custom reducer so each
    # tool call can append entries without overwriting previous ones.
    action_log: Annotated[list, _append_logs]
