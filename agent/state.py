from typing import Annotated, Optional, Union

from langgraph.graph.message import add_messages
from pydantic import BaseModel

from adapters.uber_guest_client import UberGuestInfo


def _append_logs(existing: list, new: list) -> list:
    return existing + new


class AgentState(BaseModel):
    # Conversation history — add_messages merges new messages by id.
    messages: Annotated[list, add_messages] = []

    # Name of the active platform adapter, e.g. "uber".
    platform_adapter: Optional[str] = None

    # Raw results returned by the last search_rides call.
    search_results: list[dict] = []

    # Ride the agent selected from search_results and presented to the user.
    suggested_ride: dict = {}

    # Set to True only after the user explicitly confirms via the UI interrupt.
    # book_ride checks this before proceeding.
    ride_confirmed: Optional[bool] = None

    # Confirmation payload returned by adapter.book_ride() on success.
    booked_ride: Optional[dict] = {}

    # Guest personal details required by the Uber Guest Rides API.
    # Accepts either a UberGuestInfo instance or a plain dict with the same keys.
    guest_info: Optional[Union[UberGuestInfo, dict]] = None

    # Append-only log of every tool action.  Uses a custom reducer so each
    # tool call can append entries without overwriting previous ones.
    action_log: Annotated[list, _append_logs] = []
