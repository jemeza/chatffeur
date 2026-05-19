from typing import Annotated, Optional, Union

from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict

from adapters.mock_uber_client import UberGuestInfo


def _append_logs(existing: list, new: list) -> list:
    return existing + new


def _merge_dicts(existing: dict, new: dict) -> dict:
    return {**existing, **new}


class AgentState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
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

    # Live adapter instances keyed by platform name.  Storing them in state
    # (rather than a module-level dict) keeps per-session trip data isolated
    # across concurrent sessions while surviving across tool calls within one.
    adapter_instances: Annotated[dict, _merge_dicts] = {}
