from __future__ import annotations

from typing import Annotated, Any
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from adapters.base import Booking, RideOption, RideStatus


class AgentState(BaseModel):
    # Conversation messages (LangGraph managed)
    messages: Annotated[list[Any], add_messages] = Field(default_factory=list)

    # Ride request inputs
    pickup: str = ""
    dropoff: str = ""
    platform: str = "mock"

    # Discovered options
    options: list[RideOption] = Field(default_factory=list)

    # User-selected option index into `options`
    selected_option_index: int | None = None

    # Active booking (set after book_ride node)
    booking: Booking | None = None

    # Latest polled ride status
    ride_status: RideStatus | None = None

    # Graph flow control
    awaiting_confirmation: bool = False
    user_confirmed: bool | None = None  # None = not answered yet
    cancel_requested: bool = False

    # Error message if any node fails
    error: str | None = None

    class Config:
        arbitrary_types_allowed = True
