from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal


@dataclass
class RideOption:
    option_id: str
    product_name: str          # "UberX", "Comfort", "Black", "Lyft", "Lyft XL" …
    price_low: float
    price_high: float
    currency: str
    eta_minutes: int           # driver ETA to pickup
    duration_minutes: int      # estimated trip duration
    surge_multiplier: float = 1.0

    @property
    def price_display(self) -> str:
        if self.price_low == self.price_high:
            return f"{self.currency}{self.price_low:.2f}"
        return f"{self.currency}{self.price_low:.2f}–{self.price_high:.2f}"


@dataclass
class Booking:
    booking_id: str
    platform: str
    option: RideOption
    pickup: str
    dropoff: str
    driver_name: str
    driver_phone: str
    vehicle_make: str
    vehicle_model: str
    license_plate: str
    status: RideStatus


RideStatus = Literal[
    "PROCESSING",
    "DRIVER_ASSIGNED",
    "EN_ROUTE",        # driver heading to pickup
    "ARRIVED",         # driver at pickup
    "IN_PROGRESS",     # trip underway
    "COMPLETED",
    "CANCELLED",
    "NO_DRIVERS",
]


@dataclass
class CancelResult:
    success: bool
    message: str
    cancellation_fee: float = 0.0


class RidePlatformAdapter(ABC):
    """
    Implement this interface for any ride platform.
    Swapping platforms = swapping the concrete class; nothing else changes.
    """

    platform_name: str  # override in subclass, e.g. "uber"

    @abstractmethod
    def get_price_estimates(self, pickup: str, dropoff: str) -> list[RideOption]:
        """Return available ride options with price estimates."""

    @abstractmethod
    def book_ride(self, option: RideOption, pickup: str, dropoff: str) -> Booking:
        """Book the selected ride option. Returns a Booking with initial status."""

    @abstractmethod
    def get_ride_status(self, booking_id: str) -> RideStatus:
        """Poll the current status of an active booking."""

    @abstractmethod
    def cancel_ride(self, booking_id: str) -> CancelResult:
        """Cancel an active booking."""
