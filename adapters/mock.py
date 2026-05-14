from __future__ import annotations

import random
import time
import uuid
from typing import ClassVar

from adapters.base import (
    Booking,
    CancelResult,
    RideOption,
    RidePlatformAdapter,
    RideStatus,
)

_MOCK_PRODUCTS = [
    {"name": "UberX",    "low": 10, "high": 16, "eta": 7,  "duration": 20},
    {"name": "Comfort",  "low": 18, "high": 24, "eta": 5,  "duration": 20},
    {"name": "Black",    "low": 35, "high": 45, "eta": 3,  "duration": 20},
    {"name": "UberXL",   "low": 22, "high": 30, "eta": 9,  "duration": 20},
]

_STATUS_PROGRESSION: list[RideStatus] = [
    "DRIVER_ASSIGNED",
    "EN_ROUTE",
    "ARRIVED",
    "IN_PROGRESS",
    "COMPLETED",
]

_DRIVERS = [
    ("Alex M.", "+1-555-0101", "Toyota", "Camry"),
    ("Sam K.",  "+1-555-0102", "Honda",  "Civic"),
    ("Jordan P.", "+1-555-0103", "Ford", "Fusion"),
]


class MockAdapter(RidePlatformAdapter):
    """
    Fully in-memory mock for development and demos.
    No network calls; simulates realistic data and status progression.
    """

    platform_name: ClassVar[str] = "mock"

    # Tracks bookings and their simulated progression
    _bookings: ClassVar[dict[str, dict]] = {}

    def get_price_estimates(self, pickup: str, dropoff: str) -> list[RideOption]:
        surge = random.choice([1.0, 1.0, 1.0, 1.2, 1.5])
        options = []
        for i, p in enumerate(_MOCK_PRODUCTS):
            low = round(p["low"] * surge, 2)
            high = round(p["high"] * surge, 2)
            options.append(
                RideOption(
                    option_id=f"mock-{i}",
                    product_name=p["name"],
                    price_low=low,
                    price_high=high,
                    currency="$",
                    eta_minutes=p["eta"] + random.randint(-1, 2),
                    duration_minutes=p["duration"] + random.randint(-3, 5),
                    surge_multiplier=surge,
                )
            )
        return options

    def book_ride(self, option: RideOption, pickup: str, dropoff: str) -> Booking:
        booking_id = f"mock-booking-{uuid.uuid4().hex[:8]}"
        driver_name, driver_phone, v_make, v_model = random.choice(_DRIVERS)
        plate = f"{random.choice('ABCDEFGH')}{random.choice('ABCDEFGH')}{random.randint(100,999)}"

        self._bookings[booking_id] = {
            "created_at": time.time(),
            "status_index": 0,
        }

        return Booking(
            booking_id=booking_id,
            platform=self.platform_name,
            option=option,
            pickup=pickup,
            dropoff=dropoff,
            driver_name=driver_name,
            driver_phone=driver_phone,
            vehicle_make=v_make,
            vehicle_model=v_model,
            license_plate=plate,
            status="DRIVER_ASSIGNED",
        )

    def get_ride_status(self, booking_id: str) -> RideStatus:
        record = self._bookings.get(booking_id)
        if record is None:
            return "CANCELLED"

        elapsed = time.time() - record["created_at"]
        # Advance through statuses every ~8 seconds for demo speed
        index = min(int(elapsed / 8), len(_STATUS_PROGRESSION) - 1)
        record["status_index"] = index
        return _STATUS_PROGRESSION[index]

    def cancel_ride(self, booking_id: str) -> CancelResult:
        record = self._bookings.pop(booking_id, None)
        if record is None:
            return CancelResult(success=False, message="Booking not found.")

        # Cancellation fee if driver already assigned
        elapsed = time.time() - record["created_at"]
        fee = 5.0 if elapsed > 5 else 0.0
        return CancelResult(
            success=True,
            message="Ride cancelled." + (f" Cancellation fee: ${fee:.2f}" if fee else ""),
            cancellation_fee=fee,
        )
