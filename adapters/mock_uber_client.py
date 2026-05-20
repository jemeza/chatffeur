"""
Mock Uber Guest Rides API client — no network calls, no credentials required.

Trip lifecycle (time-based auto-progression):
  0–8s    → processing
  8–25s   → accepted
  25–60s  → arriving
  60–180s → in_progress
  180s+   → completed
"""
import math
import random
import time
import uuid

from pydantic import BaseModel


_PRODUCTS = [
    {
        "product_id": "mock-uberx-0001",
        "display_name": "UberX",
        "description": "Affordable, everyday rides",
        "capacity": 4,
        "_base_price": (12.0, 18.0),
        "_eta_range": (4, 10),
    },
    {
        "product_id": "mock-comfort-0001",
        "display_name": "Uber Comfort",
        "description": "Newer cars with extra legroom",
        "capacity": 4,
        "_base_price": (18.0, 26.0),
        "_eta_range": (6, 14),
    },
    {
        "product_id": "mock-uberxl-0001",
        "display_name": "UberXL",
        "description": "Affordable rides for groups up to 6",
        "capacity": 6,
        "_base_price": (22.0, 34.0),
        "_eta_range": (8, 18),
    },
    {
        "product_id": "mock-black-0001",
        "display_name": "Uber Black",
        "description": "Premium rides in luxury cars",
        "capacity": 4,
        "_base_price": (40.0, 65.0),
        "_eta_range": (3, 8),
    },
]

_DRIVER_POOL = [
    {"name": "Marcus T.", "rating": "4.95", "phone_number": "+15550001111"},
    {"name": "Priya S.", "rating": "4.88", "phone_number": "+15550002222"},
    {"name": "Derek W.", "rating": "4.92", "phone_number": "+15550003333"},
    {"name": "Amara N.", "rating": "4.79", "phone_number": "+15550004444"},
    {"name": "Carlos R.", "rating": "4.97", "phone_number": "+15550005555"},
]

_VEHICLE_POOL = [
    {"make": "Toyota", "model": "Camry", "year": 2022, "license_plate": "ABC1234"},
    {"make": "Honda", "model": "Accord", "year": 2023, "license_plate": "XYZ5678"},
    {"make": "Tesla", "model": "Model 3", "year": 2023, "license_plate": "EV98765"},
    {"make": "Chevrolet", "model": "Suburban",
        "year": 2021, "license_plate": "SUV4321"},
    {"make": "Mercedes", "model": "E-Class",
        "year": 2022, "license_plate": "LUX8899"},
]

_STATUS_TIMELINE = [
    (0, "processing"),
    (3, "accepted"),
    (8, "arriving"),
    (15, "in_progress"),
    (30, "completed"),
]

# Phase boundary constants (seconds) — also used for driver interpolation.
_T_ARRIVING = 8
_T_IN_PROGRESS = 15
_T_COMPLETED = 30


def _pick_status(elapsed: float) -> str:
    status = "processing"
    for threshold, s in _STATUS_TIMELINE:
        if elapsed >= threshold:
            status = s
    return status


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * max(0.0, min(1.0, t))


def _compute_bearing(from_lat: float, from_lon: float, to_lat: float, to_lon: float) -> int:
    angle = math.degrees(math.atan2(to_lon - from_lon, to_lat - from_lat))
    return int(angle % 360)


def _smooth_driver_position(trip: dict, elapsed: float, status: str) -> tuple[float, float]:
    """Return (lat, lon) that smoothly moves driver from start → pickup → dropoff."""
    start = trip.get("driver_start", trip["pickup"])
    pickup = trip["pickup"]
    dropoff = trip["dropoff"]

    if status in ("processing", "accepted"):
        return start["latitude"], start["longitude"]
    elif status == "arriving":
        t = (elapsed - _T_ARRIVING) / (_T_IN_PROGRESS - _T_ARRIVING)
        return (
            _lerp(start["latitude"], pickup["latitude"], t),
            _lerp(start["longitude"], pickup["longitude"], t),
        )
    elif status == "in_progress":
        t = (elapsed - _T_IN_PROGRESS) / (_T_COMPLETED - _T_IN_PROGRESS)
        return (
            _lerp(pickup["latitude"], dropoff["latitude"], t),
            _lerp(pickup["longitude"], dropoff["longitude"], t),
        )
    else:
        return dropoff["latitude"], dropoff["longitude"]


class UberGuestInfo(BaseModel):
    """Personal details required to create a guest ride booking."""

    first_name: str
    last_name: str
    email: str
    phone_number: str  # E.164 format recommended: +12125551234


class MockUberGuestRidesClient:
    """Drop-in replacement for UberGuestRidesClient that never hits the network."""

    def __init__(self, *args, **kwargs) -> None:
        # Accept same constructor signature; credentials are ignored.
        self._trips: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Auth stubs (no-ops)
    # ------------------------------------------------------------------

    def _fetch_token(self) -> None:
        pass

    def _get_token(self) -> str:
        return "mock-token"

    # ------------------------------------------------------------------
    # Estimates
    # ------------------------------------------------------------------

    def get_estimates(self, pickup: dict, dropoff: dict) -> dict:
        """Return a realistic set of ride-type price estimates."""
        # Pick a single surge factor for this request to simulate area-wide demand.
        surge = round(random.choice([1.0, 1.0, 1.2, 1.5, 1.8, 2.0]), 1)

        prices = []
        for product in _PRODUCTS:
            low, high = product["_base_price"]
            low = round(random.uniform(low * 0.9, low * 1.1) * surge, 2)
            high = round(random.uniform(high * 0.9, high * 1.1) * surge, 2)
            if low > high:
                low, high = high, low
            eta_min, eta_max = product["_eta_range"]

            prices.append(
                {
                    "product_id": product["product_id"],
                    "display_name": product["display_name"],
                    "description": product["description"],
                    "capacity": product["capacity"],
                    "no_cars_available": False,
                    "fare": {
                        "fare_id": f"fare-{uuid.uuid4().hex[:12]}",
                        "low_value": str(low),
                        "high_value": str(high),
                        "display": f"${(low + high) / 2:.0f}",
                        "currency_code": "USD",
                        "surge_multiplier": surge,
                    },
                    "trip": {
                        "duration_estimate": random.randint(eta_min, eta_max),
                    },
                }
            )

        return {"prices": prices}

    # ------------------------------------------------------------------
    # Create trip
    # ------------------------------------------------------------------

    def create_trip(
        self,
        guest: UberGuestInfo,
        product_id: str,
        fare_id: str,
        pickup: dict,
        dropoff: dict,
        price_estimate: str = "$0.00",
    ) -> dict:
        """Instantiate a mock trip and return its initial state."""
        request_id = f"mock-trip-{uuid.uuid4().hex[:16]}"
        driver = random.choice(_DRIVER_POOL)
        vehicle = random.choice(_VEHICLE_POOL)

        # Locate the matching product so we can echo back its name.
        product_name = next(
            (p["display_name"]
             for p in _PRODUCTS if p["product_id"] == product_id),
            "UberX",
        )

        trip_record = {
            "request_id": request_id,
            "status": "processing",
            "product_id": product_id,
            "product_name": product_name,
            "guest": guest.model_dump(),
            "pickup": pickup,
            "dropoff": dropoff,
            "fare_id": fare_id,
            "price_estimate": price_estimate,
            "driver": driver,
            "vehicle": vehicle,
            "pickup_estimate": random.randint(3, 12),
            "created_at": time.monotonic(),
            "driver_start": {
                "latitude": pickup["latitude"] + random.uniform(-0.02, 0.02),
                "longitude": pickup["longitude"] + random.uniform(-0.02, 0.02),
            },
        }
        self._trips[request_id] = trip_record

        return self._format_trip(trip_record)

    # ------------------------------------------------------------------
    # Get trip
    # ------------------------------------------------------------------

    def get_trip(self, request_id: str) -> dict:
        """Return the current trip state with auto-progressed status."""
        trip = self._trips.get(request_id)
        if trip is None:
            # Return a minimal "not found" shape rather than raising so the
            # adapter can surface a readable error to the LLM.
            return {"request_id": request_id, "status": "not_found"}

        elapsed = time.monotonic() - trip["created_at"]
        trip["status"] = _pick_status(elapsed)
        return self._format_trip(trip)

    def list_trips(self) -> dict:
        """Return all mock trips."""
        return {"trips": [self._format_trip(t) for t in self._trips.values()]}

    # ------------------------------------------------------------------
    # Cancel trip
    # ------------------------------------------------------------------

    def cancel_trip(self, request_id: str) -> dict:
        """Cancel a mock trip; applies a fee if driver was already on the way."""
        trip = self._trips.get(request_id)
        if trip is None:
            return {"request_id": request_id, "status": "not_found"}

        elapsed = time.monotonic() - trip["created_at"]
        current_status = _pick_status(elapsed)

        # No fee before accepted; $5 once driver is arriving; full trip price if in progress.
        cancellation_fee = "$0.00"
        if current_status == "arriving":
            cancellation_fee = "$5.00"
        elif current_status == "in_progress":
            cancellation_fee = trip.get("price_estimate", "$0.00")

        trip["status"] = "rider_canceled"
        return {
            "request_id": request_id,
            "status": "rider_canceled",
            "cancellation_fee": cancellation_fee,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _format_trip(self, trip: dict) -> dict:
        """Render a trip record into the shape the real API returns."""
        elapsed = time.monotonic() - trip["created_at"]
        status = trip["status"]

        # Smoothly interpolate driver position across the trip lifecycle.
        drv_lat, drv_lon = _smooth_driver_position(trip, elapsed, status)

        # Compute bearing toward the next waypoint.
        if status in ("processing", "accepted", "arriving"):
            bearing = _compute_bearing(
                drv_lat, drv_lon,
                trip["pickup"]["latitude"], trip["pickup"]["longitude"],
            )
        else:
            bearing = _compute_bearing(
                drv_lat, drv_lon,
                trip["dropoff"]["latitude"], trip["dropoff"]["longitude"],
            )

        return {
            "request_id": trip["request_id"],
            "status": status,
            "driver": {
                "name": trip["driver"]["name"],
                "rating": trip["driver"]["rating"],
                "phone_number": trip["driver"]["phone_number"],
                "vehicle_year_make_model": (
                    f"{trip['vehicle']['year']} {trip['vehicle']['make']} "
                    f"{trip['vehicle']['model']}"
                ),
            },
            "vehicle": {
                "make": trip["vehicle"]["make"],
                "model": trip["vehicle"]["model"],
                "year": trip["vehicle"]["year"],
                "license_plate": trip["vehicle"]["license_plate"],
            },
            "location": {
                "latitude": round(drv_lat, 6),
                "longitude": round(drv_lon, 6),
                "bearing": bearing,
            },
            "pickup": trip["pickup"],
            "destination": {
                **trip["dropoff"],
            },
            # Seconds remaining until driver reaches pickup; 0 once picked up.
            "pickup_estimate": (
                0 if status in ("in_progress", "completed", "rider_canceled")
                else max(0, int(_T_IN_PROGRESS - elapsed))
            ),
            # Seconds remaining until dropoff; only meaningful during in_progress.
            "dropoff_estimate": (
                max(0, int(_T_COMPLETED - elapsed)) if status == "in_progress"
                else 0
            ),
            "surge_multiplier": 1.0,
        }
