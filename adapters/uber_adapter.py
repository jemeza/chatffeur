import uuid
from datetime import datetime, timedelta

from adapters.ride_adapter import RidePlatformAdapter


class UberRideAdapter(RidePlatformAdapter):
    """
    Uber implementation of RidePlatformAdapter.

    NOTE: Uber's public rider API (v1.2) was deprecated and is no longer
    available to new developers.  All methods below are placeholders that
    return realistic fake data.  Replace each # TODO block with real HTTP
    calls once API access is restored or an alternative SDK is available.

    Relevant Uber API references (for when this changes):
      POST https://api.uber.com/v1.2/requests          — create a ride
      GET  https://api.uber.com/v1.2/requests/{id}     — get ride status
      DELETE https://api.uber.com/v1.2/requests/{id}   — cancel a ride
      GET  https://api.uber.com/v1.2/estimates/price   — price estimates
      GET  https://api.uber.com/v1.2/products          — available products
    """

    platform_name = "uber"

    def __init__(self, server_token: str | None = None):
        self.server_token = server_token
        # TODO: initialise an authenticated Uber HTTP client here
        # import requests
        # self._session = requests.Session()
        # self._session.headers["Authorization"] = f"Token {server_token}"
        # self._base_url = "https://api.uber.com/v1.2"

    # ------------------------------------------------------------------
    # search_rides
    # ------------------------------------------------------------------

    def search_rides(self, pickup: str, dropoff: str) -> list[dict]:
        # TODO: geocode pickup/dropoff, then call:
        #   GET /v1.2/products?latitude={lat}&longitude={lon}
        #   GET /v1.2/estimates/price?start_latitude=...&end_latitude=...
        # Merge results by product_id and return the list below.
        return [
            {
                "product_id": "a1111c8c-c720-46c3-8534-2fcdd730040d",
                "display_name": "UberX",
                "description": "Affordable everyday rides",
                "pickup": pickup,
                "dropoff": dropoff,
                "price_estimate": "$18–24",
                "price_low": 18.00,
                "price_high": 24.00,
                "duration_estimate_minutes": 22,
                "surge_multiplier": 1.0,
                "currency": "USD",
                "capacity": 4,
            },
            {
                "product_id": "b1b4a4b4-c720-46c3-8534-2fcdd730040d",
                "display_name": "UberXL",
                "description": "Affordable rides for groups up to 6",
                "pickup": pickup,
                "dropoff": dropoff,
                "price_estimate": "$26–35",
                "price_low": 26.00,
                "price_high": 35.00,
                "duration_estimate_minutes": 24,
                "surge_multiplier": 1.0,
                "currency": "USD",
                "capacity": 6,
            },
            {
                "product_id": "c1c1c8c8-c720-46c3-8534-2fcdd730040d",
                "display_name": "Uber Black",
                "description": "Premium rides in luxury cars",
                "pickup": pickup,
                "dropoff": dropoff,
                "price_estimate": "$55–70",
                "price_low": 55.00,
                "price_high": 70.00,
                "duration_estimate_minutes": 20,
                "surge_multiplier": 1.0,
                "currency": "USD",
                "capacity": 4,
            },
        ]

    # ------------------------------------------------------------------
    # book_ride
    # ------------------------------------------------------------------

    def book_ride(self, ride_option: dict) -> dict:
        # TODO: replace with:
        #   POST /v1.2/requests
        #   body = {
        #       "product_id": ride_option["product_id"],
        #       "start_latitude": <geocoded pickup lat>,
        #       "start_longitude": <geocoded pickup lon>,
        #       "end_latitude": <geocoded dropoff lat>,
        #       "end_longitude": <geocoded dropoff lon>,
        #   }
        ride_id = str(uuid.uuid4())
        pickup_eta = datetime.now() + timedelta(minutes=5)
        return {
            "ride_id": ride_id,
            "status": "processing",
            "product": ride_option["display_name"],
            "pickup": ride_option["pickup"],
            "dropoff": ride_option["dropoff"],
            "price_estimate": ride_option["price_estimate"],
            "driver": {
                "name": "Michael S.",
                "rating": 4.92,
                "vehicle": "Toyota Camry – Silver",
                "license_plate": "ABC1234",
            },
            "pickup_eta_minutes": 5,
            "pickup_eta_iso": pickup_eta.isoformat(),
        }

    # ------------------------------------------------------------------
    # track_ride
    # ------------------------------------------------------------------

    def track_ride(self, ride_id: str) -> dict:
        # TODO: replace with:
        #   GET /v1.2/requests/{ride_id}
        return {
            "ride_id": ride_id,
            "status": "accepted",
            "driver_location": {"latitude": 40.7128, "longitude": -74.0060},
            "eta_minutes": 3,
        }

    # ------------------------------------------------------------------
    # cancel_ride
    # ------------------------------------------------------------------

    def cancel_ride(self, ride_id: str) -> dict:
        # TODO: replace with:
        #   DELETE /v1.2/requests/{ride_id}
        return {
            "ride_id": ride_id,
            "status": "cancelled",
            "cancellation_fee": "$0.00",
        }
