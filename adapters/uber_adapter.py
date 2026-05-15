import os

from adapters.geocoder import geocode
from adapters.ride_adapter import RidePlatformAdapter
from adapters.uber_guest_client import UberGuestInfo, UberGuestRidesClient


class UberRideAdapter(RidePlatformAdapter):
    """
    Uber implementation of RidePlatformAdapter backed by the Uber for Business
    Guest Rides API (v1/guests/trips).

    Credentials are read from environment variables by default:
        UBER_CLIENT_ID      — OAuth2 client ID
        UBER_CLIENT_SECRET  — OAuth2 client secret
        UBER_SANDBOX        — "true" (default) uses sandbox-api.uber.com
    """

    platform_name = "uber"

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        sandbox: bool | None = None,
    ) -> None:
        _id = client_id or os.environ.get("UBER_CLIENT_ID", "")
        _secret = client_secret or os.environ.get("UBER_CLIENT_SECRET", "")
        _sandbox = (
            sandbox
            if sandbox is not None
            else os.environ.get("UBER_SANDBOX", "true").lower() != "false"
        )
        self._client = UberGuestRidesClient(
            client_id=_id,
            client_secret=_secret,
            sandbox=_sandbox,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _location_payload(self, address: str) -> dict:
        lat, lon = geocode(address)
        return {"latitude": lat, "longitude": lon, "address": address}

    # ------------------------------------------------------------------
    # search_rides
    # ------------------------------------------------------------------

    def search_rides(self, pickup: str, dropoff: str) -> list[dict]:
        pickup_coords = self._location_payload(pickup)
        dropoff_coords = self._location_payload(dropoff)

        data = self._client.get_estimates(pickup_coords, dropoff_coords)

        results: list[dict] = []
        for product in data.get("prices", []):
            if product.get("no_cars_available"):
                continue

            fare = product.get("fare", {})
            trip = product.get("trip", {})

            price_low = float(fare.get("low_value", fare.get("value", 0)))
            price_high = float(fare.get("high_value", fare.get("value", 0)))
            display = fare.get("display") or f"${price_low:.0f}–{price_high:.0f}"

            results.append(
                {
                    "product_id": product.get("product_id", ""),
                    "display_name": product.get("display_name", ""),
                    "description": product.get("description", ""),
                    "pickup": pickup,
                    "dropoff": dropoff,
                    "price_estimate": display,
                    "price_low": price_low,
                    "price_high": price_high,
                    "duration_estimate_minutes": int(
                        trip.get("duration_estimate", 0)
                    ),
                    "surge_multiplier": float(fare.get("surge_multiplier", 1.0)),
                    "currency": fare.get("currency_code", "USD"),
                    "capacity": product.get("capacity", 4),
                    # Stored for use during booking; underscore prefix keeps them
                    # out of the LLM's visible summary lines in tools.py.
                    "_pickup_coords": pickup_coords,
                    "_dropoff_coords": dropoff_coords,
                    "_fare_id": fare.get("fare_id", ""),
                }
            )

        return results

    # ------------------------------------------------------------------
    # book_ride
    # ------------------------------------------------------------------

    def book_ride(
        self, ride_option: dict, guest: UberGuestInfo | None = None
    ) -> dict:
        if guest is None:
            raise ValueError(
                "Guest information is required to book an Uber ride. "
                "Call set_guest_info before book_ride."
            )

        pickup = ride_option.get("_pickup_coords") or self._location_payload(
            ride_option["pickup"]
        )
        dropoff = ride_option.get("_dropoff_coords") or self._location_payload(
            ride_option["dropoff"]
        )

        trip = self._client.create_trip(
            guest=guest,
            product_id=ride_option["product_id"],
            fare_id=ride_option.get("_fare_id", ""),
            pickup=pickup,
            dropoff=dropoff,
        )

        driver = trip.get("driver", {})
        vehicle = trip.get("vehicle", {})
        vehicle_str = (
            f"{vehicle.get('make', '')} {vehicle.get('model', '')}".strip()
            or driver.get("vehicle_year_make_model", "")
        )

        return {
            "ride_id": trip.get("request_id", trip.get("ride_id", "")),
            "status": trip.get("status", "processing"),
            "product": ride_option["display_name"],
            "pickup": ride_option["pickup"],
            "dropoff": ride_option["dropoff"],
            "price_estimate": ride_option["price_estimate"],
            "driver": {
                "name": driver.get("name", ""),
                "rating": float(driver.get("rating") or 0),
                "vehicle": vehicle_str,
                "license_plate": vehicle.get("license_plate", ""),
            },
            "pickup_eta_minutes": trip.get("pickup_estimate", 0),
        }

    # ------------------------------------------------------------------
    # track_ride
    # ------------------------------------------------------------------

    def track_ride(self, ride_id: str) -> dict:
        trip = self._client.get_trip(ride_id)
        location = trip.get("location", {})
        destination = trip.get("destination", {})
        driver = trip.get("driver", {})

        return {
            "ride_id": ride_id,
            "status": trip.get("status", "unknown"),
            "driver_location": {
                "latitude": location.get("latitude"),
                "longitude": location.get("longitude"),
                "bearing": location.get("bearing"),
            },
            "eta_minutes": destination.get("eta"),
            "driver": {
                "name": driver.get("name", ""),
                "rating": float(driver.get("rating") or 0),
                "phone_number": driver.get("phone_number", ""),
            },
        }

    # ------------------------------------------------------------------
    # cancel_ride
    # ------------------------------------------------------------------

    def cancel_ride(self, ride_id: str) -> dict:
        result = self._client.cancel_trip(ride_id)
        return {
            "ride_id": ride_id,
            "status": result.get("status", "cancelled"),
            "cancellation_fee": result.get("cancellation_fee", "$0.00"),
        }
