"""
HTTP client for the Uber for Business Guest Rides API.

Authentication: OAuth 2.0 client credentials (scope: guests.trips)
Production:     https://api.uber.com
Sandbox:        https://sandbox-api.uber.com  (set UBER_SANDBOX=true)

Rate limit: 200 requests/hour per endpoint by default.
"""
import time

import httpx
from pydantic import BaseModel


class UberGuestInfo(BaseModel):
    """Personal details required to create a guest ride booking."""

    first_name: str
    last_name: str
    email: str
    phone_number: str  # E.164 format recommended: +12125551234


class UberAPIError(Exception):
    def __init__(self, status_code: int, body: str) -> None:
        self.status_code = status_code
        self.body = body
        super().__init__(f"Uber API error {status_code}: {body}")


class UberGuestRidesClient:
    _PRODUCTION_BASE = "https://api.uber.com"
    _SANDBOX_BASE = "https://sandbox-api.uber.com"

    # Each environment has its own OAuth server — tokens are not cross-compatible.
    _PRODUCTION_TOKEN_URL = "https://auth.uber.com/oauth/v2/token"
    _SANDBOX_TOKEN_URL = "https://sandbox-api.uber.com/oauth/v2/token"

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        sandbox: bool = True,
    ) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._sandbox = sandbox
        self._base_url = self._SANDBOX_BASE if sandbox else self._PRODUCTION_BASE
        self._token_url = self._SANDBOX_TOKEN_URL if sandbox else self._PRODUCTION_TOKEN_URL
        self._token: str | None = None
        self._token_expires_at: float = 0.0

    # ------------------------------------------------------------------
    # Auth — client credentials, auto-refreshed 30 s before expiry
    # ------------------------------------------------------------------

    def _fetch_token(self) -> None:
        resp = httpx.post(
            self._token_url,
            data={
                "client_id": self._client_id,
                "client_secret": self._client_secret,
                "grant_type": "client_credentials",
                "scope": "guests.trips",
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=10,
        )
        self._raise_for_status(resp)
        payload = resp.json()
        self._token = payload["access_token"]
        self._token_expires_at = time.monotonic() + payload.get("expires_in", 3600)

    def _get_token(self) -> str:
        if self._token is None or time.monotonic() >= self._token_expires_at - 30:
            self._fetch_token()
        return self._token  # type: ignore[return-value]

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._get_token()}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    @staticmethod
    def _raise_for_status(resp: httpx.Response) -> None:
        if resp.is_error:
            raise UberAPIError(resp.status_code, resp.text)

    # ------------------------------------------------------------------
    # Endpoints
    # ------------------------------------------------------------------

    def list_products(self, latitude: float, longitude: float) -> dict:
        """
        GET /v1/products?latitude=X&longitude=Y

        Returns products available near the given coordinates.  Used in
        sandbox mode to discover product IDs before seeding their state.
        """
        resp = httpx.get(
            f"{self._base_url}/v1/products",
            headers=self._headers(),
            params={"latitude": latitude, "longitude": longitude},
            timeout=10,
        )
        self._raise_for_status(resp)
        return resp.json()

    def setup_sandbox(self, latitude: float, longitude: float) -> None:
        """
        Seed the Uber sandbox so that estimates calls return results.

        The sandbox starts with every product unavailable.  This method
        fetches all products for the given location then PUTs each one
        to /v1/sandbox/products/{product_id} with drivers_available=true.

        Safe to call in production mode — it becomes a no-op.
        """
        if not self._sandbox:
            return

        data = self.list_products(latitude, longitude)
        for product in data.get("products", []):
            product_id = product.get("product_id")
            if not product_id:
                continue
            resp = httpx.put(
                f"{self._base_url}/v1/sandbox/products/{product_id}",
                headers=self._headers(),
                json={"drivers_available": True, "surge_multiplier": 1.0},
                timeout=10,
            )
            self._raise_for_status(resp)

    def get_estimates(self, pickup: dict, dropoff: dict) -> dict:
        """
        POST /v1/guests/trips/estimates

        pickup / dropoff: {"latitude": float, "longitude": float, "address": str}
        Returns a dict with a "prices" list; each entry has product info,
        fare details (including fare_id for locking upfront price), and ETA.
        """
        resp = httpx.post(
            f"{self._base_url}/v1/guests/trips/estimates",
            headers=self._headers(),
            json={"pickup": pickup, "dropoff": dropoff},
            timeout=15,
        )
        self._raise_for_status(resp)
        return resp.json()

    def create_trip(
        self,
        guest: UberGuestInfo,
        product_id: str,
        fare_id: str,
        pickup: dict,
        dropoff: dict,
    ) -> dict:
        """
        POST /v1/guests/trips/

        Returns the created trip object including request_id, status, and
        driver / vehicle details.  fare_id locks the upfront price from
        get_estimates; omit or pass "" to let Uber calculate at request time.
        """
        body: dict = {
            "guest": guest.model_dump(),
            "product_id": product_id,
            "pickup": pickup,
            "dropoff": dropoff,
        }
        if fare_id:
            body["fare_id"] = fare_id
        resp = httpx.post(
            f"{self._base_url}/v1/guests/trips/",
            headers=self._headers(),
            json=body,
            timeout=15,
        )
        self._raise_for_status(resp)
        return resp.json()

    def get_trip(self, request_id: str) -> dict:
        """GET /v1/guests/trips/{request_id}"""
        resp = httpx.get(
            f"{self._base_url}/v1/guests/trips/{request_id}",
            headers=self._headers(),
            timeout=10,
        )
        self._raise_for_status(resp)
        return resp.json()

    def list_trips(self) -> dict:
        """GET /v1/guests/trips/"""
        resp = httpx.get(
            f"{self._base_url}/v1/guests/trips/",
            headers=self._headers(),
            timeout=10,
        )
        self._raise_for_status(resp)
        return resp.json()

    def cancel_trip(self, request_id: str) -> dict:
        """
        PUT /v1/guests/trips/{request_id}

        Sends status="rider_canceled" to cancel the trip.
        Verify the exact status string against your Uber account tier's docs.
        """
        resp = httpx.put(
            f"{self._base_url}/v1/guests/trips/{request_id}",
            headers=self._headers(),
            json={"status": "rider_canceled"},
            timeout=10,
        )
        self._raise_for_status(resp)
        return resp.json()
