from __future__ import annotations

"""
Lyft adapter shell.

To activate:
  1. Create a Lyft developer account at developer.lyft.com
  2. Set LYFT_CLIENT_ID and LYFT_CLIENT_SECRET in .env
  3. Implement OAuth2 token fetch and replace NotImplementedError bodies

What changes vs. UberAdapter:
  - Base URL, auth scheme (OAuth2 client_credentials vs. server token)
  - Endpoint paths and response shapes
  - Status string mapping (_map_lyft_status)
  - Product names ("Lyft", "Lyft XL", "Lyft Lux", …)
Everything else — the interface, agent nodes, UI — stays identical.
"""

import os
from typing import ClassVar

import httpx

from adapters.base import (
    Booking,
    CancelResult,
    RideOption,
    RidePlatformAdapter,
    RideStatus,
)

_BASE_URL = "https://api.lyft.com/v1"


class LyftAdapter(RidePlatformAdapter):
    platform_name: ClassVar[str] = "lyft"

    def __init__(self) -> None:
        self._client_id = os.getenv("LYFT_CLIENT_ID", "")
        self._client_secret = os.getenv("LYFT_CLIENT_SECRET", "")
        self._access_token: str | None = None

    def _ensure_token(self) -> None:
        if self._access_token:
            return
        resp = httpx.post(
            "https://api.lyft.com/oauth/token",
            data={"grant_type": "client_credentials", "scope": "public"},
            auth=(self._client_id, self._client_secret),
            timeout=10,
        )
        resp.raise_for_status()
        self._access_token = resp.json()["access_token"]

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._access_token}"}

    def _get(self, path: str, params: dict) -> dict:
        self._ensure_token()
        with httpx.Client(base_url=_BASE_URL, headers=self._headers(), timeout=10) as client:
            resp = client.get(path, params=params)
            resp.raise_for_status()
            return resp.json()

    def _post(self, path: str, body: dict) -> dict:
        self._ensure_token()
        with httpx.Client(base_url=_BASE_URL, headers=self._headers(), timeout=10) as client:
            resp = client.post(path, json=body)
            resp.raise_for_status()
            return resp.json()

    # ------------------------------------------------------------------
    # RidePlatformAdapter interface
    # ------------------------------------------------------------------

    def get_price_estimates(self, pickup: str, dropoff: str) -> list[RideOption]:
        # data = self._get("/cost", {
        #     "ride_type": "lyft",
        #     "start_lat": ..., "start_lng": ...,
        #     "end_lat":   ..., "end_lng":   ...,
        # })
        # return [_parse_lyft_estimate(e) for e in data["cost_estimates"]]
        raise NotImplementedError(
            "LyftAdapter requires Lyft API credentials. "
            "Set LYFT_CLIENT_ID and LYFT_CLIENT_SECRET in .env."
        )

    def book_ride(self, option: RideOption, pickup: str, dropoff: str) -> Booking:
        raise NotImplementedError("LyftAdapter.book_ride not yet implemented.")

    def get_ride_status(self, booking_id: str) -> RideStatus:
        raise NotImplementedError("LyftAdapter.get_ride_status not yet implemented.")

    def cancel_ride(self, booking_id: str) -> CancelResult:
        raise NotImplementedError("LyftAdapter.cancel_ride not yet implemented.")


_LYFT_STATUS_MAP: dict[str, RideStatus] = {
    "pending":      "PROCESSING",
    "accepted":     "DRIVER_ASSIGNED",
    "arrived":      "ARRIVED",
    "pickedUp":     "IN_PROGRESS",
    "droppedOff":   "COMPLETED",
    "canceled":     "CANCELLED",
    "unknown":      "PROCESSING",
}


def _map_lyft_status(lyft_status: str) -> RideStatus:
    return _LYFT_STATUS_MAP.get(lyft_status, "PROCESSING")
