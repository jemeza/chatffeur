from __future__ import annotations

"""
Uber adapter shell.

Uber deprecated their public Ride Request API in 2019. This adapter targets
Uber for Business / enterprise API endpoints. To activate:
  1. Obtain Uber for Business API credentials from developer.uber.com
  2. Set UBER_CLIENT_ID, UBER_CLIENT_SECRET, UBER_SERVER_TOKEN in .env
  3. Replace the NotImplementedError bodies with real httpx calls

The interface (get_price_estimates, book_ride, get_ride_status, cancel_ride)
is identical to every other adapter — only this file changes.
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

_BASE_URL = "https://api.uber.com/v1.2"


class UberAdapter(RidePlatformAdapter):
    platform_name: ClassVar[str] = "uber"

    def __init__(self) -> None:
        self._server_token = os.getenv("UBER_SERVER_TOKEN", "")
        self._client_id = os.getenv("UBER_CLIENT_ID", "")
        self._client_secret = os.getenv("UBER_CLIENT_SECRET", "")
        self._headers = {
            "Authorization": f"Token {self._server_token}",
            "Content-Type": "application/json",
            "Accept-Language": "en_US",
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get(self, path: str, params: dict) -> dict:
        with httpx.Client(base_url=_BASE_URL, headers=self._headers, timeout=10) as client:
            resp = client.get(path, params=params)
            resp.raise_for_status()
            return resp.json()

    def _post(self, path: str, body: dict) -> dict:
        with httpx.Client(base_url=_BASE_URL, headers=self._headers, timeout=10) as client:
            resp = client.post(path, json=body)
            resp.raise_for_status()
            return resp.json()

    def _delete(self, path: str) -> dict:
        with httpx.Client(base_url=_BASE_URL, headers=self._headers, timeout=10) as client:
            resp = client.delete(path)
            resp.raise_for_status()
            return resp.json()

    # ------------------------------------------------------------------
    # RidePlatformAdapter interface
    # ------------------------------------------------------------------

    def get_price_estimates(self, pickup: str, dropoff: str) -> list[RideOption]:
        # TODO: geocode pickup/dropoff strings → lat/lng before calling Uber API
        # data = self._get("/estimates/price", {
        #     "start_latitude": ..., "start_longitude": ...,
        #     "end_latitude": ..., "end_longitude": ...,
        # })
        # return [_parse_uber_estimate(e) for e in data["prices"]]
        raise NotImplementedError(
            "UberAdapter requires Uber for Business API credentials. "
            "Set UBER_SERVER_TOKEN in .env and implement geocoding."
        )

    def book_ride(self, option: RideOption, pickup: str, dropoff: str) -> Booking:
        # body = {
        #     "fare_id": option.option_id,
        #     "product_id": option.option_id,
        #     "start_latitude": ..., "start_longitude": ...,
        #     "end_latitude": ...,   "end_longitude": ...,
        # }
        # data = self._post("/requests", body)
        # return _parse_uber_booking(data, option, pickup, dropoff)
        raise NotImplementedError("UberAdapter.book_ride not yet implemented.")

    def get_ride_status(self, booking_id: str) -> RideStatus:
        # data = self._get(f"/requests/{booking_id}", {})
        # return _map_uber_status(data["status"])
        raise NotImplementedError("UberAdapter.get_ride_status not yet implemented.")

    def cancel_ride(self, booking_id: str) -> CancelResult:
        # self._delete(f"/requests/{booking_id}")
        # return CancelResult(success=True, message="Ride cancelled.")
        raise NotImplementedError("UberAdapter.cancel_ride not yet implemented.")


# ------------------------------------------------------------------
# Helpers for parsing Uber API responses (fill in when activating)
# ------------------------------------------------------------------

_UBER_STATUS_MAP: dict[str, RideStatus] = {
    "processing":        "PROCESSING",
    "accepted":          "DRIVER_ASSIGNED",
    "arriving":          "EN_ROUTE",
    "waiting":           "ARRIVED",
    "in_progress":       "IN_PROGRESS",
    "driver_canceled":   "CANCELLED",
    "rider_canceled":    "CANCELLED",
    "completed":         "COMPLETED",
    "no_drivers_available": "NO_DRIVERS",
}


def _map_uber_status(uber_status: str) -> RideStatus:
    return _UBER_STATUS_MAP.get(uber_status.lower(), "PROCESSING")
