"""
Address geocoding and validation via Mapbox Search API (Geocoding v5).

Set MAPBOX_ACCESS_TOKEN in your environment to enable Mapbox.
Falls back to Nominatim (OpenStreetMap) when no token is present.
"""
import os
import urllib.parse
from functools import lru_cache

import httpx

_MAPBOX_BASE = "https://api.mapbox.com/geocoding/v5/mapbox.places"
_NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
_USER_AGENT = "chatffeur/0.1 (github.com/jemeza/chatffeur)"


def _mapbox_token() -> str | None:
    return os.environ.get("MAPBOX_ACCESS_TOKEN")


@lru_cache(maxsize=256)
def geocode(address: str) -> tuple[float, float]:
    """Return (latitude, longitude) for the given address.

    Uses Mapbox if MAPBOX_ACCESS_TOKEN is set, otherwise falls back to Nominatim.
    Raises ValueError if the address cannot be geocoded.
    """
    token = _mapbox_token()
    if token:
        return _mapbox_geocode(address, token)
    return _nominatim_geocode(address)


def _mapbox_geocode(address: str, token: str) -> tuple[float, float]:
    encoded = urllib.parse.quote(address)
    resp = httpx.get(
        f"{_MAPBOX_BASE}/{encoded}.json",
        params={"access_token": token, "limit": 1, "types": "address,place,poi"},
        timeout=10,
    )
    resp.raise_for_status()
    features = resp.json().get("features", [])
    if not features:
        raise ValueError(f"Could not geocode address: {address!r}")
    lon, lat = features[0]["geometry"]["coordinates"]
    return lat, lon


def _nominatim_geocode(address: str) -> tuple[float, float]:
    resp = httpx.get(
        _NOMINATIM_URL,
        params={"q": address, "format": "json", "limit": 1},
        headers={"User-Agent": _USER_AGENT},
        timeout=10,
    )
    resp.raise_for_status()
    results = resp.json()
    if not results:
        raise ValueError(f"Could not geocode address: {address!r}")
    hit = results[0]
    return float(hit["lat"]), float(hit["lon"])


def validate_address(address: str) -> dict:
    """Validate an address and return its canonical form, coordinates, and confidence.

    Returns a dict with:
        valid (bool): whether the address could be resolved
        canonical (str | None): full formatted address
        latitude (float | None)
        longitude (float | None)
        place_type (str | None): 'address', 'place', 'poi', etc.
        confidence (float | None): Mapbox relevance score 0–1 (None if using fallback)
        suggestions (list[dict]): up to 4 alternative matches
    """
    token = _mapbox_token()
    if token:
        return _mapbox_validate(address, token)
    return _nominatim_validate(address)


def _mapbox_validate(address: str, token: str) -> dict:
    encoded = urllib.parse.quote(address)
    resp = httpx.get(
        f"{_MAPBOX_BASE}/{encoded}.json",
        params={
            "access_token": token,
            "limit": 5,
            "types": "address,place,poi,neighborhood,district,postcode",
        },
        timeout=10,
    )
    resp.raise_for_status()
    features = resp.json().get("features", [])

    if not features:
        return {
            "valid": False,
            "canonical": None,
            "latitude": None,
            "longitude": None,
            "place_type": None,
            "confidence": None,
            "suggestions": [],
        }

    best = features[0]
    lon, lat = best["geometry"]["coordinates"]
    suggestions = [
        {
            "canonical": f["place_name"],
            "latitude": f["geometry"]["coordinates"][1],
            "longitude": f["geometry"]["coordinates"][0],
            "place_type": (f.get("place_type") or [None])[0],
        }
        for f in features[1:]
    ]

    return {
        "valid": True,
        "canonical": best["place_name"],
        "latitude": lat,
        "longitude": lon,
        "place_type": (best.get("place_type") or [None])[0],
        "confidence": best.get("relevance"),
        "suggestions": suggestions,
    }


def _nominatim_validate(address: str) -> dict:
    try:
        lat, lon = _nominatim_geocode(address)
        return {
            "valid": True,
            "canonical": address,
            "latitude": lat,
            "longitude": lon,
            "place_type": None,
            "confidence": None,
            "suggestions": [],
        }
    except ValueError:
        return {
            "valid": False,
            "canonical": None,
            "latitude": None,
            "longitude": None,
            "place_type": None,
            "confidence": None,
            "suggestions": [],
        }


def suggest_addresses(query: str, limit: int = 5) -> list[dict]:
    """Return address autocomplete suggestions for a partial query.

    Returns a list of {canonical, latitude, longitude, place_type}.
    Returns an empty list when MAPBOX_ACCESS_TOKEN is not set.
    """
    token = _mapbox_token()
    if not token:
        return []

    encoded = urllib.parse.quote(query)
    resp = httpx.get(
        f"{_MAPBOX_BASE}/{encoded}.json",
        params={
            "access_token": token,
            "limit": limit,
            "autocomplete": "true",
            "types": "address,place,poi",
        },
        timeout=10,
    )
    resp.raise_for_status()
    features = resp.json().get("features", [])
    return [
        {
            "canonical": f["place_name"],
            "latitude": f["geometry"]["coordinates"][1],
            "longitude": f["geometry"]["coordinates"][0],
            "place_type": (f.get("place_type") or [None])[0],
        }
        for f in features
    ]
