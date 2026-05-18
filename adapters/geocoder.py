"""
Address geocoding via Nominatim (OpenStreetMap).

No API key required.  Nominatim asks for a descriptive User-Agent and
enforces a soft rate limit of ~1 request/second for anonymous clients.
Results are cached in-process so repeated lookups (e.g. same pickup across
multiple search_rides calls in a session) never hit the network twice.
"""
from functools import lru_cache

import httpx

_NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
_USER_AGENT = "chatffeur/0.1 (github.com/jemeza/chatffeur)"


@lru_cache(maxsize=256)
def geocode(address: str) -> tuple[float, float]:
    """Return (latitude, longitude) for the given free-text address.

    Raises ValueError if Nominatim returns no results.
    """
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
