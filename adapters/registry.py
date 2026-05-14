from __future__ import annotations

from adapters.base import RidePlatformAdapter
from adapters.mock import MockAdapter
from adapters.uber import UberAdapter
from adapters.lyft import LyftAdapter

# Add new platforms here — nothing else in the codebase needs to change.
_REGISTRY: dict[str, type[RidePlatformAdapter]] = {
    "mock": MockAdapter,
    "uber": UberAdapter,
    "lyft": LyftAdapter,
}


def get_adapter(platform: str) -> RidePlatformAdapter:
    """Return an instantiated adapter for the given platform name."""
    key = platform.lower().strip()
    cls = _REGISTRY.get(key)
    if cls is None:
        available = ", ".join(_REGISTRY.keys())
        raise ValueError(f"Unknown platform '{platform}'. Available: {available}")
    return cls()


def available_platforms() -> list[str]:
    return list(_REGISTRY.keys())
