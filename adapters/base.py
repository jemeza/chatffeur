from abc import ABC, abstractmethod


class PlatformAdapter(ABC):
    """
    Abstract base for all platform adapters.

    To add a new platform (e.g. Lyft, Zocdoc), subclass the appropriate
    domain adapter (RidePlatformAdapter, MedicalPlatformAdapter, etc.) —
    never this class directly unless building a wholly new domain.
    """

    platform_name: str

    @abstractmethod
    def search(self, **kwargs) -> list[dict]:
        """Search for available options on this platform."""
        ...

    @abstractmethod
    def book(self, option: dict) -> dict:
        """Book the given option."""
        ...

    @abstractmethod
    def track(self, booking_id: str) -> dict:
        """Return current status of an active booking."""
        ...

    @abstractmethod
    def cancel(self, booking_id: str) -> dict:
        """Cancel an active booking."""
        ...
