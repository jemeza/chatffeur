from abc import abstractmethod
from adapters.base import PlatformAdapter


class RidePlatformAdapter(PlatformAdapter):
    """
    Domain adapter for ride-hailing platforms (Uber, Lyft, …).

    To add Lyft: subclass this, override the four abstract methods below,
    and register the class in agent/tools.py's PLATFORM_REGISTRY.
    No agent or graph code needs to change.
    """

    # --- PlatformAdapter bridge ---

    def search(self, **kwargs) -> list[dict]:
        return self.search_rides(
            pickup=kwargs["pickup"],
            dropoff=kwargs["dropoff"],
        )

    def book(self, option: dict) -> dict:
        return self.book_ride(option)

    def track(self, booking_id: str) -> dict:
        return self.track_ride(booking_id)

    def cancel(self, booking_id: str) -> dict:
        return self.cancel_ride(booking_id)

    # --- Ride-specific interface ---

    @abstractmethod
    def search_rides(self, pickup: str, dropoff: str) -> list[dict]:
        """Return a list of available ride options between pickup and dropoff."""
        ...

    @abstractmethod
    def book_ride(self, ride_option: dict) -> dict:
        """Book the specified ride option; return booking confirmation."""
        ...

    @abstractmethod
    def track_ride(self, ride_id: str) -> dict:
        """Return live status and driver location for an active ride."""
        ...

    @abstractmethod
    def cancel_ride(self, ride_id: str) -> dict:
        """Cancel an active ride; return cancellation details."""
        ...
