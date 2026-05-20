"""
Agent tools exposed to the LLM.

Tool call flow:
  1. set_platform           — pick the ride platform (default: uber)
  2. search_rides           — discover options and prices
  2b. find_pickup_locations — (optional) find nearby spots with lower surge
  3. suggest_ride           — recommend one option; INTERRUPTS for user confirmation
  4. set_guest_info         — collect rider name / email / phone before booking
  5. book_ride              — executes only when state.ride_confirmed is True
  6. track_ride             — poll live driver location and ETA after booking
  7. cancel_ride            — cancel the active booking; reports any cancellation fee

Each tool returns a Command that both updates AgentState fields and injects
the ToolMessage (keyed by tool_call_id) that the LLM reads as the result.

Adding a new platform (e.g. Lyft):
  1. Subclass RidePlatformAdapter in adapters/
  2. Add the class to PLATFORM_REGISTRY below
  — no other changes required.
"""

import math
from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command, interrupt

from adapters.geocoder import geocode as _geocode
from adapters.geocoder import validate_address as _mapbox_validate_address
from adapters.uber_adapter import UberRideAdapter
from adapters.mock_uber_client import UberGuestInfo
from agent.action_log import make_log_entry
from agent.state import AgentState

PLATFORM_REGISTRY = {
    "uber": UberRideAdapter,
    # "lyft": LyftRideAdapter,   ← add here
    # "zocdoc": ZocdocAdapter,   ← different domain, subclass PlatformAdapter
}

# Singleton adapter instances — keyed by platform name so that in-memory
# state (e.g. MockUberGuestRidesClient._trips) survives across tool calls.
_ADAPTER_INSTANCES: dict = {}


def _get_adapter(platform_name: str):
    cls = PLATFORM_REGISTRY.get(platform_name)
    if cls is None:
        raise ValueError(
            f"Unknown platform '{platform_name}'. "
            f"Supported: {list(PLATFORM_REGISTRY)}"
        )
    if platform_name not in _ADAPTER_INSTANCES:
        _ADAPTER_INSTANCES[platform_name] = cls()
    return _ADAPTER_INSTANCES[platform_name]


def _resolve_guest_info(raw) -> UberGuestInfo | None:
    """Coerce state.guest_info (UberGuestInfo or dict) to UberGuestInfo."""
    if raw is None:
        return None
    if isinstance(raw, UberGuestInfo):
        return raw
    return UberGuestInfo(**raw)


# ---------------------------------------------------------------------------
# Tool 0 — validate_address
# ---------------------------------------------------------------------------


@tool
def validate_address(
    address: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """
    Validate a pickup or dropoff address using the Mapbox Search API and return
    its canonical form, coordinates, and alternative suggestions.

    Call this when the user provides an ambiguous or potentially misspelled
    address before calling search_rides, so you can confirm the exact location.

    Args:
        address: Free-text address to validate (e.g. '350 Fifth Ave, New York').
    """
    result = _mapbox_validate_address(address)

    if not result["valid"]:
        msg = (
            f"Could not find a match for **{address!r}**. "
            "Please ask the user to provide a more specific address."
        )
        log = make_log_entry(
            "validate_address",
            requested={"address": address},
            verified={"valid": False},
            executed={},
            outcome="error: address not found",
        )
    else:
        canonical = result["canonical"]
        confidence = result["confidence"]
        suggestions = result.get("suggestions", [])

        confidence_str = f" (confidence: {confidence:.0%})" if confidence is not None else ""
        msg = f"Address validated: **{canonical}**{confidence_str}"

        if suggestions:
            alt_lines = "\n".join(f"  - {s['canonical']}" for s in suggestions[:3])
            msg += f"\n\nAlternative matches if this isn't right:\n{alt_lines}"

        log = make_log_entry(
            "validate_address",
            requested={"address": address},
            verified={"valid": True, "canonical": canonical},
            executed={
                "latitude": result["latitude"],
                "longitude": result["longitude"],
                "place_type": result["place_type"],
            },
            outcome=f"validated: {canonical}",
        )

    return Command(
        update={
            "action_log": [log],
            "messages": [ToolMessage(content=msg, tool_call_id=tool_call_id)],
        }
    )


# ---------------------------------------------------------------------------
# Tool 1 — set_platform
# ---------------------------------------------------------------------------


@tool
def set_platform(
    platform: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """
    Set the ride-hailing platform to use for this session.

    Args:
        platform: Platform name, e.g. 'uber'.
    """
    supported = list(PLATFORM_REGISTRY)

    if platform not in supported:
        msg = (
            f"Unsupported platform '{platform}'. "
            f"Supported platforms: {supported}."
        )
        log = make_log_entry(
            "set_platform",
            requested={"platform": platform},
            verified={"supported": False},
            executed={},
            outcome=f"error: {msg}",
        )
        return Command(
            update={
                "action_log": [log],
                "messages": [ToolMessage(content=msg, tool_call_id=tool_call_id)],
            }
        )

    log = make_log_entry(
        "set_platform",
        requested={"platform": platform},
        verified={"supported": True},
        executed={"adapter_class": PLATFORM_REGISTRY[platform].__name__},
        outcome="success",
    )
    return Command(
        update={
            "platform_adapter": platform,
            "action_log": [log],
            "messages": [
                ToolMessage(
                    content=f"Platform set to '{platform}'. Ready to search for rides.",
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )


# ---------------------------------------------------------------------------
# Tool 2 — search_rides
# ---------------------------------------------------------------------------


@tool
def search_rides(
    pickup: str,
    dropoff: str,
    state: Annotated[AgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """
    Search for available rides between two locations.

    Args:
        pickup:  Starting address or landmark (e.g. '123 Main St, New York').
        dropoff: Destination address or landmark (e.g. 'JFK Airport').
    """
    platform = state.platform_adapter or "uber"

    try:
        adapter = _get_adapter(platform)
        results = adapter.search_rides(pickup=pickup, dropoff=dropoff)

        def _surge_tag(multiplier: float) -> str:
            return f" 🔺 {multiplier}× surge" if multiplier > 1.0 else ""

        summary_lines = [
            f"{i}. **{r['display_name']}** — {r['price_estimate']}"
            f"{_surge_tag(r.get('surge_multiplier', 1.0))} "
            f"(~{r['duration_estimate_minutes']} min, up to {r['capacity']} passengers)"
            for i, r in enumerate(results)
        ]
        surge_multiplier = results[0].get("surge_multiplier", 1.0) if results else 1.0
        surge_note = (
            f"\n\n⚠️ Surge pricing is active at **{surge_multiplier}×** the normal rate."
            if surge_multiplier > 1.0 else ""
        )
        content = f"Found {len(results)} ride options:\n" + "\n".join(summary_lines) + surge_note

        log = make_log_entry(
            "search_rides",
            requested={"pickup": pickup,
                       "dropoff": dropoff, "platform": platform},
            verified={"adapter_available": True},
            executed={"method": "search_rides",
                      "pickup": pickup, "dropoff": dropoff},
            outcome=f"found {len(results)} options",
        )
        return Command(
            update={
                "search_results": results,
                "action_log": [log],
                "messages": [ToolMessage(content=content, tool_call_id=tool_call_id)],
            }
        )

    except Exception as exc:
        log = make_log_entry(
            "search_rides",
            requested={"pickup": pickup, "dropoff": dropoff},
            verified={},
            executed={},
            outcome=f"error: {exc}",
        )
        return Command(
            update={
                "action_log": [log],
                "messages": [
                    ToolMessage(
                        content=f"Error searching rides: {exc}",
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )


# ---------------------------------------------------------------------------
# Tool 3 — suggest_ride  (triggers UI confirmation via interrupt)
# ---------------------------------------------------------------------------


@tool
def suggest_ride(
    ride_index: int,
    reasoning: str,
    state: Annotated[AgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """
    Suggest a specific ride option to the user and wait for their confirmation.
    The graph will PAUSE here until the user clicks Confirm or Reject in the UI.
    Do NOT call book_ride until this tool has returned.

    Args:
        ride_index: Zero-based index into the list returned by search_rides.
        reasoning:  Short explanation of why this option was chosen.
    """
    results = state.search_results

    if not results:
        msg = "No search results available. Call search_rides first."
        log = make_log_entry(
            "suggest_ride",
            requested={"ride_index": ride_index},
            verified={"has_results": False},
            executed={},
            outcome="error: no search results",
        )
        return Command(
            update={
                "action_log": [log],
                "messages": [ToolMessage(content=msg, tool_call_id=tool_call_id)],
            }
        )

    if ride_index < 0 or ride_index >= len(results):
        msg = f"Invalid ride_index {ride_index}. Valid range: 0–{len(results) - 1}."
        log = make_log_entry(
            "suggest_ride",
            requested={"ride_index": ride_index},
            verified={"valid_index": False},
            executed={},
            outcome="error: invalid index",
        )
        return Command(
            update={
                "action_log": [log],
                "messages": [ToolMessage(content=msg, tool_call_id=tool_call_id)],
            }
        )

    ride = results[ride_index]

    # ----------------------------------------------------------------
    # INTERRUPT — graph pauses here; Streamlit shows confirm/reject UI.
    # On resume, `confirmed` is the bool the user chose.
    # Code before interrupt() may replay on resume; keep it side-effect-free.
    # ----------------------------------------------------------------
    confirmed: bool = interrupt(
        {
            "suggested_ride": ride,
            "reasoning": reasoning,
        }
    )

    # Everything below runs exactly once — after the user responds.
    outcome = "confirmed" if confirmed else "rejected"
    log = make_log_entry(
        "suggest_ride",
        requested={"ride_index": ride_index, "reasoning": reasoning},
        verified={"ride": ride["display_name"]},
        executed={"user_decision": outcome},
        outcome=outcome,
    )

    if confirmed:
        content = (
            f"User confirmed **{ride['display_name']}**. "
            "Proceeding to book the ride."
        )
    else:
        content = (
            f"User rejected **{ride['display_name']}**. "
            "Ask the user what they'd prefer or suggest an alternative."
        )

    return Command(
        update={
            "suggested_ride": ride,
            "ride_confirmed": confirmed,
            "action_log": [log],
            "messages": [ToolMessage(content=content, tool_call_id=tool_call_id)],
        }
    )


# ---------------------------------------------------------------------------
# Tool 4 — set_guest_info
# ---------------------------------------------------------------------------


@tool
def set_guest_info(
    first_name: str,
    last_name: str,
    email: str,
    phone_number: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """
    Store the guest's personal details required by the Uber Guest Rides API.
    Call this before book_ride whenever guest information has not yet been set.
    Ask the user for these details if they have not been volunteered.

    Args:
        first_name:   Guest's first name.
        last_name:    Guest's last name.
        email:        Guest's email address.
        phone_number: Guest's phone number in E.164 format (e.g. +12125551234).
    """
    try:
        guest = UberGuestInfo(
            first_name=first_name,
            last_name=last_name,
            email=email,
            phone_number=phone_number,
        )
    except Exception as exc:
        log = make_log_entry(
            "set_guest_info",
            requested={
                "first_name": first_name,
                "last_name": last_name,
                "email": email,
                "phone_number": phone_number,
            },
            verified={"valid": False},
            executed={},
            outcome=f"error: {exc}",
        )
        return Command(
            update={
                "action_log": [log],
                "messages": [
                    ToolMessage(
                        content=f"Invalid guest info: {exc}",
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )

    log = make_log_entry(
        "set_guest_info",
        requested={"first_name": first_name,
                   "last_name": last_name, "email": email},
        verified={"valid": True},
        executed={"stored": True},
        outcome=f"guest info set for {first_name} {last_name}",
    )
    return Command(
        update={
            "guest_info": guest,
            "action_log": [log],
            "messages": [
                ToolMessage(
                    content=(
                        f"Guest info saved for **{first_name} {last_name}** "
                        f"({email}). Ready to proceed with booking."
                    ),
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )


# ---------------------------------------------------------------------------
# Tool 5 — book_ride
# ---------------------------------------------------------------------------


@tool
def book_ride(
    state: Annotated[AgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """
    Book the ride that the user has already confirmed via suggest_ride.
    Requires guest info to have been set via set_guest_info.
    Will refuse to proceed if the confirmation gate has not been passed.
    """
    confirmed = state.ride_confirmed or False
    ride = state.suggested_ride
    platform = state.platform_adapter or "uber"
    guest = _resolve_guest_info(state.guest_info)

    # Safety gate — cannot be bypassed by the LLM
    if not confirmed or not ride:
        msg = (
            "Cannot book: the user has not confirmed a ride yet. "
            "Call suggest_ride first and wait for user confirmation."
        )
        log = make_log_entry(
            "book_ride",
            requested={"platform": platform},
            verified={"ride_confirmed": confirmed,
                      "ride_available": bool(ride)},
            executed={},
            outcome="blocked: confirmation gate not satisfied",
        )
        return Command(
            update={
                "action_log": [log],
                "messages": [ToolMessage(content=msg, tool_call_id=tool_call_id)],
            }
        )

    if guest is None:
        msg = (
            "Cannot book: guest information is missing. "
            "Call set_guest_info with the rider's name, email, and phone number first."
        )
        log = make_log_entry(
            "book_ride",
            requested={"platform": platform},
            verified={"ride_confirmed": True, "guest_info_set": False},
            executed={},
            outcome="blocked: guest info missing",
        )
        return Command(
            update={
                "action_log": [log],
                "messages": [ToolMessage(content=msg, tool_call_id=tool_call_id)],
            }
        )

    try:
        adapter = _get_adapter(platform)
        booking = adapter.book_ride(ride, guest=guest)
        state.booked_ride = booking

        driver = booking.get("driver", {})
        content = (
            f"Ride booked successfully!\n\n"
            f"**Ride ID:** `{booking['ride_id']}`\n"
            f"**Driver:** {driver.get('name')} "
            f"({driver.get('vehicle')}, plate {driver.get('license_plate')})\n"
            f"**Driver rating:** ⭐ {driver.get('rating')}\n"
            f"**Pickup ETA:** {booking['pickup_eta_minutes']} minutes\n"
            f"**Price estimate:** {booking['price_estimate']}\n"
        )

        log = make_log_entry(
            "book_ride",
            requested={"ride": ride["display_name"], "platform": platform},
            verified={"ride_confirmed": True, "ride_available": True},
            executed={
                "method": "book_ride",
                "product": ride["display_name"],
                "ride_id": booking["ride_id"],
            },
            outcome=f"success: ride_id={booking['ride_id']}",
        )
        return Command(
            update={
                "booked_ride": booking,
                "action_log": [log],
                "messages": [ToolMessage(content=content, tool_call_id=tool_call_id)],
            }
        )

    except Exception as exc:
        log = make_log_entry(
            "book_ride",
            requested={"ride": ride.get("display_name"), "platform": platform},
            verified={"ride_confirmed": True},
            executed={},
            outcome=f"error: {exc}",
        )
        return Command(
            update={
                "action_log": [log],
                "messages": [
                    ToolMessage(
                        content=f"Booking failed: {exc}",
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )


# ---------------------------------------------------------------------------
# Tool 6 — track_ride
# ---------------------------------------------------------------------------


@tool
def track_ride(
    state: Annotated[AgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """
    Get the current status and driver location for the active booked ride.
    Call this after book_ride to check ride progress.
    """
    booked = state.booked_ride
    platform = state.platform_adapter or "uber"

    if not booked or not booked.get("ride_id"):
        msg = "No active booking found. Call book_ride first."
        log = make_log_entry(
            "track_ride",
            requested={"platform": platform},
            verified={"has_booking": False},
            executed={},
            outcome="error: no active booking",
        )
        return Command(
            update={
                "action_log": [log],
                "messages": [ToolMessage(content=msg, tool_call_id=tool_call_id)],
            }
        )

    ride_id = booked["ride_id"]

    try:
        adapter = _get_adapter(platform)
        status = adapter.track_ride(ride_id)

        loc = status.get("driver_location", {})
        driver = status.get("driver", {})
        eta = status.get("eta_minutes")
        eta_str = f"{eta} minutes" if eta is not None else "unknown"

        content = (
            f"**Ride status:** {status.get('status', 'unknown')}\n"
            f"**Driver:** {driver.get('name')} ⭐ {driver.get('rating')}\n"
            f"**ETA:** {eta_str}\n"
            f"**Driver location:** lat {loc.get('latitude')}, "
            f"lon {loc.get('longitude')}"
        )

        log = make_log_entry(
            "track_ride",
            requested={"ride_id": ride_id, "platform": platform},
            verified={"has_booking": True},
            executed={"method": "track_ride", "ride_id": ride_id},
            outcome=f"status={status.get('status', 'unknown')}",
        )
        return Command(
            update={
                "action_log": [log],
                "messages": [ToolMessage(content=content, tool_call_id=tool_call_id)],
            }
        )

    except Exception as exc:
        log = make_log_entry(
            "track_ride",
            requested={"ride_id": ride_id, "platform": platform},
            verified={"has_booking": True},
            executed={},
            outcome=f"error: {exc}",
        )
        return Command(
            update={
                "action_log": [log],
                "messages": [
                    ToolMessage(
                        content=f"Error tracking ride: {exc}",
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )


# ---------------------------------------------------------------------------
# Tool 7 — cancel_ride
# ---------------------------------------------------------------------------


@tool
def cancel_ride(
    state: Annotated[AgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """
    Cancel the currently active booked ride.
    Reports any cancellation fee that applies.
    A fee of $5.00 may be charged if the driver is already on the way.
    """
    booked = state.booked_ride
    platform = state.platform_adapter or "uber"

    if not booked or not booked.get("ride_id"):
        msg = "No active booking to cancel. Call book_ride first."
        log = make_log_entry(
            "cancel_ride",
            requested={"platform": platform},
            verified={"has_booking": False},
            executed={},
            outcome="error: no active booking",
        )
        return Command(
            update={
                "action_log": [log],
                "messages": [ToolMessage(content=msg, tool_call_id=tool_call_id)],
            }
        )

    ride_id = booked["ride_id"]

    try:
        adapter = _get_adapter(platform)
        result = adapter.cancel_ride(ride_id)

        fee = result.get("cancellation_fee", "$0.00")
        fee_note = (
            f"\n\n**Cancellation fee:** {fee}"
            if fee != "$0.00"
            else "\n\nNo cancellation fee was charged."
        )
        content = (
            f"Ride `{ride_id}` has been cancelled successfully.{fee_note}"
        )

        log = make_log_entry(
            "cancel_ride",
            requested={"ride_id": ride_id, "platform": platform},
            verified={"has_booking": True},
            executed={"method": "cancel_ride", "ride_id": ride_id},
            outcome=f"cancelled; fee={fee}",
        )
        return Command(
            update={
                "booked_ride": None,
                "ride_confirmed": False,
                "action_log": [log],
                "messages": [ToolMessage(content=content, tool_call_id=tool_call_id)],
            }
        )

    except Exception as exc:
        log = make_log_entry(
            "cancel_ride",
            requested={"ride_id": ride_id, "platform": platform},
            verified={"has_booking": True},
            executed={},
            outcome=f"error: {exc}",
        )
        return Command(
            update={
                "action_log": [log],
                "messages": [
                    ToolMessage(
                        content=f"Error cancelling ride: {exc}",
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )


# ---------------------------------------------------------------------------
# Tool 8 — find_pickup_locations
# ---------------------------------------------------------------------------


@tool
def find_pickup_locations(
    location: str,
    dropoff: str,
    state: Annotated[AgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    radius_meters: int = 400,
) -> Command:
    """
    Find nearby pickup (or drop-off) spots that may have lower surge pricing.

    Use this when search_rides shows a surge multiplier above 1.0 and the user
    wants alternatives within walking distance. Checks up to 9 candidate points
    (the original location plus 8 compass directions at the requested radius)
    and returns the top 5 ranked by surge multiplier, lowest first.

    Args:
        location:      The original pickup address to centre the search on.
        dropoff:       The destination address (needed for accurate fare estimates).
        radius_meters: How far from the original location to search (default 400 m,
                       roughly a 5-minute walk).
    """
    platform = state.platform_adapter or "uber"

    try:
        # Geocode center pickup with a deterministic fallback for offline envs.
        def _safe_geocode(address: str) -> tuple[float, float]:
            try:
                return _geocode(address)
            except Exception:
                seed = sum(ord(c) for c in address)
                return (
                    round(40.7128 + (seed % 1000) / 10000, 6),
                    round(-74.0060 - (seed % 1000) / 10000, 6),
                )

        center_lat, center_lon = _safe_geocode(location)
        drop_lat, drop_lon = _safe_geocode(dropoff)

        dropoff_payload = {
            "latitude": drop_lat,
            "longitude": drop_lon,
            "address": dropoff,
        }

        # Build candidate pickup points: original + 8 compass offsets.
        # 1 degree lat ≈ 111,000 m; 1 degree lon ≈ 111,000 × cos(lat) m.
        meters_per_deg_lat = 111_000
        meters_per_deg_lon = 111_000 * math.cos(math.radians(center_lat))

        directions = [
            ("north", 0),
            ("northeast", 45),
            ("east", 90),
            ("southeast", 135),
            ("south", 180),
            ("southwest", 225),
            ("west", 270),
            ("northwest", 315),
        ]

        candidates = [
            {
                "label": location,
                "walk_meters": 0,
                "lat": center_lat,
                "lon": center_lon,
                "is_original": True,
            }
        ]
        for direction, bearing in directions:
            rad = math.radians(bearing)
            candidates.append(
                {
                    "label": f"~{radius_meters}m {direction} of {location}",
                    "walk_meters": radius_meters,
                    "lat": round(center_lat + (radius_meters * math.cos(rad)) / meters_per_deg_lat, 6),
                    "lon": round(center_lon + (radius_meters * math.sin(rad)) / meters_per_deg_lon, 6),
                    "is_original": False,
                }
            )

        # Query estimates for each candidate via the adapter's mock client.
        client = _get_adapter(platform)._client
        evaluated: list[dict] = []
        for candidate in candidates:
            pickup_payload = {
                "latitude": candidate["lat"],
                "longitude": candidate["lon"],
                "address": candidate["label"],
            }
            prices = client.get_estimates(pickup_payload, dropoff_payload).get("prices", [])
            if not prices:
                continue

            # Surge is uniform across products within one estimate call.
            surge = float(prices[0].get("fare", {}).get("surge_multiplier", 1.0))

            # Use UberX as the reference price (cheapest standard option).
            ref = next(
                (p for p in prices if "UberX" in p.get("display_name", "") and not p.get("no_cars_available")),
                prices[0],
            )
            fare = ref.get("fare", {})
            low = float(fare.get("low_value", 0))
            high = float(fare.get("high_value", 0))
            price_display = fare.get("display") or f"${(low + high) / 2:.0f}"

            evaluated.append(
                {
                    "label": candidate["label"],
                    "walk_meters": candidate["walk_meters"],
                    "surge_multiplier": surge,
                    "price_display": price_display,
                    "is_original": candidate["is_original"],
                }
            )

        evaluated.sort(key=lambda r: (r["surge_multiplier"], r["walk_meters"]))

        def _walk_time(meters: int) -> str:
            if meters == 0:
                return "current location"
            return f"~{max(1, round(meters / 84))} min walk"  # 84 m/min ≈ 5 km/h

        def _surge_label(multiplier: float) -> str:
            return "no surge" if multiplier <= 1.0 else f"{multiplier}× surge"

        lines = []
        for i, r in enumerate(evaluated[:5]):
            tag = " *(your location)*" if r["is_original"] else ""
            lines.append(
                f"{i + 1}. **{r['label']}**{tag}\n"
                f"   Surge: {_surge_label(r['surge_multiplier'])} | "
                f"UberX est.: {r['price_display']} | "
                f"{_walk_time(r['walk_meters'])}"
            )

        no_surge = [r for r in evaluated[:5] if r["surge_multiplier"] <= 1.0]
        if no_surge:
            summary = (
                f"Found **{len(no_surge)}** nearby spot(s) with no surge pricing "
                f"within {radius_meters} m of {location}."
            )
        elif evaluated:
            best = evaluated[0]
            summary = (
                f"Surge is active across the area. "
                f"The best nearby option is **{best['label']}** at "
                f"{best['surge_multiplier']}× — consider a larger radius or waiting "
                f"a few minutes for demand to ease."
            )
        else:
            summary = "No fare estimates available for nearby locations."

        content = f"{summary}\n\n" + "\n".join(lines)

        log = make_log_entry(
            "find_pickup_locations",
            requested={"location": location, "dropoff": dropoff, "radius_meters": radius_meters},
            verified={"candidates_checked": len(evaluated)},
            executed={
                "platform": platform,
                "best_surge": evaluated[0]["surge_multiplier"] if evaluated else None,
            },
            outcome=(
                f"{len(no_surge)} no-surge options found out of {len(evaluated)} candidates"
                if evaluated else "no estimates returned"
            ),
        )

        return Command(
            update={
                "action_log": [log],
                "messages": [ToolMessage(content=content, tool_call_id=tool_call_id)],
            }
        )

    except Exception as exc:
        log = make_log_entry(
            "find_pickup_locations",
            requested={"location": location, "dropoff": dropoff, "radius_meters": radius_meters},
            verified={},
            executed={},
            outcome=f"error: {exc}",
        )
        return Command(
            update={
                "action_log": [log],
                "messages": [
                    ToolMessage(
                        content=f"Error finding nearby pickup locations: {exc}",
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )


ALL_TOOLS = [validate_address, set_platform, search_rides, suggest_ride,
             set_guest_info, book_ride, track_ride, cancel_ride,
             find_pickup_locations]
