"""Shared state schema for the travel planning graph.

Each field is filled in by either the preference-collection node (via
human-in-the-loop interrupts) or one of the downstream generation nodes.
A field being ``None`` is how nodes know it still needs to be produced.
"""

from typing import Optional, TypedDict


class TravelState(TypedDict, total=False):
    # Collected from the traveler
    destination: Optional[str]
    trip_length_days: Optional[int]
    group_type: Optional[str]
    interests: Optional[str]
    budget_level: Optional[str]

    # Produced by the generation nodes
    itinerary: Optional[str]
    activities: Optional[str]
    transportation: Optional[str]
    budget_estimate: Optional[str]
    packing_list: Optional[str]


REQUIRED_PREFERENCE_FIELDS = (
    "destination",
    "trip_length_days",
    "group_type",
    "interests",
    "budget_level",
)
