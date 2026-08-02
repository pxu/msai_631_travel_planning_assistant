"""Shared state schema for the travel planning graph.

Preferences are collected via free-form extraction rather than one-field-
at-a-time questions: a field being ``None``/absent is how a node knows
whether it still needs to be extracted (preferences) or produced (outputs).
``collected_fields``/``missing_fields`` are a derived, display-only snapshot
recomputed on every pass through ``validate_preferences`` so they can never
drift from the underlying typed fields.
"""

from typing import Dict, List, Literal, Optional, TypedDict

ConversationStage = Literal[
    "collecting",
    "extracting",
    "awaiting_missing_fields",
    "ready",
    "awaiting_confirmation",
    "reviewing",
    "complete",
]

ConfirmationStatus = Literal["pending", "confirmed"]


class TravelState(TypedDict, total=False):
    # Raw conversational input
    last_user_input: Optional[str]

    # Collected from the traveler — required
    destination: Optional[str]
    trip_length_days: Optional[int]
    group_type: Optional[str]
    interests: Optional[str]
    budget_level: Optional[str]

    # Collected from the traveler — optional
    travel_style: Optional[str]
    travel_season: Optional[str]
    must_visit_attractions: Optional[str]

    # Gap-analysis / conversation bookkeeping
    collected_fields: Dict[str, str]
    missing_fields: List[str]
    conversation_stage: ConversationStage
    confirmation_status: Optional[ConfirmationStatus]
    extraction_attempts: int
    # Ordered subset of generate_* node names still needing to (re)run for
    # the current edit. Absent/None means "not in an edit dispatch" (the
    # initial generation pass uses its own fixed node order instead).
    regeneration_targets: Optional[List[str]]

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

OPTIONAL_PREFERENCE_FIELDS = (
    "travel_style",
    "travel_season",
    "must_visit_attractions",
)

ALL_PREFERENCE_FIELDS = REQUIRED_PREFERENCE_FIELDS + OPTIONAL_PREFERENCE_FIELDS

FIELD_LABELS = {
    "destination": "Destination",
    "trip_length_days": "Trip Length",
    "group_type": "Group Type",
    "interests": "Interests",
    "budget_level": "Budget",
    "travel_style": "Travel Style",
    "travel_season": "Travel Season",
    "must_visit_attractions": "Must-Visit Attractions",
}
