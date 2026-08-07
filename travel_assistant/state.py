"""Shared state schema for the travel planning graph.

Preferences are collected via free-form extraction rather than one-field-
at-a-time questions: a field being ``None``/absent is how a node knows
whether it still needs to be extracted (preferences) or produced (outputs).
``collected_fields``/``missing_fields`` are a derived, display-only snapshot
recomputed on every pass through ``validate_preferences`` so they can never
drift from the underlying typed fields.
"""

from typing import Literal, TypedDict

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

# Closed vocabularies. These two fields index into rate/multiplier tables in
# `graph.py`, so an un-normalized value ("mid-range", "family of four") would
# silently fall through to a default and cost the trip at the wrong rate.
# Keeping them as literals makes the set of legal values a single source of
# truth shared by extraction, validation and estimation.
BudgetLevel = Literal["budget", "moderate", "luxury"]
GroupType = Literal["solo", "couple", "family", "friends", "students"]

BUDGET_LEVELS: tuple[BudgetLevel, ...] = ("budget", "moderate", "luxury")
GROUP_TYPES: tuple[GroupType, ...] = ("solo", "couple", "family", "friends", "students")


class TravelState(TypedDict, total=False):
    # Raw conversational input
    last_user_input: str | None

    # Collected from the traveler — required
    destination: str | None
    trip_length_days: int | None
    # Normalized to one of GROUP_TYPES / BUDGET_LEVELS at extraction time; a
    # value that can't be normalized is left None so the gap analysis asks
    # for it, rather than being carried forward and silently defaulted at
    # estimate time.
    group_type: GroupType | None
    interests: str | None
    budget_level: BudgetLevel | None

    # Collected from the traveler — optional
    travel_style: str | None
    travel_season: str | None
    must_visit_attractions: str | None

    # An explicit total the traveler named ("$4000"). Captured separately
    # from `budget_level` so a stated number is used as the estimate instead
    # of being flattened into a tier and re-derived from a rate table.
    budget_total_usd: int | None

    # Gap-analysis / conversation bookkeeping
    collected_fields: dict[str, str]
    missing_fields: list[str]
    conversation_stage: ConversationStage
    confirmation_status: ConfirmationStatus | None
    extraction_attempts: int
    # Ordered subset of generate_* node names still needing to (re)run for
    # the current edit. Absent/None means "not in an edit dispatch" (the
    # initial generation pass uses its own fixed node order instead).
    regeneration_targets: list[str] | None

    # Produced by the generation nodes
    itinerary: str | None
    activities: str | None
    transportation: str | None
    budget_estimate: str | None
    packing_list: str | None


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

# `budget_total_usd` is deliberately NOT a preference field: it is never
# asked for (a tier is), so it must stay out of `missing_fields` and out of
# the extraction prompt's key list. But it is still user-supplied state that
# has to be echoed back in the summary and diffed on an edit — a user who
# changes only their dollar amount must see the budget section regenerate.
# This tuple is what those two concerns iterate.
DISPLAYED_FIELDS = (*ALL_PREFERENCE_FIELDS, "budget_total_usd")

FIELD_LABELS = {
    "destination": "Destination",
    "trip_length_days": "Trip Length",
    "group_type": "Group Type",
    "interests": "Interests",
    "budget_level": "Budget",
    "travel_style": "Travel Style",
    "travel_season": "Travel Season",
    "must_visit_attractions": "Must-Visit Attractions",
    "budget_total_usd": "Stated Total Budget",
}
