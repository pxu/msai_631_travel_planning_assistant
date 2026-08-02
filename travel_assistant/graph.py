"""LangGraph state machine for the AI-Powered Travel Assistant Planner.

Preference collection is adaptive rather than a fixed questionnaire: the
user describes their trip in free text, ``extract_preferences`` turns that
into structured fields with one LLM call, and ``validate_preferences`` (pure
Python, no LLM call) decides — via a conditional edge — whether to loop back
for more information, show a confirmation summary, or proceed to
generation. This keeps the same principle the original design relied on:
the LLM only ever generates or extracts text for one step at a time; a
small local model is never asked to decide what happens next in the graph.

    START -> collect_initial_request -> extract_preferences
          -> validate_preferences -[missing]-> request_missing_fields -> extract_preferences (loop)
          -> validate_preferences -[complete]-> show_summary -> confirm_preferences
          -[edit]-> apply_preference_edit -> validate_preferences (loop)
          -[confirmed]-> generate_itinerary -> ... -> generate_packing -> review_plan
          -[edit]-> apply_plan_edit -> regenerate_affected -> review_plan (loop)
          -[confirmed]-> END
"""

import json
import re
from typing import Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from travel_assistant.state import (
    ALL_PREFERENCE_FIELDS,
    FIELD_LABELS,
    REQUIRED_PREFERENCE_FIELDS,
    TravelState,
)

# ---------------------------------------------------------------------------
# Extraction: turns free-form user text into structured preference fields
# ---------------------------------------------------------------------------

_EXTRACTION_SYSTEM_PROMPT = (
    "You extract structured travel preferences from a traveler's message. "
    "Return ONLY a JSON object with exactly these keys: destination, "
    "trip_length_days, group_type, interests, budget_level, travel_style, "
    "travel_season, must_visit_attractions. Use null for any field not "
    "explicitly mentioned in the message. Do NOT guess, assume, or fill in "
    "a typical/default value for a field just because it's missing — e.g. "
    "if the message doesn't say who is traveling, group_type MUST be null, "
    "not a guess like 'solo' or 'individual'. Only return a value you can "
    "point to specific words for in the message. trip_length_days must be "
    "a plain integer or null."
)

_EDIT_EXTRACTION_SYSTEM_PROMPT = (
    "You update structured travel preferences based on a traveler's edit "
    "request, given their currently known preferences. Return ONLY a JSON "
    "object with exactly these keys: destination, trip_length_days, "
    "group_type, interests, budget_level, travel_style, travel_season, "
    "must_visit_attractions. For each field: if the edit request changes or "
    "adds to it, return the FULL updated value combining the old value with "
    "the new information (e.g. if interests were 'food, museums' and the "
    "edit says 'add outdoor activities', return 'food, museums, outdoor "
    "activities', not just 'outdoor activities'). If the edit request does "
    "not mention a field at all, return null for it so it stays unchanged. "
    "trip_length_days must be a plain integer or null."
)

_MAX_EXTRACTION_ATTEMPTS = 3


def _extract_json(text: str) -> dict:
    """Parse the JSON object out of the model's raw output, tolerating a
    response that got cut off mid-object (a small model generating a long
    field list not infrequently stops before emitting the final closing
    brace). Scans from the first ``{`` tracking bracket depth and string
    state: if depth returns to zero, that's a complete object (possibly
    followed by trailing chatter, which is simply ignored); if the text
    ends while still inside the object, an unterminated string is closed,
    a dangling trailing comma is dropped, and every still-open bracket is
    closed in nesting order before parsing. Without this, a single
    truncated response would silently discard every field it contained,
    not just the one at the point of truncation.
    """
    start = text.find("{")
    if start == -1:
        return {}

    stack: list[str] = []
    in_string = False
    escape = False
    closers = {"{": "}", "[": "]"}

    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()
            if not stack:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    return {}

    # Reached the end of the text with the object still open — repair the
    # truncation rather than discarding everything extracted so far.
    repaired = text[start:]
    if in_string:
        repaired += '"'
    repaired = repaired.rstrip()
    if repaired.endswith(","):
        repaired = repaired[:-1]
    for opener in reversed(stack):
        repaired += closers[opener]

    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        return {}


def _clean_value(value):
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned or cleaned.lower() in ("null", "none", "n/a", "unknown", "unspecified"):
            return None
        return cleaned
    if isinstance(value, list):
        parts = [str(item).strip() for item in value if str(item).strip()]
        return ", ".join(parts) if parts else None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    return None


_WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15,
}


def _find_number(text: str):
    match = re.search(r"\d+", text or "")
    if match:
        return int(match.group())
    for word, number in _WORD_NUMBERS.items():
        if re.search(rf"\b{word}\b", text or "", re.IGNORECASE):
            return number
    return None


def _find_dollar_amount(text: str):
    match = re.search(r"\$\s?[\d,]+", text or "")
    return match.group().replace(" ", "") if match else None


def _coerce_trip_length(value, raw_text: str):
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        number = _find_number(value)
        if number is not None:
            return number
    return _find_number(raw_text)


# Keyword groups used to infer group_type directly from the raw message
# when the model doesn't return one. Checked in this order (family before
# couple) so "my wife and son" resolves to "family," not "couple" — the
# presence of a child-related word takes priority over a partner-only word.
_GROUP_TYPE_KEYWORDS = (
    ("family", ("son", "daughter", "kids", "kid", "children", "child", "family",
                "parents", "grandma", "grandpa", "grandmother", "grandfather")),
    ("couple", ("wife", "husband", "spouse", "partner", "girlfriend", "boyfriend", "couple")),
    ("friends", ("friends", "friend", "buddies", "buddy")),
    ("students", ("students", "classmates", "student")),
    ("solo", ("solo", "alone", "myself", "by myself", "byself", "just me")),
)


def _infer_group_type(raw_text: str):
    lowered = (raw_text or "").lower()
    for group, keywords in _GROUP_TYPE_KEYWORDS:
        for keyword in keywords:
            if re.search(rf"\b{re.escape(keyword)}\b", lowered):
                return group
    return None


# Destination names are trusted at face value: they're sometimes given as
# nicknames/aliases with no word overlap with a formal name (e.g. "the Big
# Apple" -> "New York"), so grounding would reject too many correct
# extractions. Every other field is a case where the model filling in a
# plausible-sounding value it invented (not read off the message) is the
# actual failure mode being guarded against, so they stay grounded.
_UNGROUNDED_FIELDS = {"destination"}

_STOPWORDS = {"a", "an", "the", "and", "or", "for", "to", "of", "in", "on", "with"}


def _is_grounded(value: str, raw_text: str) -> bool:
    """Reject a field value the model appears to have invented rather than
    read off the message. A small (~1B) model does not reliably follow a
    "don't guess" instruction, so this is a code-level backstop: for
    fields prone to being filled in with a plausible-sounding default
    (group_type, budget_level, travel_style, travel_season), require that
    at least one meaningful word from the extracted value actually appears
    in what the user typed.
    """
    words = [w for w in re.findall(r"[a-z0-9]+", value.lower()) if w not in _STOPWORDS]
    if not words:
        return True
    lowered_text = raw_text.lower()
    return any(word in lowered_text for word in words)


def _looks_like_specific_place(value: str) -> bool:
    """Heuristic for "is this a named place, not a generic theme?" A real
    place name the model extracts is reliably Title Case in its own output
    (e.g. "Eiffel Tower", "Statue of Liberty"); a generic theme it misfiled
    into the wrong field is lowercase (e.g. "food", "hiking", "mountain").
    """
    words = value.split()
    return any(word[:1].isupper() for word in words)


def _known_fields_context(state: TravelState) -> str:
    known = [f"{field}: {state[field]}" for field in ALL_PREFERENCE_FIELDS if state.get(field)]
    return "Currently known preferences:\n" + "\n".join(known) if known else "No preferences known yet."


def _merge_extracted(llm: BaseChatModel, state: TravelState, *, overwrite: bool) -> dict:
    """Extract preference fields from ``last_user_input`` via one LLM call.

    With ``overwrite=False`` (initial collection), a field already present
    in state is left untouched — a later reply that only mentions budget
    must not null out an earlier destination. With ``overwrite=True`` (the
    user is explicitly editing something), the model is given the current
    values as context and asked to return each field's full updated value
    (merging rather than replacing, e.g. "add outdoor activities" extends
    interests instead of overwriting it) — the result still replaces
    whatever was there before, but the *model* did the merging, not this
    function blindly stomping on prior state.
    """
    raw_text = state.get("last_user_input") or ""
    if overwrite:
        system_prompt = _EDIT_EXTRACTION_SYSTEM_PROMPT
        user_prompt = f"{_known_fields_context(state)}\n\nEdit request: {raw_text}"
    else:
        system_prompt = _EXTRACTION_SYSTEM_PROMPT
        user_prompt = raw_text
    text = _generate(llm, system_prompt, user_prompt, deterministic=True)
    parsed = _extract_json(text)

    updates: dict = {}
    for field in ALL_PREFERENCE_FIELDS:
        if not overwrite and state.get(field):
            continue
        value = _clean_value(parsed.get(field))
        if field == "trip_length_days":
            # Always attempt this, even if the model returned null — a
            # spelled-out or oddly formatted number ("seven-day") is common
            # enough that the model misses it while a raw-text scan won't.
            value = _coerce_trip_length(value, raw_text)
        if (
            not overwrite
            and field not in _UNGROUNDED_FIELDS
            and isinstance(value, str)
            and not _is_grounded(value, raw_text)
        ):
            # The model returned a plausible-sounding value with no actual
            # support in what the user typed (e.g. inventing "solo" for
            # group_type when the user never said who's traveling) — treat
            # it as not extracted rather than trusting a guess.
            value = None
        if field == "group_type":
            # "my wife and son" implies "family" without ever using that
            # word — a small model often fails to make that inference (or
            # the word-overlap grounding check above rejects it even when
            # the model gets it right, since "family" isn't literally in
            # the message). A direct keyword scan doesn't depend on the
            # model guessing correctly, and unlike a free-text field this
            # one has a small, fixed vocabulary that's safe to hard-code.
            inferred = _infer_group_type(raw_text)
            if not overwrite and value is None:
                value = inferred
            elif overwrite and inferred is not None and inferred != value:
                # During an edit, the model sometimes just echoes back the
                # previous group_type unchanged instead of recognizing the
                # edit request as a change (e.g. "will only go byself" left
                # group_type="family" untouched). The edit text itself is a
                # deliberate statement about who's traveling, so a keyword
                # match there overrides whatever the model returned.
                value = inferred
        if field == "budget_level" and value is None and not overwrite:
            # The model sometimes paraphrases an explicit dollar figure
            # into a category ("$4000" -> "mid-range") that then fails
            # grounding above, discarding a value the user was completely
            # explicit about. If the message contains a literal dollar
            # amount, prefer that over any category the model invents.
            dollar_amount = _find_dollar_amount(raw_text)
            if dollar_amount:
                value = dollar_amount
        if field == "must_visit_attractions" and isinstance(value, str) and not _looks_like_specific_place(value):
            # The model frequently misfiles a generic theme ("food",
            # "hiking") into this field instead of `interests` — reserve it
            # for actual named places and recover the value into `interests`
            # below rather than surfacing it as an attraction.
            if not overwrite and not state.get("interests") and "interests" not in updates:
                updates["interests"] = value
            value = None
        if value is not None:
            updates[field] = value
    return updates


def collect_initial_request(state: TravelState) -> dict:
    """Capture the user's opening free-form trip description.

    Normally ``app.py`` seeds ``last_user_input`` with the user's first chat
    message before the graph even starts, so this returns immediately. The
    ``interrupt()`` below only fires as a fallback (e.g. an empty first
    message), asking explicitly instead of extracting from nothing.
    """
    if state.get("last_user_input"):
        return {"conversation_stage": "collecting"}
    answer = interrupt(
        {"kind": "initial_prompt", "prompt": "Tell me about the trip you would like to plan."}
    )
    return {"last_user_input": answer, "conversation_stage": "collecting"}


def _make_extract_preferences_node(llm: BaseChatModel) -> Callable[[TravelState], dict]:
    def extract_preferences(state: TravelState) -> dict:
        updates = _merge_extracted(llm, state, overwrite=False)
        found_required_field = any(field in updates for field in REQUIRED_PREFERENCE_FIELDS)
        updates["extraction_attempts"] = 0 if found_required_field else state.get("extraction_attempts", 0) + 1
        updates["conversation_stage"] = "extracting"
        return updates

    return extract_preferences


def validate_preferences(state: TravelState) -> dict:
    """Pure gap analysis — no LLM call. The conditional edge out of this
    node reads ``missing_fields`` to decide where to go next."""
    missing = [field for field in REQUIRED_PREFERENCE_FIELDS if not state.get(field)]
    collected = {field: str(state[field]) for field in ALL_PREFERENCE_FIELDS if state.get(field)}
    stage = "ready" if not missing else "awaiting_missing_fields"
    return {"missing_fields": missing, "collected_fields": collected, "conversation_stage": stage}


def route_after_validation(state: TravelState) -> str:
    return "request_missing_fields" if state.get("missing_fields") else "show_summary"


def request_missing_fields(state: TravelState) -> dict:
    """The other interrupt point during collection. Accepts any number of
    missing fields answered in one reply — the resumed value flows back
    into ``extract_preferences``, which extracts as many as it can."""
    missing_fields = state.get("missing_fields", [])
    if state.get("extraction_attempts", 0) >= _MAX_EXTRACTION_ATTEMPTS and missing_fields:
        # Graceful degradation: a model that isn't extracting cleanly falls
        # back to asking for exactly one field, rather than looping forever.
        field = missing_fields[0]
        answer = interrupt(
            {
                "kind": "single_field_fallback",
                "field": field,
                "label": FIELD_LABELS.get(field, field),
                "collected_fields": state.get("collected_fields", {}),
            }
        )
    else:
        answer = interrupt(
            {
                "kind": "missing_fields",
                "collected_fields": state.get("collected_fields", {}),
                "missing_fields": missing_fields,
            }
        )
    return {"last_user_input": answer}


def show_summary(state: TravelState) -> dict:
    return {"conversation_stage": "awaiting_confirmation"}


# Short tokens are matched as whole words (via regex word boundaries) so
# e.g. "ok" doesn't false-positive inside "looking" or "book"; multi-word
# phrases are safe to match as plain substrings.
_CONFIRM_WORDS = ("yes", "yeah", "yep", "confirm", "correct", "finalize", "perfect", "ok", "okay", "sure")
_CONFIRM_PHRASES = (
    "sounds good", "looks good", "that's perfect", "that's right",
    "no changes", "no change", "all set", "nothing else", "great, thanks",
    "proceed", "go ahead",
)

# Catches affirmative sentence *patterns* the literal lists above miss —
# e.g. "all look right" (subject/quantifier + look/sound/be + a positive
# adjective) — rather than requiring every possible phrasing to be listed
# verbatim. A user saying this is unambiguously confirming, not asking for
# a change, so treating it as an edit request (which just re-runs
# extraction on it, finds no field to change, and re-shows the identical
# summary) makes the assistant look stuck in a loop.
_CONFIRM_PATTERN = re.compile(
    r"(\b(all|that|this|it|everything)\b.{0,15})?\b(look|looks|sound|sounds|is|are|seem|seems)\b"
    r".{0,10}\b(right|good|correct|fine|great|perfect)\b"
)


def _is_confirmation(text: str) -> bool:
    lowered = (text or "").strip().lower()
    if not lowered:
        return False
    if any(phrase in lowered for phrase in _CONFIRM_PHRASES):
        return True
    if _CONFIRM_PATTERN.search(lowered):
        return True
    return any(re.search(rf"\b{re.escape(word)}\b", lowered) for word in _CONFIRM_WORDS)


def confirm_preferences(state: TravelState) -> dict:
    answer = interrupt({"kind": "summary_confirmation", "collected_fields": state.get("collected_fields", {})})
    if _is_confirmation(answer):
        return {"confirmation_status": "confirmed", "conversation_stage": "ready", "last_user_input": answer}
    return {"last_user_input": answer, "confirmation_status": "pending"}


def route_after_confirmation(state: TravelState) -> str:
    return "generate_itinerary" if state.get("confirmation_status") == "confirmed" else "apply_preference_edit"


def _make_apply_preference_edit_node(llm: BaseChatModel) -> Callable[[TravelState], dict]:
    def apply_preference_edit(state: TravelState) -> dict:
        updates = _merge_extracted(llm, state, overwrite=True)
        updates["confirmation_status"] = "pending"
        updates["conversation_stage"] = "extracting"
        return updates

    return apply_preference_edit


def _generate(
    llm: BaseChatModel, system_prompt: str, user_prompt: str, *, deterministic: bool = False
) -> str:
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    # Extraction must be deterministic (greedy decoding): sampling at the
    # model's normal temperature can make it miss an unambiguous, explicitly
    # stated field (e.g. returning null for a destination the message
    # plainly names) on one run and get it right on the next. Content-
    # generation calls keep the model's configured temperature for varied,
    # natural-sounding text. ``do_sample`` must be nested under
    # ``pipeline_kwargs`` — that's the only shape HuggingFacePipeline reads
    # invoke-time generation overrides from; top-level kwargs are silently
    # ignored.
    kwargs = {"pipeline_kwargs": {"do_sample": False}} if deterministic else {}
    response = llm.invoke(messages, **kwargs)
    return response.content.strip()


_ASSISTANT_SYSTEM_PROMPT = (
    "You are a friendly, practical travel planning assistant. "
    "Keep responses concise and well organized."
)


def _optional_context_lines(state: TravelState) -> str:
    lines = []
    if state.get("travel_style"):
        lines.append(f"Travel style: {state['travel_style']}")
    if state.get("travel_season"):
        lines.append(f"Travel season: {state['travel_season']}")
    if state.get("must_visit_attractions"):
        lines.append(f"Must-visit attractions: {state['must_visit_attractions']}")
    return ("\n" + "\n".join(lines)) if lines else ""


def _itinerary_update(llm: BaseChatModel, state: TravelState) -> dict:
    prompt = (
        f"Destination: {state['destination']}\n"
        f"Trip length: {state['trip_length_days']} days\n"
        f"Traveler group: {state['group_type']}\n"
        f"Interests: {state['interests']}\n"
        f"Budget level: {state['budget_level']}"
        f"{_optional_context_lines(state)}\n\n"
        "Write a day-by-day itinerary for this trip, with a short heading per day."
    )
    text = _generate(llm, _ASSISTANT_SYSTEM_PROMPT, prompt)
    return {"itinerary": text}


def _make_itinerary_node(llm: BaseChatModel) -> Callable[[TravelState], dict]:
    return lambda state: _itinerary_update(llm, state)


def _activities_update(llm: BaseChatModel, state: TravelState) -> dict:
    prompt = (
        f"Destination: {state['destination']}\n"
        f"Interests: {state['interests']}\n"
        f"Traveler group: {state['group_type']}"
        f"{_optional_context_lines(state)}\n\n"
        "Recommend 5-8 specific attractions or activities that match these interests, "
        "as a bulleted list."
    )
    text = _generate(llm, _ASSISTANT_SYSTEM_PROMPT, prompt)
    return {"activities": text}


def _make_activities_node(llm: BaseChatModel) -> Callable[[TravelState], dict]:
    return lambda state: _activities_update(llm, state)


def _transportation_update(llm: BaseChatModel, state: TravelState) -> dict:
    prompt = (
        f"Destination: {state['destination']}\n"
        f"Trip length: {state['trip_length_days']} days\n"
        f"Traveler group: {state['group_type']}\n"
        f"Budget level: {state['budget_level']}\n\n"
        "Suggest how this traveler should get around at the destination "
        "(e.g. public transit, rideshare, rental car, walking) and why, in 3-4 sentences."
    )
    text = _generate(llm, _ASSISTANT_SYSTEM_PROMPT, prompt)
    return {"transportation": text}


def _make_transportation_node(llm: BaseChatModel) -> Callable[[TravelState], dict]:
    return lambda state: _transportation_update(llm, state)


# Rough, deterministic per-day rate by budget tier (USD), independent of the LLM
# so the estimate is stable and reproducible rather than an LLM guess.
_DAILY_RATE_BY_BUDGET = {"budget": 75, "moderate": 175, "luxury": 400}
_GROUP_MULTIPLIER = {"solo": 1, "couple": 2, "family": 4, "friends": 3, "students": 2}


def _estimate_budget(state: TravelState) -> int:
    daily_rate = _DAILY_RATE_BY_BUDGET.get(
        (state.get("budget_level") or "moderate").lower(), 175
    )
    multiplier = _GROUP_MULTIPLIER.get((state.get("group_type") or "solo").lower(), 1)
    days = state.get("trip_length_days") or 3
    return daily_rate * multiplier * days


def _budget_update(llm: BaseChatModel, state: TravelState) -> dict:
    estimate = _estimate_budget(state)
    prompt = (
        f"Destination: {state['destination']}\n"
        f"Trip length: {state['trip_length_days']} days\n"
        f"Traveler group: {state['group_type']}\n"
        f"Budget level: {state['budget_level']}\n"
        f"Rough total budget estimate: ${estimate}\n\n"
        "Write a 2-3 sentence budget summary that presents this estimate and briefly "
        "breaks down what it should roughly cover (lodging, food, activities, local "
        "transport). Make clear this is a rough estimate, not a live price."
    )
    text = _generate(llm, _ASSISTANT_SYSTEM_PROMPT, prompt)
    return {"budget_estimate": text}


def _make_budget_node(llm: BaseChatModel) -> Callable[[TravelState], dict]:
    return lambda state: _budget_update(llm, state)


def _packing_update(llm: BaseChatModel, state: TravelState) -> dict:
    prompt = (
        f"Destination: {state['destination']}\n"
        f"Trip length: {state['trip_length_days']} days\n"
        f"Traveler group: {state['group_type']}\n"
        f"Interests: {state['interests']}"
        f"{_optional_context_lines(state)}\n\n"
        "Generate a packing checklist for this trip as a bulleted list."
    )
    text = _generate(llm, _ASSISTANT_SYSTEM_PROMPT, prompt)
    return {"packing_list": text}


def _make_packing_node(llm: BaseChatModel) -> Callable[[TravelState], dict]:
    return lambda state: _packing_update(llm, state)


# ---------------------------------------------------------------------------
# Review + edit/regenerate loop (post-generation)
# ---------------------------------------------------------------------------

_CANONICAL_GENERATION_ORDER = (
    "generate_itinerary",
    "generate_activities",
    "generate_transportation",
    "generate_budget",
    "generate_packing",
)

_FIELD_TO_REGEN_NODES = {
    "destination": ("generate_itinerary", "generate_activities", "generate_transportation", "generate_packing"),
    "trip_length_days": ("generate_itinerary", "generate_budget", "generate_packing"),
    "group_type": ("generate_itinerary", "generate_transportation", "generate_budget"),
    "interests": ("generate_activities", "generate_packing"),
    "budget_level": ("generate_budget",),
    "travel_style": ("generate_itinerary", "generate_activities"),
    "travel_season": ("generate_packing",),
    "must_visit_attractions": ("generate_itinerary", "generate_activities"),
}

_REGEN_DISPATCH = {
    "generate_itinerary": _itinerary_update,
    "generate_activities": _activities_update,
    "generate_transportation": _transportation_update,
    "generate_budget": _budget_update,
    "generate_packing": _packing_update,
}


def review_plan(state: TravelState) -> dict:
    answer = interrupt(
        {"kind": "plan_review", "prompt": "Would you like any changes, or shall I finalize this plan?"}
    )
    if _is_confirmation(answer):
        return {"confirmation_status": "confirmed", "conversation_stage": "complete", "last_user_input": answer}
    return {"last_user_input": answer, "conversation_stage": "reviewing", "confirmation_status": "pending"}


def route_after_review(state: TravelState) -> str:
    return "end" if state.get("confirmation_status") == "confirmed" else "apply_plan_edit"


def _make_apply_plan_edit_node(llm: BaseChatModel) -> Callable[[TravelState], dict]:
    def apply_plan_edit(state: TravelState) -> dict:
        updates = _merge_extracted(llm, state, overwrite=True)
        changed_fields = [
            field for field in ALL_PREFERENCE_FIELDS
            if field in updates and str(updates[field]) != str(state.get(field))
        ]

        targets: list = []
        for field in changed_fields:
            for node_name in _FIELD_TO_REGEN_NODES.get(field, ()):
                if node_name not in targets:
                    targets.append(node_name)
        ordered_targets = [name for name in _CANONICAL_GENERATION_ORDER if name in targets]

        updates["regeneration_targets"] = ordered_targets
        updates["conversation_stage"] = "reviewing"
        return updates

    return apply_plan_edit


def _make_regenerate_affected_node(llm: BaseChatModel) -> Callable[[TravelState], dict]:
    def regenerate_affected(state: TravelState) -> dict:
        targets = state.get("regeneration_targets") or []
        merged_state = dict(state)
        updates: dict = {}
        for name in targets:
            update_fn = _REGEN_DISPATCH.get(name)
            if update_fn is None:
                continue
            result = update_fn(llm, merged_state)
            updates.update(result)
            merged_state.update(result)
        updates["regeneration_targets"] = []
        return updates

    return regenerate_affected


def build_graph(llm: BaseChatModel, checkpointer: BaseCheckpointSaver):
    """Compile the travel-planning StateGraph with the given chat model."""
    builder = StateGraph(TravelState)

    builder.add_node("collect_initial_request", collect_initial_request)
    builder.add_node("extract_preferences", _make_extract_preferences_node(llm))
    builder.add_node("validate_preferences", validate_preferences)
    builder.add_node("request_missing_fields", request_missing_fields)
    builder.add_node("show_summary", show_summary)
    builder.add_node("confirm_preferences", confirm_preferences)
    builder.add_node("apply_preference_edit", _make_apply_preference_edit_node(llm))
    builder.add_node("generate_itinerary", _make_itinerary_node(llm))
    builder.add_node("generate_activities", _make_activities_node(llm))
    builder.add_node("generate_transportation", _make_transportation_node(llm))
    builder.add_node("generate_budget", _make_budget_node(llm))
    builder.add_node("generate_packing", _make_packing_node(llm))
    builder.add_node("review_plan", review_plan)
    builder.add_node("apply_plan_edit", _make_apply_plan_edit_node(llm))
    builder.add_node("regenerate_affected", _make_regenerate_affected_node(llm))

    builder.add_edge(START, "collect_initial_request")
    builder.add_edge("collect_initial_request", "extract_preferences")
    builder.add_edge("extract_preferences", "validate_preferences")
    builder.add_conditional_edges(
        "validate_preferences",
        route_after_validation,
        {"request_missing_fields": "request_missing_fields", "show_summary": "show_summary"},
    )
    builder.add_edge("request_missing_fields", "extract_preferences")
    builder.add_edge("show_summary", "confirm_preferences")
    builder.add_conditional_edges(
        "confirm_preferences",
        route_after_confirmation,
        {"generate_itinerary": "generate_itinerary", "apply_preference_edit": "apply_preference_edit"},
    )
    builder.add_edge("apply_preference_edit", "validate_preferences")
    builder.add_edge("generate_itinerary", "generate_activities")
    builder.add_edge("generate_activities", "generate_transportation")
    builder.add_edge("generate_transportation", "generate_budget")
    builder.add_edge("generate_budget", "generate_packing")
    builder.add_edge("generate_packing", "review_plan")
    builder.add_conditional_edges(
        "review_plan",
        route_after_review,
        {"end": END, "apply_plan_edit": "apply_plan_edit"},
    )
    builder.add_edge("apply_plan_edit", "regenerate_affected")
    builder.add_edge("regenerate_affected", "review_plan")

    return builder.compile(checkpointer=checkpointer)


def format_summary(state: TravelState) -> str:
    """Render the final state as one chat message for the user."""
    return (
        f"## Your trip to {state['destination']}\n\n"
        f"**Itinerary**\n{state['itinerary']}\n\n"
        f"**Activities**\n{state['activities']}\n\n"
        f"**Getting around**\n{state['transportation']}\n\n"
        f"**Estimated budget**\n{state['budget_estimate']}\n\n"
        f"**Packing checklist**\n{state['packing_list']}\n"
    )
