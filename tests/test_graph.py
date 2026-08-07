"""Tests for the adaptive travel-planning LangGraph pipeline.

Uses a fake chat model so these run without a GPU or downloading the real
local model. The fake model branches on the system prompt: extraction
calls get a small regex-based fake extractor (tailored to the exact
wording used in these tests, standing in for the real LLM's JSON output);
generation calls get an echoed snippet, as before. Generation prompts are
recorded so tests can verify *which* sections got (re)generated without
depending on the fake model's exact echoed text.
"""

import json
import re

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from travel_assistant.graph import (
    _estimate_budget,
    _extract_json,
    _find_duration_days,
    _is_confirmation,
    _normalize_budget_level,
    build_graph,
    format_summary,
)


@pytest.mark.parametrize(
    "text, expected",
    [
        # A count next to its unit, in every shape the eval set turned up.
        ("I want a 7-day family trip", 7),
        ("I want a seven-day family trip to Japan.", 7),
        ("Five days in Rome", 5),
        ("About 12 days.", 12),
        ("3 nights in Vegas", 3),
        # Weeks convert; "a week" has no digit at all.
        ("A week in Athens with my parents", 7),
        ("Two weeks across Portugal and Spain", 14),
        # An adjective may sit between the count and the unit.
        ("My partner and I want 5 high-end days in Dubai.", 5),
        ("A relaxed, slow-paced 9 days in Bali", 9),
        # A number with no duration unit is NOT a trip length. Scanning for
        # any digit is how "Group of 4" became a 4-day trip — a
        # hallucination, the worst extraction failure because nothing
        # downstream can tell the value was invented.
        ("Group of 4 going to Tokyo, we love ramen", None),
        ("Hi there, can you help me?", None),
        # "weekend" is not "week".
        ("A long weekend in Paris with my girlfriend", None),
        # An earlier number must not shadow the real one, in either direction.
        ("Group of 4 people going to Tokyo for 3 days", 3),
        ("Group of 4 going to Tokyo for a week", 7),
    ],
)
def test_find_duration_days(text, expected):
    assert _find_duration_days(text) == expected


class FakeMessage:
    def __init__(self, content):
        self.content = content


def _fake_extract(text: str) -> dict:
    result = {
        "destination": None,
        "trip_length_days": None,
        "group_type": None,
        "interests": None,
        "budget_level": None,
        "travel_style": None,
        "travel_season": None,
        "must_visit_attractions": None,
    }

    match = re.search(r"trip to ([A-Za-z]+)", text)
    if match:
        result["destination"] = match.group(1)

    match = re.search(r"(\d+)[- ]day", text.lower()) or re.search(r"(\d+)\s+days", text.lower())
    if match:
        result["trip_length_days"] = int(match.group(1))

    for group in ("family", "couple", "solo", "friends", "students"):
        if group in text.lower():
            result["group_type"] = group
            break

    match = re.search(r"(?:we love|i love|interested in) ([a-zA-Z ]+?)[.,]", text, re.IGNORECASE)
    if match:
        result["interests"] = match.group(1).strip()

    for budget in ("luxury", "moderate", "budget"):
        if budget in text.lower():
            result["budget_level"] = budget
            break

    return result


def _fake_extract_edit(user_prompt: str) -> dict:
    """Simulates an LLM's merge-aware edit extraction: known-field values
    come from the "Currently known preferences" context block, and the
    result combines them with the edit request rather than just re-running
    the plain extractor on the edit text alone (which would lose context,
    e.g. "add outdoor activities" has no "interested in ..." phrasing for
    the plain extractor to find)."""
    context, _, edit_text = user_prompt.partition("Edit request:")
    existing = dict(re.findall(r"^(\w+): (.+)$", context, re.MULTILINE))

    result = {field: None for field in _fake_extract("")}

    match = re.search(r"budget to (budget|moderate|luxury)", edit_text, re.IGNORECASE)
    if match:
        result["budget_level"] = match.group(1).lower()

    match = re.search(r"(\d+)[- ]day", edit_text.lower())
    if match:
        result["trip_length_days"] = int(match.group(1))

    match = re.search(r"add (?:more )?([a-zA-Z ]+?)(?:\.|$)", edit_text, re.IGNORECASE)
    if match:
        addition = match.group(1).strip()
        base = existing.get("interests")
        result["interests"] = f"{base}, {addition}" if base else addition

    return result


class FakeChatModel:
    """Extracts via a fake regex parser; echoes a prompt snippet for generation."""

    def __init__(self):
        self.generation_prompts = []

    def invoke(self, messages, **kwargs):
        system_prompt = messages[0].content
        user_prompt = messages[-1].content

        if "update structured travel preferences" in system_prompt:
            return FakeMessage(content=json.dumps(_fake_extract_edit(user_prompt)))

        if "extract structured travel preferences" in system_prompt:
            return FakeMessage(content=json.dumps(_fake_extract(user_prompt)))

        self.generation_prompts.append(user_prompt)
        return FakeMessage(content=f"[fake output for] {user_prompt[:40]}")


@pytest.fixture
def llm():
    return FakeChatModel()


@pytest.fixture
def graph(llm):
    return build_graph(llm, checkpointer=InMemorySaver())


def _confirm(graph, config):
    return graph.invoke(Command(resume="yes, that all looks correct"), config=config)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("all look right", True),
        ("looks right", True),
        ("everything looks good", True),
        ("that is fine", True),
        ("yes", True),
        ("no, change the budget to luxury", False),
        ("make it 10 days", False),
        ("increase my budget", False),
    ],
)
def test_is_confirmation_recognizes_affirmative_phrasing_patterns(text, expected):
    """Regression: 'all look right' previously fell through every literal
    word/phrase in _CONFIRM_WORDS/_CONFIRM_PHRASES, so it was treated as an
    edit request. The edit-extraction model then just echoed back the
    unchanged fields, and the identical summary got shown again — from the
    user's perspective, the assistant looked stuck ignoring a clear
    confirmation. A regex pattern for "<subject> look/is/sound <positive
    adjective>" now catches phrasings like this without needing every
    variant listed verbatim."""
    assert _is_confirmation(text) is expected


def test_single_message_extracts_all_required_fields_and_skips_to_summary(graph):
    config = {"configurable": {"thread_id": "t1"}}
    message = (
        "I want a 7-day family trip to Japan. We love food and theme parks. Our budget is moderate."
    )

    result = graph.invoke({"last_user_input": message}, config=config)

    assert "__interrupt__" in result
    payload = result["__interrupt__"][0].value
    assert payload["kind"] == "summary_confirmation"
    collected = payload["collected_fields"]
    assert collected["destination"] == "Japan"
    assert collected["trip_length_days"] == "7"
    assert collected["group_type"] == "family"
    assert collected["interests"] == "food and theme parks"
    assert collected["budget_level"] == "moderate"


def test_full_conversation_reaches_review_and_produces_all_outputs(graph):
    config = {"configurable": {"thread_id": "t2"}}
    message = (
        "I want a 7-day family trip to Japan. We love food and theme parks. Our budget is moderate."
    )

    graph.invoke({"last_user_input": message}, config=config)
    result = _confirm(graph, config)

    assert "__interrupt__" in result
    assert result["__interrupt__"][0].value["kind"] == "plan_review"

    result = graph.invoke(Command(resume="no changes, finalize"), config=config)

    assert "__interrupt__" not in result
    assert result["destination"] == "Japan"
    assert result["trip_length_days"] == 7
    assert result["group_type"] == "family"
    assert result["budget_level"] == "moderate"
    for field in ("itinerary", "activities", "transportation", "budget_estimate", "packing_list"):
        assert result[field]

    summary = format_summary(result)
    assert "Japan" in summary


def test_multi_field_reply_fills_all_remaining_missing_fields(graph):
    config = {"configurable": {"thread_id": "t3"}}

    result = graph.invoke({"last_user_input": "Family trip to Japan."}, config=config)

    assert "__interrupt__" in result
    payload = result["__interrupt__"][0].value
    assert payload["kind"] == "missing_fields"
    assert set(payload["missing_fields"]) == {"trip_length_days", "interests", "budget_level"}
    assert payload["collected_fields"]["destination"] == "Japan"
    assert payload["collected_fields"]["group_type"] == "family"

    result = graph.invoke(
        Command(resume="5 days, moderate budget, interested in food and nature."), config=config
    )

    assert "__interrupt__" in result
    payload = result["__interrupt__"][0].value
    assert payload["kind"] == "summary_confirmation"
    collected = payload["collected_fields"]
    assert collected["trip_length_days"] == "5"
    assert collected["budget_level"] == "moderate"
    assert collected["interests"] == "food and nature"


def test_fields_already_known_are_never_asked_again(graph):
    config = {"configurable": {"thread_id": "t4"}}

    result = graph.invoke(
        {
            "destination": "Rome",
            "trip_length_days": 4,
            "last_user_input": "I love hiking and museums, budget is luxury, traveling as a couple.",
        },
        config=config,
    )

    assert "__interrupt__" in result
    payload = result["__interrupt__"][0].value
    # Reaching summary_confirmation directly (not missing_fields) proves
    # destination/trip_length_days were never re-asked.
    assert payload["kind"] == "summary_confirmation"
    collected = payload["collected_fields"]
    assert collected["destination"] == "Rome"
    assert collected["trip_length_days"] == "4"
    assert collected["group_type"] == "couple"
    assert collected["interests"] == "hiking and museums"
    assert collected["budget_level"] == "luxury"


def test_editing_a_field_during_confirmation_updates_only_that_field(graph):
    config = {"configurable": {"thread_id": "t5"}}
    message = (
        "I want a 7-day family trip to Japan. We love food and theme parks. Our budget is moderate."
    )

    graph.invoke({"last_user_input": message}, config=config)
    result = graph.invoke(Command(resume="actually change the budget to luxury"), config=config)

    assert "__interrupt__" in result
    payload = result["__interrupt__"][0].value
    assert payload["kind"] == "summary_confirmation"
    collected = payload["collected_fields"]
    assert collected["budget_level"] == "luxury"
    assert collected["destination"] == "Japan"  # untouched by the edit


def test_post_generation_edit_regenerates_only_the_affected_section(graph, llm):
    config = {"configurable": {"thread_id": "t6"}}
    message = (
        "I want a 7-day family trip to Japan. We love food and theme parks. Our budget is moderate."
    )

    graph.invoke({"last_user_input": message}, config=config)
    result = _confirm(graph, config)
    assert result["__interrupt__"][0].value["kind"] == "plan_review"

    def count(marker):
        return sum(marker in prompt for prompt in llm.generation_prompts)

    budget_calls_before = count("Rough total budget estimate")
    itinerary_calls_before = count("Write a day-by-day itinerary")
    activities_calls_before = count("Recommend 5-8 specific attractions")
    packing_calls_before = count("Generate a packing checklist")

    result = graph.invoke(Command(resume="please increase my budget to luxury"), config=config)

    assert "__interrupt__" in result
    assert result["__interrupt__"][0].value["kind"] == "plan_review"

    assert count("Rough total budget estimate") == budget_calls_before + 1
    assert count("Write a day-by-day itinerary") == itinerary_calls_before
    assert count("Recommend 5-8 specific attractions") == activities_calls_before
    assert count("Generate a packing checklist") == packing_calls_before

    result = graph.invoke(Command(resume="no changes, finalize"), config=config)
    assert "__interrupt__" not in result
    assert result["budget_level"] == "luxury"


class InterestsMisfiledChatModel:
    """Simulates a small model that misfiles a generic theme (e.g. 'food')
    into must_visit_attractions instead of interests — the failure mode
    reported where 'delicus food' extracted interests=['delicious']
    (ungrounded, since 'delicious' != 'delicus') while the real word 'food'
    landed in must_visit_attractions, so interests stayed empty forever."""

    def invoke(self, messages, **kwargs):
        system_prompt = messages[0].content
        user_prompt = messages[-1].content

        if "extract structured travel preferences" in system_prompt:
            return FakeMessage(
                content=json.dumps(
                    {
                        "destination": None,
                        "trip_length_days": None,
                        "group_type": None,
                        "interests": "delicious",
                        "budget_level": None,
                        "travel_style": None,
                        "travel_season": None,
                        "must_visit_attractions": "food",
                    }
                )
            )

        return FakeMessage(content=f"[fake output for] {user_prompt[:40]}")


def test_generic_theme_misfiled_as_attraction_is_recovered_into_interests():
    graph = build_graph(InterestsMisfiledChatModel(), checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "t13"}}

    result = graph.invoke(
        {
            "destination": "Seattle",
            "trip_length_days": 3,
            "group_type": "family",
            "budget_level": "$4000",
            "last_user_input": "I would like to have delicus food",
        },
        config=config,
    )

    assert "__interrupt__" in result
    payload = result["__interrupt__"][0].value
    assert payload["kind"] == "summary_confirmation"
    assert payload["collected_fields"].get("interests") == "food"
    assert "must_visit_attractions" not in payload["collected_fields"]


class GroupTypeEchoChatModel:
    """Simulates a small model that, during an edit, echoes back the
    previous group_type unchanged instead of recognizing the edit text as
    a change — the failure mode reported where 'will only go byself' left
    group_type='family' untouched."""

    def invoke(self, messages, **kwargs):
        system_prompt = messages[0].content
        user_prompt = messages[-1].content

        if "update structured travel preferences" in system_prompt:
            return FakeMessage(
                content=json.dumps(
                    {
                        "destination": "Seattle",
                        "trip_length_days": 4,
                        "group_type": "family",  # stale — should have become "solo"
                        "interests": "delicious food",
                        "budget_level": "$4000",
                        "travel_style": None,
                        "travel_season": None,
                        "must_visit_attractions": None,
                    }
                )
            )

        return FakeMessage(content=f"[fake output for] {user_prompt[:40]}")


def test_group_type_edit_overrides_models_stale_echoed_value():
    graph = build_graph(GroupTypeEchoChatModel(), checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "t15"}}

    graph.invoke(
        {
            "destination": "Seattle",
            "trip_length_days": 3,
            "group_type": "family",
            "interests": "delicious food",
            "budget_level": "$4000",
            "last_user_input": "placeholder",
        },
        config=config,
    )
    result = graph.invoke(
        Command(resume="Oh, I think I need 4 days and will only go byself"), config=config
    )

    assert "__interrupt__" in result
    payload = result["__interrupt__"][0].value
    assert payload["kind"] == "summary_confirmation"
    collected = payload["collected_fields"]
    assert collected["group_type"] == "solo"
    assert collected["trip_length_days"] == "4"


class SpecificAttractionChatModel:
    """A model that correctly names a real, specific place — must NOT be
    swept into the misfiled-theme recovery path."""

    def invoke(self, messages, **kwargs):
        system_prompt = messages[0].content
        user_prompt = messages[-1].content

        if "extract structured travel preferences" in system_prompt:
            return FakeMessage(
                content=json.dumps(
                    {
                        "destination": None,
                        "trip_length_days": None,
                        "group_type": None,
                        "interests": "sightseeing",
                        "budget_level": None,
                        "travel_style": None,
                        "travel_season": None,
                        "must_visit_attractions": "Eiffel Tower",
                    }
                )
            )

        return FakeMessage(content=f"[fake output for] {user_prompt[:40]}")


def test_specific_named_place_is_kept_as_an_attraction_not_recovered():
    graph = build_graph(SpecificAttractionChatModel(), checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "t14"}}

    result = graph.invoke(
        {
            "destination": "Paris",
            "trip_length_days": 3,
            "group_type": "family",
            "budget_level": "moderate",
            "last_user_input": "we want to visit the Eiffel Tower for sightseeing",
        },
        config=config,
    )

    assert "__interrupt__" in result
    payload = result["__interrupt__"][0].value
    assert payload["collected_fields"].get("must_visit_attractions") == "Eiffel Tower"
    assert payload["collected_fields"].get("interests") == "sightseeing"


def test_extract_json_repairs_a_response_truncated_mid_object():
    """Regression: a small model generating a long field list sometimes
    stops before the closing brace. Previously the regex-based parser
    required a fully balanced {...} match, so a truncated response
    silently discarded every field, not just the one being generated when
    it cut off (e.g. 'we like food' lost destination/trip_length_days too,
    even though those weren't part of the truncated tail)."""
    truncated = (
        '{\n  "destination": null,\n  "trip_length_days": null,\n  '
        '"group_type": null,\n  "interests": ["food"],\n  "budget_level": null,\n  '
        '"travel_style": null,\n  "travel_season": null,\n  '
        '"must_visit_attractions": ["food"]'
    )
    assert _extract_json(truncated) == {
        "destination": None,
        "trip_length_days": None,
        "group_type": None,
        "interests": ["food"],
        "budget_level": None,
        "travel_style": None,
        "travel_season": None,
        "must_visit_attractions": ["food"],
    }


class DollarParaphraseChatModel:
    """Simulates a small model that paraphrases an explicit dollar figure
    into a vague category ('$4000' -> 'mid-range') instead of returning
    the literal amount — the failure mode where an explicit budget gets
    dropped because the paraphrase doesn't ground against the message."""

    def __init__(self):
        self.generation_prompts = []

    def invoke(self, messages, **kwargs):
        system_prompt = messages[0].content
        user_prompt = messages[-1].content

        if "extract structured travel preferences" in system_prompt:
            return FakeMessage(
                content=json.dumps(
                    {
                        "destination": None,
                        "trip_length_days": None,
                        "group_type": None,
                        "interests": None,
                        "budget_level": "mid-range",
                        "travel_style": None,
                        "travel_season": None,
                        "must_visit_attractions": None,
                    }
                )
            )

        self.generation_prompts.append(user_prompt)
        return FakeMessage(content=f"[fake output for] {user_prompt[:40]}")


def test_explicit_dollar_amount_is_captured_and_used_as_the_estimate():
    """A stated total is kept as a number in its own field, and the derived
    tier comes from that number rather than the model's vague paraphrase.

    The model here says "mid-range". Left alone that would normalize to
    "moderate" and cost a family of four for 3 days at $2,100 — quietly
    contradicting the $4,000 the user just named. The stated total wins.
    """
    graph = build_graph(DollarParaphraseChatModel(), checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "t12"}}

    result = graph.invoke(
        {
            "destination": "Seattle",
            "trip_length_days": 3,
            "group_type": "family",
            "last_user_input": "we have around $4000 budget",
        },
        config=config,
    )

    assert "__interrupt__" in result
    payload = result["__interrupt__"][0].value
    state = graph.get_state(config).values

    # The number is preserved exactly, not flattened into a tier...
    assert state["budget_total_usd"] == 4000
    # ...and it is echoed back, so the summary shows the figure the user
    # actually named rather than only the tier inferred from it.
    assert payload["collected_fields"]["budget_total_usd"] == "4000"
    # The tier is derived from that number: $4000 / (3 days x 4 people) =
    # $333/person/day, above the luxury threshold. Note the model said
    # "mid-range"; the stated total overrules it.
    assert state["budget_level"] == "luxury"
    # The estimate is the user's own number, not a rate-table recomputation.
    assert _estimate_budget(state) == 4000


class InventedAmountChatModel(FakeChatModel):
    """Narrates the budget with a dollar figure it made up — the observed
    1B behaviour ("approximately $1,200" against a stated $5,000)."""

    def invoke(self, messages, **kwargs):
        if "budget summary" in messages[-1].content:
            return FakeMessage(content="Plan for approximately $1,200 to cover lodging and food.")
        return super().invoke(messages, **kwargs)


def test_budget_narration_contradicting_the_figure_is_replaced(monkeypatch):
    """The budget is the output users are most likely to sanity-check, so a
    narration quoting a number that disagrees with the authoritative total
    must not reach them — the prompt says "don't invent amounts" and a small
    model ignores it."""
    from travel_assistant.graph import _budget_update

    state = {
        "destination": "Seattle",
        "trip_length_days": 3,
        "group_type": "family",
        "budget_level": "luxury",
        "budget_total_usd": 5000,
    }
    text = _budget_update(InventedAmountChatModel(), state)["budget_estimate"]

    assert "$1,200" not in text
    assert "$5,000" in text
    assert "rough planning number" in text


def test_budget_narration_quoting_the_right_figure_is_kept():
    """The backstop must not fire on a correct narration."""

    # budget tier = $75/day x family(4) x 3 days = $900.
    class GoodModel(FakeChatModel):
        def invoke(self, messages, **kwargs):
            if "budget summary" in messages[-1].content:
                return FakeMessage(content="Budget about $900 total for the trip.")
            return super().invoke(messages, **kwargs)

    from travel_assistant.graph import _budget_update

    state = {
        "destination": "Rome",
        "trip_length_days": 3,
        "group_type": "family",
        "budget_level": "budget",
    }
    text = _budget_update(GoodModel(), state)["budget_estimate"]
    assert text == "Budget about $900 total for the trip."


@pytest.mark.parametrize(
    "text, authoritative, expected",
    [
        ("about $5,000 total", 5000, []),
        ("about $5000 total", 5000, []),
        ("roughly $1,200", 5000, [1200]),
        ("$1,800 - $2,200 per night", 5000, [1800, 2200]),
        ("no figures at all here", 5000, []),
    ],
)
def test_conflicting_amounts(text, authoritative, expected):
    from travel_assistant.graph import _conflicting_amounts

    assert _conflicting_amounts(text, authoritative) == expected


def test_stated_total_suppresses_the_budget_tier_question():
    """Regression found by running the app: the assistant replied "I have
    your stated total budget ($5,000). Could you tell me your trip length
    and budget?" — acknowledging the figure and asking for it in the same
    breath. The tier is derivable from the total once trip length and group
    land (both required anyway), so it must not be requested separately."""
    graph = build_graph(FakeChatModel(), checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "t12c"}}

    result = graph.invoke(
        {"last_user_input": "Trip to Seattle with my family. I have $5000. We love food."},
        config=config,
    )
    payload = result["__interrupt__"][0].value
    assert payload["collected_fields"]["budget_total_usd"] == "5000"
    assert "budget_level" not in payload["missing_fields"]

    # Once trip length arrives the tier derives itself: $5000 / (5 days x 4
    # people) = $250/person/day -> moderate. The estimate stays the user's
    # own figure rather than a rate-table recomputation.
    graph.invoke(Command(resume="5 days"), config=config)
    state = graph.get_state(config).values
    assert state["budget_level"] == "moderate"
    assert _estimate_budget(state) == 5000


@pytest.mark.parametrize(
    "raw, expected",
    [
        # "budget" as the NOUN for the field — not a tier choice. Reading it
        # as the cheap tier would contradict the stated number and skip the
        # derivation from it.
        ("we have around $4000 budget", None),
        ("my budget is whatever", None),
        ("what's your budget?", None),
        # "budget" as the TIER, unambiguously.
        ("4-day solo trip to Lisbon, budget travel", "budget"),
        ("we're on a budget", "budget"),
        ("looking for budget-friendly options", "budget"),
        ("a tight budget", "budget"),
        # Other tiers still win over the noun.
        ("a moderate budget", "moderate"),
        ("luxury budget, no limits", "luxury"),
    ],
)
def test_budget_level_from_raw_text_distinguishes_noun_from_tier(raw, expected):
    assert _normalize_budget_level(raw, allow_bare_noun=False) == expected


@pytest.mark.parametrize(
    "raw",
    [
        # Found by code review: plain substring matching fired "mid" inside
        # "pyramids" and "low" inside "flower", so a message that never
        # mentioned money silently acquired a budget tier — then skipped the
        # question and costed the trip at that invented rate. This is the
        # exact silently-wrong-answer failure the normalization layer exists
        # to prevent, so it needs boundary-anchored matching.
        "A 7-day trip to Egypt to see the pyramids",
        "I love flower markets, 5 days in Amsterdam solo",
        "a slower pace, 4 days in Oslo",
        "humid weather is fine",
        "3 days in Midtown Manhattan",
        # Words that are real tier synonyms for the *model's* answer but
        # mean something else in free text.
        "mid September in Rome",
        "low season travel",
        "a standard room please",
        # "budget" as the noun, already covered elsewhere but kept adjacent.
        "we have around $4000 budget",
    ],
)
def test_raw_text_scan_does_not_invent_a_budget_tier(raw):
    assert _normalize_budget_level(raw, allow_bare_noun=False) is None


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("mid-range budget", "moderate"),
        ("we want midrange", "moderate"),
        ("a moderate budget", "moderate"),
        ("keeping it cheap", "budget"),
        ("low-cost trip", "budget"),
        ("shoestring", "budget"),
        ("budget travel", "budget"),
        ("on a budget", "budget"),
        ("high-end days in Dubai", "luxury"),
        ("we want to splurge", "luxury"),
        ("five star hotels", "luxury"),
    ],
)
def test_raw_text_scan_still_finds_real_tiers(raw, expected):
    """Boundary matching must not cost recall on genuine statements."""
    assert _normalize_budget_level(raw, allow_bare_noun=False) == expected


@pytest.mark.parametrize(
    "model_value, expected",
    [("mid", "moderate"), ("standard", "moderate"), ("economy", "budget"), ("premium", "luxury")],
)
def test_model_value_path_keeps_the_ambiguous_words(model_value, expected):
    """The model's answer is already scoped to this field, so a bare "mid"
    there does mean the tier — only the free-text scan has to be strict."""
    assert _normalize_budget_level(model_value) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        # Rates are not trip totals. `_estimate_budget` returns a stated
        # total verbatim, so reading "$200 a night" as the total caps a
        # seven-day family trip at $200.
        ("4 of us, about $200 a night", None),
        ("we saw flights for $800 each", None),
        ("$150 per person per day", None),
        ("rooms are $180/night", None),
        # Genuine totals, including magnitude suffixes that the old regex
        # truncated ("$4.5k" became 4).
        ("we have around $4000 budget", 4000),
        ("budget is $12,000", 12000),
        ("budget $4.5k", 4500),
        ("$1.2 million", 1200000),
        ("total of $3000 for the trip", 3000),
        ("no figures here", None),
    ],
)
def test_find_dollar_amount(text, expected):
    from travel_assistant.graph import _find_dollar_amount

    assert _find_dollar_amount(text) == expected


@pytest.mark.parametrize(
    "model_value, raw_text, expected",
    [
        # The model answering "2 weeks" must not become a 2-day trip.
        ("2 weeks", "we want two weeks in Italy", 14),
        ("3 nights", "three nights in Vegas", 3),
        ("a week", "a week in Athens", 7),
        # A bare number from the model needs no unit — the field is already
        # scoped to trip length.
        ("7", "some trip", 7),
        ("seven", "some trip", 7),
    ],
)
def test_coerce_trip_length_keeps_the_unit_from_the_models_answer(model_value, raw_text, expected):
    from travel_assistant.graph import _coerce_trip_length

    assert _coerce_trip_length(model_value, raw_text) == expected


def test_bare_budget_from_the_model_still_means_the_tier():
    """The model returning "budget" as this field's *value* does mean the
    cheap tier — only the raw-message scan has to be careful."""
    assert _normalize_budget_level("budget") == "budget"


def test_editing_only_the_stated_total_regenerates_the_budget_section():
    """`budget_total_usd` is not a preference field, so it is missing from
    ALL_PREFERENCE_FIELDS. If the edit diff iterates that tuple, an edit
    changing only the dollar amount yields no regeneration targets and the
    user is shown the identical plan — the assistant looks stuck."""
    llm = DollarParaphraseChatModel()
    graph = build_graph(llm, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "t12b"}}

    graph.invoke(
        {
            "destination": "Seattle",
            "trip_length_days": 3,
            "group_type": "family",
            "interests": "food",
            "last_user_input": "we have around $4000 budget",
        },
        config=config,
    )
    graph.invoke(Command(resume="yes, that all looks correct"), config=config)
    llm.generation_prompts.clear()

    graph.invoke(Command(resume="actually our budget is $9000"), config=config)
    after_state = graph.get_state(config).values

    assert after_state["budget_total_usd"] == 9000
    assert _estimate_budget(after_state) == 9000
    # Assert on the prompts, not the rendered text: the fake model echoes a
    # 40-character prefix that is identical before and after, so comparing
    # `budget_estimate` strings would pass even if nothing regenerated.
    # This stub doesn't branch on the edit-extraction system prompt, so that
    # call lands in the same list — filter it out by its context header.
    regenerated = [
        p for p in llm.generation_prompts if not p.startswith("Currently known preferences:")
    ]
    assert len(regenerated) == 1, f"only the budget section should rerun, got {len(regenerated)}"
    assert "$9000" in regenerated[0]
    assert "budget summary" in regenerated[0]


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("mid-range", "moderate"),
        ("midrange", "moderate"),
        # "budget" is also the noun for the field, so the tier word next to
        # it has to win — otherwise "a moderate budget" costs as cheap.
        ("a comfortable budget", "moderate"),
        ("a moderate budget", "moderate"),
        ("budget", "budget"),
        ("low budget", "budget"),
        ("cheap", "budget"),
        ("shoestring", "budget"),
        ("high-end", "luxury"),
        ("we want to splurge", "luxury"),
        ("zzz", None),
        ("", None),
        (None, None),
    ],
)
def test_budget_level_normalizes_to_the_closed_vocabulary(raw, expected):
    """An unmappable budget must become None (and so be re-asked) rather
    than reaching _DAILY_RATE_BY_BUDGET and silently costing as moderate."""
    assert _normalize_budget_level(raw) == expected


@pytest.mark.parametrize(
    "state, message",
    [
        (
            {"budget_level": "mid-range", "group_type": "solo", "trip_length_days": 3},
            "budget_level",
        ),
        (
            {"budget_level": "moderate", "group_type": "a big family", "trip_length_days": 3},
            "group_type",
        ),
        (
            {"budget_level": "moderate", "group_type": "solo", "trip_length_days": None},
            "trip_length_days",
        ),
    ],
)
def test_estimate_budget_raises_rather_than_silently_defaulting(state, message):
    """Every input here is required and normalized before generation runs,
    so an unrecognized value is a pipeline bug. It must not produce a
    confident, plausible, wrong number."""
    with pytest.raises(ValueError, match=message):
        _estimate_budget(state)


class GroupTypeBlindChatModel:
    """Simulates a small model that extracts every field except group_type,
    the failure mode reported when 'my wife and son' should imply 'family'
    but the model returns null for group_type anyway."""

    def invoke(self, messages, **kwargs):
        system_prompt = messages[0].content
        user_prompt = messages[-1].content

        if "extract structured travel preferences" in system_prompt:
            return FakeMessage(
                content=json.dumps(
                    {
                        "destination": None,
                        "trip_length_days": None,
                        "group_type": None,
                        "interests": "food, cuisine",
                        "budget_level": "$3000",
                        "travel_style": None,
                        "travel_season": None,
                        "must_visit_attractions": None,
                    }
                )
            )

        return FakeMessage(content=f"[fake output for] {user_prompt[:40]}")


def test_group_type_is_inferred_from_relationship_words_when_model_misses_it():
    graph = build_graph(GroupTypeBlindChatModel(), checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "t11"}}

    result = graph.invoke(
        {
            "destination": "Seattle",
            "trip_length_days": 3,
            "last_user_input": "I will go with my wife and son. we are interesting in foods and have a $3000 budget",
        },
        config=config,
    )

    assert "__interrupt__" in result
    payload = result["__interrupt__"][0].value
    assert payload["kind"] == "summary_confirmation"
    assert payload["collected_fields"]["group_type"] == "family"


class HallucinatingChatModel:
    """Simulates a small model that ignores the "don't guess" instruction
    and fills in plausible-sounding defaults for every field regardless of
    what the message actually says — the failure mode this test guards
    against (grounding must catch these, not the extraction prompt alone)."""

    def invoke(self, messages, **kwargs):
        system_prompt = messages[0].content
        user_prompt = messages[-1].content

        if "extract structured travel preferences" in system_prompt:
            match = re.search(r"travel to ([A-Za-z]+)", user_prompt)
            destination = match.group(1) if match else None
            match = re.search(r"(\d+)\s+days?", user_prompt.lower())
            trip_length = int(match.group(1)) if match else None
            return FakeMessage(
                content=json.dumps(
                    {
                        "destination": destination,
                        "trip_length_days": trip_length,
                        "group_type": "individual",
                        "interests": "coffee, foodie, nature",
                        "budget_level": "moderate",
                        "travel_style": "city",
                        "travel_season": "summer",
                        "must_visit_attractions": None,
                    }
                )
            )

        return FakeMessage(content=f"[fake output for] {user_prompt[:40]}")


def test_ungrounded_extracted_fields_are_discarded_not_trusted():
    graph = build_graph(HallucinatingChatModel(), checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "t8"}}
    message = "I would like to travel to seattle for 4 days. help me to prepare the plan"

    result = graph.invoke({"last_user_input": message}, config=config)

    assert "__interrupt__" in result
    payload = result["__interrupt__"][0].value
    # Must still be asking for missing fields, not jumping to confirmation
    # with hallucinated group_type/interests/budget_level/etc.
    assert payload["kind"] == "missing_fields"
    assert set(payload["missing_fields"]) == {"group_type", "interests", "budget_level"}
    collected = payload["collected_fields"]
    assert collected["destination"] == "seattle"
    assert collected["trip_length_days"] == "4"
    assert "group_type" not in collected
    assert "interests" not in collected
    assert "budget_level" not in collected
    assert "travel_style" not in collected
    assert "travel_season" not in collected


def test_spelled_out_trip_length_is_recognized_even_if_model_misses_it():
    """Regression: 'seven-day' previously left trip_length_days null when
    the model's JSON value was null and the digit-only regex fallback found
    nothing — the fallback must also recognize spelled-out numbers."""
    graph = build_graph(FakeChatModel(), checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "t9"}}

    result = graph.invoke(
        {
            "last_user_input": "I want a seven-day family trip to Japan. We love food. Moderate budget."
        },
        config=config,
    )

    assert "__interrupt__" in result
    payload = result["__interrupt__"][0].value
    assert payload["kind"] == "summary_confirmation"
    assert payload["collected_fields"]["trip_length_days"] == "7"


def test_extraction_calls_use_deterministic_decoding(llm, graph):
    """Regression: extraction previously used the model's normal sampling
    temperature, so it could non-deterministically return null for a field
    the message explicitly stated (e.g. destination). Extraction calls must
    request greedy decoding via pipeline_kwargs."""
    calls = []
    original_invoke = llm.invoke

    def spy_invoke(messages, **kwargs):
        calls.append((messages[0].content, kwargs))
        return original_invoke(messages, **kwargs)

    llm.invoke = spy_invoke

    config = {"configurable": {"thread_id": "t10"}}
    graph.invoke({"last_user_input": "Family trip to Japan."}, config=config)

    extraction_calls = [
        kwargs for prompt, kwargs in calls if "structured travel preferences" in prompt
    ]
    assert extraction_calls, "expected at least one extraction call"
    for kwargs in extraction_calls:
        assert kwargs.get("pipeline_kwargs", {}).get("do_sample") is False


def test_add_more_activities_edit_extends_interests_instead_of_replacing(graph):
    config = {"configurable": {"thread_id": "t7"}}
    message = (
        "I want a 7-day family trip to Japan. We love food and theme parks. Our budget is moderate."
    )

    graph.invoke({"last_user_input": message}, config=config)
    result = _confirm(graph, config)
    assert result["__interrupt__"][0].value["kind"] == "plan_review"

    result = graph.invoke(Command(resume="add more outdoor activities"), config=config)

    assert "__interrupt__" in result
    assert result["__interrupt__"][0].value["kind"] == "plan_review"

    result = graph.invoke(Command(resume="no changes, finalize"), config=config)
    assert "__interrupt__" not in result
    assert "food and theme parks" in result["interests"]
    assert "outdoor activities" in result["interests"]
