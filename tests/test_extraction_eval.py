"""Tests for the extraction eval harness.

The evals themselves need the real model and a lot of time, so they are not
run here. What *is* tested is the scoring: a harness that quietly counts a
hallucination as a hit would make the whole exercise worse than useless,
because it would report a confident number that means nothing.

Runs against a fake model, like the rest of ``tests/`` — no GPU, no download.
"""

import json

import pytest

from evals.extraction_eval import (
    HALLUCINATION,
    HIT,
    MISS,
    TRUE_NULL,
    WRONG,
    classify,
    format_report,
    load_cases,
    matches,
    run_case,
    summarize,
)


class StubChatModel:
    """Returns a fixed extraction payload regardless of the prompt."""

    def __init__(self, payload: dict):
        self._payload = payload

    def invoke(self, messages, **kwargs):
        class _Msg:
            content = json.dumps(self._payload)

        return _Msg()


# --------------------------------------------------------------------------
# Field matching
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field_name, expected, actual, should_match",
    [
        # Closed vocabularies and integers are exact — "close" is meaningless
        # for a value that indexes into a rate table.
        ("group_type", "family", "family", True),
        ("group_type", "family", "families", False),
        ("trip_length_days", 7, 7, True),
        ("trip_length_days", 7, "7", False),
        ("budget_level", "moderate", "mid-range", False),
        ("budget_total_usd", 4000, 4000, True),
        # Destination matches in either direction: extra or missing region.
        ("destination", "Kyoto", "Kyoto, Japan", True),
        ("destination", "Kyoto, Japan", "Kyoto", True),
        ("destination", "Kyoto", "Osaka", False),
        # Free text is scored on content-word coverage, not phrasing.
        ("interests", "food, temples", "food and temples", True),
        ("interests", "food, temples", "temples", True),
        ("interests", "food, temples", "nightlife", False),
    ],
)
def test_matches(field_name, expected, actual, should_match):
    assert matches(field_name, expected, actual) is should_match


# --------------------------------------------------------------------------
# Outcome classification — the distinction the whole harness exists for
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expected, actual, outcome",
    [
        ("family", "family", HIT),
        ("family", None, MISS),
        ("family", "solo", WRONG),
        # Expecting nothing and getting nothing is a success, not a no-op:
        # declining to invent a value is what triggers the follow-up question.
        (None, None, TRUE_NULL),
        # ...and inventing one is the failure mode that matters most.
        (None, "solo", HALLUCINATION),
    ],
)
def test_classify(expected, actual, outcome):
    assert classify("group_type", expected, actual) == outcome


def test_hallucination_is_never_counted_as_a_hit():
    """Guards the harness against the one bug that would invalidate it."""
    assert classify("group_type", None, "solo") != HIT
    assert classify("budget_level", None, "moderate") != TRUE_NULL


# --------------------------------------------------------------------------
# End-to-end over a stub model
# --------------------------------------------------------------------------


def test_run_case_scores_a_perfect_extraction():
    case = {
        "id": "x",
        "utterance": "I'm planning a 7-day family trip to Japan with a moderate budget.",
        "tags": [],
        "expected": {
            "destination": "Japan",
            "trip_length_days": 7,
            "group_type": "family",
            "budget_level": "moderate",
        },
    }
    llm = StubChatModel(
        {
            "destination": "Japan",
            "trip_length_days": 7,
            "group_type": "family",
            "interests": None,
            "budget_level": "moderate",
            "travel_style": None,
            "travel_season": None,
            "must_visit_attractions": None,
        }
    )
    result = run_case(llm, case)
    assert result.perfect
    assert set(result.outcomes.values()) == {HIT}


def test_run_case_flags_an_invented_group_type():
    """The exact failure `_is_grounded` exists to catch: the sentence never
    says who is travelling, but the model fills in 'solo' anyway."""
    case = {
        "id": "y",
        "utterance": "Five days in Rome.",
        "tags": [],
        "expected": {"destination": "Rome", "trip_length_days": 5, "group_type": None},
    }
    llm = StubChatModel(
        {
            "destination": "Rome",
            "trip_length_days": 5,
            "group_type": "solo",
            "interests": None,
            "budget_level": None,
            "travel_style": None,
            "travel_season": None,
            "must_visit_attractions": None,
        }
    )
    result = run_case(llm, case)
    # Grounding should reject "solo" — it appears nowhere in the utterance.
    assert result.outcomes["group_type"] == TRUE_NULL
    assert result.perfect


def test_run_case_records_an_exception_instead_of_propagating():
    """One dead case must not abort a 30-case run, and the recorded error
    has to name the underlying cause — `LLMInvocationError` alone says
    nothing about why the model died."""

    class ExplodingModel:
        def invoke(self, messages, **kwargs):
            raise RuntimeError("model went down")

    case = {"id": "z", "utterance": "hi", "tags": [], "expected": {"destination": None}}
    result = run_case(ExplodingModel(), case)
    assert result.error is not None
    assert "LLMInvocationError" in result.error
    assert "model went down" in result.error
    assert not result.perfect


def test_unscored_fields_are_skipped_not_assumed_null():
    """A case that says nothing about `interests` must not be scored on it —
    absent is not the same expectation as expected-null."""
    case = {"id": "w", "utterance": "Rome", "tags": [], "expected": {"destination": "Rome"}}
    llm = StubChatModel(
        {
            "destination": "Rome",
            "trip_length_days": None,
            "group_type": None,
            "interests": "food",
            "budget_level": None,
            "travel_style": None,
            "travel_season": None,
            "must_visit_attractions": None,
        }
    )
    result = run_case(llm, case)
    assert "interests" not in result.outcomes


def test_summarize_and_report_render():
    case = {
        "id": "r",
        "utterance": "Rome for 5 days",
        "tags": [],
        "expected": {"destination": "Rome"},
    }
    llm = StubChatModel(
        {
            f: None
            for f in (
                "destination",
                "trip_length_days",
                "group_type",
                "interests",
                "budget_level",
                "travel_style",
                "travel_season",
                "must_visit_attractions",
            )
        }
    )
    results = [run_case(llm, case)]
    summary = summarize(results)
    assert summary["cases"] == 1
    report = format_report(results, summary)
    assert "field accuracy" in report and "hallucination rate" in report


# --------------------------------------------------------------------------
# The case file itself
# --------------------------------------------------------------------------


def test_case_file_is_well_formed():
    cases = load_cases()
    assert len(cases) >= 25, "the eval set should be big enough to be informative"

    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids)), "duplicate case ids"

    valid_fields = {
        "destination",
        "trip_length_days",
        "group_type",
        "interests",
        "budget_level",
        "travel_style",
        "travel_season",
        "must_visit_attractions",
        "budget_total_usd",
    }
    for case in cases:
        assert case["utterance"].strip(), f"{case['id']}: empty utterance"
        assert case["expected"], f"{case['id']}: no expectations"
        unknown = set(case["expected"]) - valid_fields
        assert not unknown, f"{case['id']}: unknown field(s) {unknown}"
        for name in ("group_type", "budget_level"):
            value = case["expected"].get(name)
            if value is not None:
                assert (
                    value
                    in {
                        "group_type": {"solo", "couple", "family", "friends", "students"},
                        "budget_level": {"budget", "moderate", "luxury"},
                    }[name]
                ), f"{case['id']}: {name}={value!r} is outside the closed vocabulary"


def test_case_file_covers_the_null_expectation():
    """If no case expects a null, the hallucination metric is vacuous."""
    cases = load_cases()
    null_expectations = sum(1 for c in cases for v in c["expected"].values() if v is None)
    assert null_expectations >= 10, "not enough expected-null fields to measure hallucination"


def test_load_cases_filters_by_tag():
    assert all("grounding" in c["tags"] for c in load_cases(tag="grounding"))
    assert len(load_cases(limit=3)) == 3
