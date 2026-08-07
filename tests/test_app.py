"""Tests for the Gradio wiring in ``travel_assistant.app``.

These cover the session lifecycle and the rendering of interrupt payloads —
the layer between the graph and the chat window, which had no tests and
where two user-visible bugs were found by actually running the app:
a finalized plan replaying forever, and a stated dollar total vanishing
from the confirmation card.

Imports `app` directly, which is only cheap because the model is built
lazily in `get_graph()`; the fixture below replaces it before it is called,
so nothing here downloads or loads a checkpoint.
"""

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from tests.test_graph import FakeChatModel
from travel_assistant import app as app_module
from travel_assistant.graph import LLMInvocationError, build_graph


@pytest.fixture
def fake_graph(monkeypatch):
    """Point `app.get_graph()` at a graph backed by the fake chat model."""
    graph = build_graph(FakeChatModel(), checkpointer=InMemorySaver())
    monkeypatch.setattr(app_module, "get_graph", lambda: graph)
    return graph


@pytest.fixture
def session():
    return app_module._new_session()


TRIP = "I want a 7-day family trip to Japan. We love food. Our budget is moderate."


# --------------------------------------------------------------------------
# Session lifecycle
# --------------------------------------------------------------------------


def test_finalized_plan_does_not_replay_on_the_next_request(fake_graph, session):
    """Regression: `session["started"]` stayed True after the graph reached
    END, so every later message went to `Command(resume=...)` on a finished
    thread — which returns the finished state. A user who asked for Rome
    after finalizing Japan got the Japan plan back, permanently."""
    app_module.respond(TRIP, [], session)
    app_module.respond("yes, that all looks correct", [], session)
    finished = app_module.respond("no changes, finalize", [], session)
    assert "Japan" in finished

    first_thread = session["thread_id"]
    reply = app_module.respond("Now plan a 4-day solo trip to Rome. I love ruins.", [], session)

    assert session["thread_id"] != first_thread, "a finished plan must not be resumed"
    assert "Japan" not in reply
    assert "Rome" in reply


def test_clearing_the_transcript_mid_conversation_keeps_the_thread(fake_graph, session):
    """Gradio's Clear wipes the transcript but not `gr.State`. A conversation
    still waiting at an interrupt should carry on, not restart — the graph's
    `next` is what decides, so no session bookkeeping is involved."""
    app_module.respond("Family trip to Japan.", [], session)
    thread = session["thread_id"]

    reply = app_module.respond("7 days, we love food, moderate budget", [], session)

    assert session["thread_id"] == thread
    assert "Here's what I have for your trip" in reply


def test_each_browser_session_gets_its_own_thread(fake_graph):
    a, b = app_module._new_session(), app_module._new_session()
    assert a["thread_id"] != b["thread_id"]

    app_module.respond(TRIP, [], a)
    reply_b = app_module.respond("Family trip to Japan.", [], b)

    # b is still collecting; a's completed extraction must not leak into it.
    assert "Could you tell me your" in reply_b


# --------------------------------------------------------------------------
# Error handling
# --------------------------------------------------------------------------


def test_llm_failure_returns_a_message_not_a_traceback(monkeypatch, session):
    class Dead:
        def get_state(self, config):
            raise LLMInvocationError("extract call to the language model failed")

    monkeypatch.setattr(app_module, "get_graph", lambda: Dead())
    reply = app_module.respond("hi", [], session)
    assert reply == app_module._LLM_ERROR_REPLY
    assert "Traceback" not in reply


def test_unexpected_node_failure_returns_a_message_not_a_traceback(monkeypatch, session):
    class Exploding:
        def get_state(self, config):
            raise ValueError("budget_level must be one of ('budget', 'moderate', 'luxury')")

    monkeypatch.setattr(app_module, "get_graph", lambda: Exploding())
    reply = app_module.respond("hi", [], session)
    assert reply == app_module._UNEXPECTED_ERROR_REPLY
    assert "budget_level" not in reply, "internal detail must not reach the chat window"


def test_a_deterministic_node_failure_does_not_trap_the_session(fake_graph, session, monkeypatch):
    """Found by code review. LangGraph leaves the checkpoint pointing at the
    node that raised, so `snapshot.next` stays truthy and every later message
    takes the resume branch and re-runs the same failing node. The session
    was stuck forever while the reply told the user to rephrase — which could
    not possibly help, since the message is discarded before the node runs.
    """
    boom = ValueError("budget_level must be one of ('budget', 'moderate', 'luxury')")
    calls = {"n": 0}
    real_invoke = fake_graph.invoke

    def failing_invoke(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise boom
        return real_invoke(*args, **kwargs)

    monkeypatch.setattr(fake_graph, "invoke", failing_invoke)

    first_thread = session["thread_id"]
    assert app_module.respond(TRIP, [], session) == app_module._UNEXPECTED_ERROR_REPLY
    assert session["thread_id"] != first_thread, "the poisoned thread must be abandoned"

    # The next message must reach the graph normally, not replay the failure.
    reply = app_module.respond(TRIP, [], session)
    assert reply != app_module._UNEXPECTED_ERROR_REPLY
    assert "Here's what I have for your trip" in reply


def test_a_transient_llm_failure_keeps_the_thread_for_retry(fake_graph, session, monkeypatch):
    """The opposite case: an LLM backend blip should *not* discard the
    conversation. Replaying the pending node is the correct retry, and
    throwing away the user's collected preferences would be worse than the
    error itself."""
    calls = {"n": 0}
    real_invoke = fake_graph.invoke

    def flaky_invoke(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise LLMInvocationError("extract call to the language model failed")
        return real_invoke(*args, **kwargs)

    monkeypatch.setattr(fake_graph, "invoke", flaky_invoke)

    thread = session["thread_id"]
    assert app_module.respond(TRIP, [], session) == app_module._LLM_ERROR_REPLY
    assert session["thread_id"] == thread, "a transient failure must not discard the thread"


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def test_stated_total_is_echoed_back_in_the_summary():
    """Regression: `_describe_known` iterated ALL_PREFERENCE_FIELDS, which
    excludes `budget_total_usd`. Someone who said "I have $5000" saw only
    "budget (moderate)" — no confirmation their figure had registered."""
    rendered = app_module._render_interrupt(
        {
            "kind": "summary_confirmation",
            "collected_fields": {
                "destination": "Seattle",
                "trip_length_days": "5",
                "group_type": "family",
                "budget_level": "moderate",
                "budget_total_usd": "5000",
            },
        }
    )
    assert "stated total budget ($5,000)" in rendered
    assert "budget (moderate)" in rendered


@pytest.mark.parametrize(
    "field, value, expected",
    [
        ("trip_length_days", "1", "1 day"),
        ("trip_length_days", "7", "7 days"),
        ("budget_total_usd", "5000", "$5,000"),
        ("budget_total_usd", "12000", "$12,000"),
        ("destination", "Kyoto", "Kyoto"),
    ],
)
def test_format_field_value(field, value, expected):
    assert app_module._format_field_value(field, value) == expected


def test_missing_fields_card_never_asks_for_the_stated_total():
    """`budget_total_usd` is not a preference and is never requested."""
    rendered = app_module._render_interrupt(
        {
            "kind": "missing_fields",
            "collected_fields": {"destination": "Kyoto"},
            "missing_fields": ["trip_length_days", "budget_level", "budget_total_usd"],
        }
    )
    assert "trip length" in rendered and "budget" in rendered
    assert "stated total budget" not in rendered


def test_unknown_interrupt_kind_falls_back_to_a_usable_prompt():
    assert app_module._render_interrupt({"kind": "something_new"})


# --------------------------------------------------------------------------
# The plan has to be visible before it can be reviewed
# --------------------------------------------------------------------------


def test_review_step_shows_the_plan_it_is_asking_about(fake_graph, session):
    """Regression reported from a live run: confirming the preferences
    jumped straight to "Would you like any changes, or shall I finalize
    this plan?" — the five sections had been generated but never rendered,
    so the user was asked to approve something invisible."""
    app_module.respond(TRIP, [], session)
    reply = app_module.respond("go ahead to generate the plan", [], session)

    for heading in (
        "Itinerary",
        "Activities",
        "Getting around",
        "Estimated budget",
        "Packing checklist",
    ):
        assert heading in reply, f"{heading!r} missing from the review message"
    assert "Would you like any changes" in reply


def test_regenerated_plan_is_shown_after_an_edit(fake_graph, session):
    """Same hole on the edit loop: `regenerate_affected` returns to
    `review_plan`, so a requested change produced no visible result."""
    app_module.respond(TRIP, [], session)
    app_module.respond("go ahead to generate the plan", [], session)
    reply = app_module.respond("make it 10 days instead", [], session)

    assert "Itinerary" in reply and "Packing checklist" in reply
    assert "Would you like any changes" in reply


def test_finalizing_leads_with_confirmation_not_a_bare_replay(fake_graph, session):
    app_module.respond(TRIP, [], session)
    app_module.respond("go ahead to generate the plan", [], session)
    reply = app_module.respond("no changes, finalize", [], session)

    assert reply.startswith("All set")
    assert "Itinerary" in reply


def test_go_ahead_to_generate_the_plan_is_read_as_a_confirmation(fake_graph, session):
    """The phrasing that prompted the bug report. It must not be treated as
    an edit request — that path re-extracts, finds nothing to change, and
    re-shows the identical summary, which looks like being ignored."""
    app_module.respond(TRIP, [], session)
    reply = app_module.respond("go ahead to generate the plan", [], session)
    assert "Here's what I have for your trip" not in reply
