"""Gradio chat UI wired to the LangGraph travel-planning graph.

Each browser session gets its own LangGraph thread (persisted in memory
via the checkpointer) so multiple people can use the demo at once without
crosstalk. Unlike a fixed questionnaire, the first message a user sends is
real content — a free-form trip description — and is passed straight into
the graph as ``last_user_input`` rather than discarded.
"""

import logging
import uuid
from functools import lru_cache

import gradio as gr
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from travel_assistant.graph import LLMInvocationError, build_graph, format_summary
from travel_assistant.llm import build_chat_model
from travel_assistant.state import ALL_PREFERENCE_FIELDS, DISPLAYED_FIELDS, FIELD_LABELS

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_graph():
    """Build the graph on first use, not at import.

    Loading the model at module scope meant importing this module — for a
    test, a lint pass, a CI job, or just to read ``demo`` — downloaded and
    resident-loaded a 1B-parameter model. Deferring it here keeps import
    free and cheap; the first chat message pays the load cost once, and
    ``lru_cache`` shares that single instance across every session.
    """
    logger.info("building chat model and graph (first request)")
    return build_graph(build_chat_model(), checkpointer=InMemorySaver())


def _new_session() -> dict:
    return {"thread_id": str(uuid.uuid4())}


def _join_with_and(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def _format_field_value(field: str, value: str) -> str:
    if field == "trip_length_days":
        return f"{value} day" if value == "1" else f"{value} days"
    if field == "budget_total_usd":
        # Read back as currency. A bare "5000" next to "budget (moderate)"
        # reads as a stray number rather than the figure the user named.
        return f"${int(value):,}"
    return value


def _describe_known(collected: dict) -> str:
    # DISPLAYED_FIELDS, not ALL_PREFERENCE_FIELDS: `budget_total_usd` is not
    # a preference (it is never asked for), but it is user-supplied and has
    # to be echoed back. Iterating the preference tuple silently dropped it,
    # so someone who said "I have $5000" saw only "budget (moderate)" and no
    # confirmation their number had registered at all.
    parts = [
        f"{FIELD_LABELS.get(f, f).lower()} ({_format_field_value(f, collected[f])})"
        for f in DISPLAYED_FIELDS
        if f in collected
    ]
    return _join_with_and(parts)


def _describe_missing(missing: list) -> str:
    # Only ever preference fields — budget_total_usd is never requested.
    labels = [FIELD_LABELS.get(f, f).lower() for f in ALL_PREFERENCE_FIELDS if f in missing]
    return _join_with_and(labels)


def _render_interrupt(payload: dict) -> str:
    kind = payload.get("kind")

    if kind == "initial_prompt":
        return payload["prompt"]

    if kind == "missing_fields":
        collected = payload.get("collected_fields", {})
        missing = payload.get("missing_fields", [])
        known_part = f"So far I have your {_describe_known(collected)}. " if collected else ""
        return (
            f"Great! {known_part}Could you tell me your {_describe_missing(missing)}? "
            "Feel free to share one or all of those in your next message."
        )

    if kind == "single_field_fallback":
        return f"Just to confirm — what's your {payload['label'].lower()}?"

    if kind == "summary_confirmation":
        collected = payload.get("collected_fields", {})
        return (
            f"Here's what I have for your trip: {_describe_known(collected)}. "
            "Does that all look right, or would you like to change anything?"
        )

    if kind == "plan_review":
        # Show the plan, then ask about it. Returning `prompt` alone asked
        # the user to review something they had never been shown.
        plan = payload.get("plan")
        if plan:
            return f"{format_summary(plan)}\n---\n\n{payload['prompt']}"
        return payload["prompt"]

    return "Could you tell me more about your trip?"


_LLM_ERROR_REPLY = (
    "Sorry — I had trouble reaching the language model just now. Please send that message again."
)
_UNEXPECTED_ERROR_REPLY = (
    "Sorry — something went wrong on my side while building your plan, so "
    "I've cleared it. Tell me about your trip again and I'll start fresh."
)


def respond(message: str, history, session: dict) -> str:
    graph = get_graph()
    config = {"configurable": {"thread_id": session["thread_id"]}}

    try:
        # The graph's own checkpoint decides whether this message resumes an
        # in-flight conversation or starts a new one. A `started` flag on the
        # session dict could not: it stayed True once the plan was finalized,
        # so every later message was fed to `Command(resume=...)` on a thread
        # sitting at END — which re-returns the finished state. Asking for a
        # trip to Rome after finalizing a trip to Japan replayed the Japan
        # plan, permanently. Gradio's Clear button has the same shape: it
        # wipes the transcript but not `gr.State`.
        snapshot = graph.get_state(config)
        if snapshot.next:
            result = graph.invoke(Command(resume=message), config=config)
        else:
            if snapshot.created_at:
                # Thread ran to completion — this message begins a new plan.
                session["thread_id"] = str(uuid.uuid4())
                config = {"configurable": {"thread_id": session["thread_id"]}}
                logger.info("previous plan finalized; starting thread %s", session["thread_id"])
            result = graph.invoke({"last_user_input": message}, config=config)
    except LLMInvocationError:
        # Already logged with a traceback at the call site in graph.py.
        # The checkpoint still points at the failed node, so the next
        # message replays it — which is the correct retry for a transient
        # backend failure (OOM, a dropped pipeline).
        return _LLM_ERROR_REPLY
    except Exception:
        # A node raised deterministically — e.g. _estimate_budget rejecting
        # an unnormalized value. Replaying it would fail identically, and
        # because LangGraph leaves the checkpoint pointing at the failed
        # node, *every* later message would take the resume branch and hit
        # the same exception. The session would be permanently stuck while
        # the reply told the user to rephrase, which could not possibly
        # help. Abandon the thread so the next message starts clean.
        logger.exception(
            "graph.invoke failed on thread_id=%s; abandoning thread", session["thread_id"]
        )
        session["thread_id"] = str(uuid.uuid4())
        return _UNEXPECTED_ERROR_REPLY

    if "__interrupt__" in result:
        return _render_interrupt(result["__interrupt__"][0].value)

    # Graph ran to completion — the user confirmed the plan they were just
    # shown, so lead with the confirmation rather than re-printing the same
    # five sections as if they were new.
    return (
        "All set — here's your finalized plan. Safe travels!\n\n"
        f"{format_summary(result)}\n---\n\n"
        "Tell me about another trip any time and I'll start a fresh plan."
    )


_GREETING = (
    "Hi! I'm your AI travel planning assistant. Tell me about the trip you'd "
    "like to take — destination, how many days, who's going, what you're "
    "into, and your budget — and I'll take care of the rest: a day-by-day "
    "itinerary, activity picks, transportation tips, a budget estimate, and "
    "a packing checklist. Just describe it in your own words, and I'll only "
    "ask about whatever you leave out."
)

demo = gr.ChatInterface(
    fn=respond,
    chatbot=gr.Chatbot(value=[{"role": "assistant", "content": _GREETING}], height="100%"),
    additional_inputs=[gr.State(value=_new_session)],
    title="AI-Powered Travel Assistant Planner",
    description="Tell me about the trip you would like to plan.",
    fill_height=True,
)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
    )
    demo.launch()
