"""Gradio chat UI wired to the LangGraph travel-planning graph.

Each browser session gets its own LangGraph thread (persisted in memory
via the checkpointer) so multiple people can use the demo at once without
crosstalk. Unlike a fixed questionnaire, the first message a user sends is
real content — a free-form trip description — and is passed straight into
the graph as ``last_user_input`` rather than discarded.
"""

import uuid

import gradio as gr
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from travel_assistant.graph import build_graph, format_summary
from travel_assistant.llm import build_chat_model
from travel_assistant.state import ALL_PREFERENCE_FIELDS, FIELD_LABELS

_checkpointer = InMemorySaver()
_llm = build_chat_model()
_graph = build_graph(_llm, checkpointer=_checkpointer)


def _new_session() -> dict:
    return {"thread_id": str(uuid.uuid4()), "started": False}


def _join_with_and(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def _format_field_value(field: str, value: str) -> str:
    if field == "trip_length_days":
        return f"{value} day" if value == "1" else f"{value} days"
    return value


def _describe_known(collected: dict) -> str:
    parts = [
        f"{FIELD_LABELS.get(f, f).lower()} ({_format_field_value(f, collected[f])})"
        for f in ALL_PREFERENCE_FIELDS
        if f in collected
    ]
    return _join_with_and(parts)


def _describe_missing(missing: list) -> str:
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
        return payload["prompt"]

    return "Could you tell me more about your trip?"


def respond(message: str, history, session: dict) -> str:
    config = {"configurable": {"thread_id": session["thread_id"]}}

    if not session["started"]:
        session["started"] = True
        result = _graph.invoke({"last_user_input": message}, config=config)
    else:
        result = _graph.invoke(Command(resume=message), config=config)

    if "__interrupt__" in result:
        return _render_interrupt(result["__interrupt__"][0].value)

    return format_summary(result)


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
    demo.launch()
