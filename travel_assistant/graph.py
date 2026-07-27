"""LangGraph state machine for the AI-Powered Travel Assistant Planner.

The graph mirrors the project's in-scope steps as a linear pipeline:
collect preferences (human-in-the-loop) -> itinerary -> activities ->
transportation -> budget -> packing list -> summary. Routing between
steps is fixed in code rather than left to the LLM, so even a small
local model only ever has to do text generation, never decide what
happens next.
"""

import re
from typing import Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from travel_assistant.state import TravelState

PREFERENCE_QUESTIONS = (
    ("destination", "Where would you like to travel to?"),
    ("trip_length_days", "How many days will your trip be?"),
    ("group_type", "Who's traveling — solo, a couple, family, friends, or students?"),
    (
        "interests",
        "What are you most interested in on this trip "
        "(e.g. food, history, nature, nightlife, relaxation)?",
    ),
    ("budget_level", "What's your budget level — budget, moderate, or luxury?"),
)


def _parse_trip_length(answer: str) -> int:
    match = re.search(r"\d+", answer)
    return int(match.group()) if match else 3


_TRANSFORMS = {"trip_length_days": _parse_trip_length}


def collect_preferences(state: TravelState) -> dict:
    """Ask for any preference fields not yet filled, one at a time.

    Each missing field triggers its own ``interrupt()`` call. Fields
    already answered in a prior turn are skipped without pausing, since
    LangGraph replays this node from the top on every resume.
    """
    updates: dict = {}

    for field, question in PREFERENCE_QUESTIONS:
        if state.get(field) or updates.get(field):
            continue
        answer = interrupt({"field": field, "question": question})
        transform = _TRANSFORMS.get(field, lambda x: x.strip())
        updates[field] = transform(answer)

    return updates


def _generate(llm: BaseChatModel, system_prompt: str, user_prompt: str) -> str:
    response = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    )
    return response.content.strip()


_ASSISTANT_SYSTEM_PROMPT = (
    "You are a friendly, practical travel planning assistant. "
    "Keep responses concise and well organized."
)


def _make_itinerary_node(llm: BaseChatModel) -> Callable[[TravelState], dict]:
    def generate_itinerary(state: TravelState) -> dict:
        prompt = (
            f"Destination: {state['destination']}\n"
            f"Trip length: {state['trip_length_days']} days\n"
            f"Traveler group: {state['group_type']}\n"
            f"Interests: {state['interests']}\n"
            f"Budget level: {state['budget_level']}\n\n"
            "Write a day-by-day itinerary for this trip, with a short heading per day."
        )
        text = _generate(llm, _ASSISTANT_SYSTEM_PROMPT, prompt)
        return {"itinerary": text}

    return generate_itinerary


def _make_activities_node(llm: BaseChatModel) -> Callable[[TravelState], dict]:
    def generate_activities(state: TravelState) -> dict:
        prompt = (
            f"Destination: {state['destination']}\n"
            f"Interests: {state['interests']}\n"
            f"Traveler group: {state['group_type']}\n\n"
            "Recommend 5-8 specific attractions or activities that match these interests, "
            "as a bulleted list."
        )
        text = _generate(llm, _ASSISTANT_SYSTEM_PROMPT, prompt)
        return {"activities": text}

    return generate_activities


def _make_transportation_node(llm: BaseChatModel) -> Callable[[TravelState], dict]:
    def generate_transportation(state: TravelState) -> dict:
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

    return generate_transportation


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


def _make_budget_node(llm: BaseChatModel) -> Callable[[TravelState], dict]:
    def generate_budget(state: TravelState) -> dict:
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

    return generate_budget


def _make_packing_node(llm: BaseChatModel) -> Callable[[TravelState], dict]:
    def generate_packing(state: TravelState) -> dict:
        prompt = (
            f"Destination: {state['destination']}\n"
            f"Trip length: {state['trip_length_days']} days\n"
            f"Traveler group: {state['group_type']}\n"
            f"Interests: {state['interests']}\n\n"
            "Generate a packing checklist for this trip as a bulleted list."
        )
        text = _generate(llm, _ASSISTANT_SYSTEM_PROMPT, prompt)
        return {"packing_list": text}

    return generate_packing


def build_graph(llm: BaseChatModel, checkpointer: BaseCheckpointSaver):
    """Compile the travel-planning StateGraph with the given chat model."""
    builder = StateGraph(TravelState)

    builder.add_node("collect_preferences", collect_preferences)
    builder.add_node("generate_itinerary", _make_itinerary_node(llm))
    builder.add_node("generate_activities", _make_activities_node(llm))
    builder.add_node("generate_transportation", _make_transportation_node(llm))
    builder.add_node("generate_budget", _make_budget_node(llm))
    builder.add_node("generate_packing", _make_packing_node(llm))

    builder.add_edge(START, "collect_preferences")
    builder.add_edge("collect_preferences", "generate_itinerary")
    builder.add_edge("generate_itinerary", "generate_activities")
    builder.add_edge("generate_activities", "generate_transportation")
    builder.add_edge("generate_transportation", "generate_budget")
    builder.add_edge("generate_budget", "generate_packing")
    builder.add_edge("generate_packing", END)

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
