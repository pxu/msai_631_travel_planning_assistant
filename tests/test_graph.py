"""Tests for the travel-planning LangGraph pipeline.

Uses a fake chat model so these run without a GPU or downloading the
real local model — they only verify the graph's control flow (which
questions get asked, in what order, and that generation nodes eventually
produce every expected field).
"""

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from travel_assistant.graph import build_graph, format_summary


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChatModel:
    """Echoes back a snippet of the prompt instead of calling a real model."""

    def invoke(self, messages):
        user_prompt = messages[-1].content
        return FakeMessage(content=f"[fake output for] {user_prompt[:40]}")


@pytest.fixture
def graph():
    return build_graph(FakeChatModel(), checkpointer=InMemorySaver())


def test_first_invoke_asks_for_destination(graph):
    config = {"configurable": {"thread_id": "t1"}}
    result = graph.invoke({}, config=config)

    assert "__interrupt__" in result
    assert result["__interrupt__"][0].value["field"] == "destination"


def test_full_conversation_produces_all_outputs(graph):
    config = {"configurable": {"thread_id": "t2"}}
    answers = ["Tokyo", "5 days", "couple", "food and nightlife", "moderate"]

    result = graph.invoke({}, config=config)
    for answer in answers:
        assert "__interrupt__" in result
        result = graph.invoke(Command(resume=answer), config=config)

    assert "__interrupt__" not in result
    assert result["destination"] == "Tokyo"
    assert result["trip_length_days"] == 5
    assert result["group_type"] == "couple"
    assert result["budget_level"] == "moderate"
    for field in (
        "itinerary",
        "activities",
        "transportation",
        "budget_estimate",
        "packing_list",
    ):
        assert result[field]

    summary = format_summary(result)
    assert "Tokyo" in summary


def test_fields_already_set_are_not_asked_again():
    graph = build_graph(FakeChatModel(), checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "t3"}}

    result = graph.invoke(
        {"destination": "Rome", "trip_length_days": 4}, config=config
    )

    assert result["__interrupt__"][0].value["field"] == "group_type"
