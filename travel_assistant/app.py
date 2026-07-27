"""Gradio chat UI wired to the LangGraph travel-planning graph.

Each browser session gets its own LangGraph thread (persisted in memory
via the checkpointer) so multiple people can use the demo at once without
crosstalk. The first message a user sends simply kicks off the graph
(its content is not read as an answer); every message after that answers
whatever question the graph most recently asked.
"""

import uuid

import gradio as gr
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from travel_assistant.graph import build_graph, format_summary
from travel_assistant.llm import build_chat_model

_checkpointer = InMemorySaver()
_llm = build_chat_model()
_graph = build_graph(_llm, checkpointer=_checkpointer)


def _new_session() -> dict:
    return {"thread_id": str(uuid.uuid4()), "started": False}


def respond(message: str, history, session: dict) -> str:
    config = {"configurable": {"thread_id": session["thread_id"]}}

    if not session["started"]:
        session["started"] = True
        result = _graph.invoke({}, config=config)
    else:
        result = _graph.invoke(Command(resume=message), config=config)

    if "__interrupt__" in result:
        return result["__interrupt__"][0].value["question"]

    return format_summary(result)


demo = gr.ChatInterface(
    fn=respond,
    additional_inputs=[gr.State(value=_new_session)],
    title="AI-Powered Travel Assistant Planner",
    description=(
        "Say hi to get started — I'll ask a few quick questions about your trip, "
        "then put together an itinerary, activity picks, transportation tips, a "
        "budget estimate, and a packing checklist."
    ),
)

if __name__ == "__main__":
    demo.launch()
