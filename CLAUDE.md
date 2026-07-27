# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run all tests (fast — uses a fake chat model, no GPU/model download needed)
pytest tests/

# Run a single test
pytest tests/test_graph.py::test_full_conversation_produces_all_outputs -v

# Run the real chatbot locally (downloads and loads the actual LLM;
# needs an Apple Silicon Mac, a CUDA GPU, or CPU as a slow fallback —
# backend is auto-detected, see Architecture below)
python -m travel_assistant.app
```

There is no separate lint/build step — this is a small pure-Python project with no linter or build config.

`travel_assistant_project.zip` at the repo root is a manually-maintained,
point-in-time snapshot of the project for teammates to upload directly to
Google Colab (see README for why). If you change anything under
`travel_assistant/`, regenerate it:

```bash
zip -rq travel_assistant_project.zip . \
  -x ".git/*" -x ".venv/*" -x "*__pycache__*" -x ".pytest_cache/*" \
  -x ".idea/*" -x ".claude/*" -x "*.DS_Store" -x "travel_assistant_project.zip"
```

## Architecture

This is an AI-Powered Travel Assistant Planner chatbot (a school project —
see `Group 4_Project Proposal.pdf`). The proposal's plan was to call
`transformers` directly; this implementation instead uses **LangGraph** to
model the conversation as an explicit state machine, with **LangChain**
wrapping the LLM.

### The conversation is a fixed pipeline, not an LLM-driven agent

`travel_assistant/graph.py` builds a `StateGraph` with six nodes in a
straight line:

```
START -> collect_preferences -> generate_itinerary -> generate_activities
      -> generate_transportation -> generate_budget -> generate_packing -> END
```

Routing between steps is hard-coded (`build_graph()`'s `add_edge` calls),
never decided by the LLM. This matters because the project's model is a
small ~1B-parameter model, not reliable enough to plan multi-step tool use
on its own — it is only ever asked to generate text for one stage at a
time.

- **`collect_preferences`** is a single node that asks up to five
  questions in order (destination, trip length, group type, interests,
  budget level), defined in `PREFERENCE_QUESTIONS`. It uses LangGraph's
  `interrupt()` / `Command(resume=...)` human-in-the-loop mechanism to
  pause and wait for each answer — multiple sequential `interrupt()`
  calls in one node, not separate graph nodes per question. Already-set
  fields are skipped, since the node re-runs from the top on every
  resume.
- **The five `generate_*` nodes** each make exactly one LLM call
  (`_generate()`) to fill in one field of the shared state. The
  `generate_budget` node is the one exception: the dollar figure comes
  from a deterministic rate table (`_DAILY_RATE_BY_BUDGET` ×
  `_GROUP_MULTIPLIER` × days), and the LLM is only asked to narrate that
  fixed number — so the one field users are likely to sanity-check
  doesn't depend on the model's arithmetic.
- **`travel_assistant/state.py`** defines `TravelState`, the single
  `TypedDict` shared across every node. A field being `None`/absent is
  how a node knows whether it still needs to ask for it or generate it.
  LangGraph merges each node's returned dict into the persisted state
  per `thread_id` — there's no other state-passing mechanism.
- **`travel_assistant/app.py`** wires the graph to a Gradio
  `ChatInterface`. Each browser session gets its own random `thread_id`
  (via `gr.State(value=_new_session)`) so concurrent users don't share
  conversations. The routing logic there is: the first message a user
  sends is discarded/just kicks off `graph.invoke({}, config)`; every
  message after that is passed back in as `Command(resume=message)`. If
  the dict returned by `graph.invoke()` contains a `"__interrupt__"` key,
  its `.value["question"]` is shown as the bot's reply; otherwise the
  graph has run to completion and `format_summary()` renders the final
  trip plan.

### LLM backend auto-detection

`travel_assistant/llm.py`'s `build_chat_model()` picks CUDA → Apple
Silicon MPS → CPU (`_pick_device()`) and loads a different checkpoint
accordingly, since `bitsandbytes` (used for 4-bit quantization) has no
Metal/CPU backend:

- **CUDA** (e.g. Colab's free T4, per the proposal): the pre-quantized
  `unsloth/Llama-3.2-1B-Instruct-bnb-4bit`, loaded via
  `HuggingFacePipeline.from_model_id(..., device_map="auto")` so
  `accelerate`/`bitsandbytes` can dispatch it.
- **MPS or CPU**: the non-quantized `unsloth/Llama-3.2-1B-Instruct` in
  fp16 (MPS) or fp32 (CPU). Because
  `HuggingFacePipeline.from_model_id(device=...)` only accepts a legacy
  int CUDA device index (not device strings like `"mps"`), this path
  builds the `transformers` pipeline manually
  (`AutoModelForCausalLM`/`AutoTokenizer`/`pipeline(device=...)`) and
  wraps it with `HuggingFacePipeline(pipeline=pipe)` instead of
  `from_model_id`.
- `return_full_text: False` in `pipeline_kwargs` is required — without
  it the pipeline echoes the whole formatted prompt back inside
  `response.content`, which would break every `_generate()` call in
  `graph.py`.

This was verified end-to-end (real model, real inference) on an Apple M4
Max via the MPS path — see `docs/design_document.md` for details.

### Testing strategy

`tests/test_graph.py` never loads the real model — it swaps in a
`FakeChatModel` that just echoes a snippet of the prompt back, so it can
verify the graph's *control flow* (right questions in the right order,
no field asked twice, all five output fields eventually populated)
without a GPU or a multi-minute model download. When changing
`graph.py`, run these tests first; only run the real
`python -m travel_assistant.app` demo to verify actual generation
quality.

## Documentation

- `README.md` — setup and running instructions (local + Google Colab),
  kept in sync with the actual `notebooks/travel_assistant_colab.ipynb`
  cells.
- `docs/design_document.md` — architecture/workflow/sequence diagrams
  (Mermaid), state schema, UI design, and HCI heuristics.
- `docs/understanding_the_code.md` — plain-English file-by-file
  walkthrough for non-coding team members; also documents which small
  edits (question wording, UI title text) are safe to make without
  touching graph wiring.
