# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

AI-Powered Travel Assistant Planner (MSAI-631-B01, Group 4) — a conversational
chatbot that collects trip preferences via natural-language extraction (not
a fixed questionnaire) and generates an itinerary, activity picks,
transportation guidance, a budget estimate, and a packing checklist, with a
pre-generation confirmation step and a post-generation edit/regenerate loop.

## Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run tests (logic-only, no GPU/model download required)
pytest tests/
pytest tests/test_graph.py::test_post_generation_edit_regenerates_only_the_affected_section  # single test

# Run the full demo locally (auto-detects CUDA / Apple Silicon MPS / CPU)
python -m travel_assistant.app
```

There is no lint/format tooling configured in this repo (no linter config,
no pre-commit). Tests are the only automated check — `pytest tests/` uses a
fake chat model (it branches on the system prompt: extraction calls get a
small regex-based fake extractor, generation calls get an echoed prompt
snippet), so it never downloads the real model or needs a GPU.

To try the full demo without local setup, use Google Colab (free T4 GPU) —
see the README's "Running in Google Colab" section for the upload/clone
steps. `travel_assistant_project.zip` is a point-in-time snapshot for that
workflow; regenerate it (command in the README) if teammates need it after
changes under `travel_assistant/`.

## Architecture

The conversation is an explicit LangGraph `StateGraph`
(`travel_assistant/graph.py`), not an LLM-driven agent loop. Preference
collection is adaptive rather than a fixed sequence of questions:

```
START -> collect_initial_request -> extract_preferences -> validate_preferences
      -[missing fields]-> request_missing_fields -> extract_preferences (loop)
      -[all collected]-> show_summary -> confirm_preferences
      -[edit]-> apply_preference_edit -> validate_preferences (loop)
      -[confirmed]-> generate_itinerary -> generate_activities
      -> generate_transportation -> generate_budget -> generate_packing -> review_plan
      -[edit]-> apply_plan_edit -> regenerate_affected -> review_plan (loop)
      -[confirmed]-> END
```

This is a deliberate design choice, not an incidental one: all routing
decisions (is a field missing? did the user confirm or ask for a change?)
are made by plain Python reading state — via fixed edges or
`add_conditional_edges` — so the small local model (~1B params) only ever
has to extract structured fields from one message or generate text for one
section at a time. It never decides what happens next in the graph. Keep
this separation when extending the graph: add control flow in Python, and
reserve the LLM for extraction/generation inside a node.

- **`travel_assistant/state.py`** — `TravelState` TypedDict, the single
  state shared across every node. The five required preference fields
  (`destination`, `trip_length_days`, `group_type`, `interests`,
  `budget_level`) and three optional ones (`travel_style`,
  `travel_season`, `must_visit_attractions`) are filled in by extraction,
  not by asking one question per field. `collected_fields`/`missing_fields`
  are **derived, display-only** — recomputed from scratch by
  `validate_preferences` on every pass, never mutated incrementally, so
  they can't drift from the underlying typed fields. `conversation_stage`
  and `confirmation_status` track where the conversation is;
  `extraction_attempts` bounds the extract/re-ask loop;
  `regeneration_targets` is a transient field populated by
  `apply_plan_edit` and consumed by `regenerate_affected`.
- **`travel_assistant/graph.py`** — the graph and all nodes.
  - `extract_preferences` makes one LLM call per pass, asking for a JSON
    object of all eight preference fields; a field already known in state
    is never overwritten (`_merge_extracted(..., overwrite=False)`) —
    a later reply that only mentions budget can't null out an
    already-known destination.
  - `validate_preferences` (pure Python, no LLM call) is what
    `route_after_validation` reads to decide between
    `request_missing_fields` (loops back to `extract_preferences`) and
    `show_summary`.
  - `show_summary` → `confirm_preferences` shows everything collected and
    waits for a confirmation or an edit request
    (`route_after_confirmation`); edits go through
    `apply_preference_edit` (`overwrite=True`) and re-validate.
  - Each `generate_*` node's core logic lives in a plain `_*_update(llm,
    state) -> dict` function, separate from the `_make_*_node` closure —
    this split exists so `regenerate_affected` can call the same logic
    directly for just the affected section(s), without rerunning the whole
    five-node chain.
  - The budget estimate is intentionally hybrid: the dollar figure comes
    from a deterministic rate table (`_DAILY_RATE_BY_BUDGET` ×
    `_GROUP_MULTIPLIER` × days) in plain Python; the LLM only narrates that
    number. This keeps the one output users are most likely to sanity-check
    independent of the small model's arithmetic.
  - After generation, `review_plan` lets the user request further changes;
    `apply_plan_edit` re-extracts with a *merge-aware* edit prompt
    (`_EDIT_EXTRACTION_SYSTEM_PROMPT`, given current values as context, so
    "add outdoor activities" extends `interests` rather than replacing it),
    diffs which fields actually changed, and maps them to affected
    `generate_*` node(s) via `_FIELD_TO_REGEN_NODES`. `regenerate_affected`
    runs only those.
  - `_is_confirmation()` matches a fixed set of confirmation
    words/phrases (`_CONFIRM_WORDS`/`_CONFIRM_PHRASES`, short tokens via
    regex word boundaries to avoid false positives like "ok" inside
    "looking") plus a regex pattern (`_CONFIRM_PATTERN`) for affirmative
    sentence shapes not literally in those lists (e.g. "all look right,"
    "everything sounds good") — without it, an unrecognized confirmation
    gets treated as an edit request that finds no field to change, making
    the assistant appear to loop on the same summary.
  - During an edit, `_infer_group_type` is also consulted even when the
    model's edit-extraction call returns a `group_type` — a keyword match
    in the edit text (e.g. "byself" → `solo`) overrides the model's value
    if they disagree, since the model sometimes just echoes the old value
    back unchanged instead of applying the edit.
- **`travel_assistant/llm.py`** — wraps the local HF model as a LangChain
  chat model (`HuggingFacePipeline` + `ChatHuggingFace`) so the rest of the
  app talks to a standard `.invoke(messages) -> AIMessage` interface,
  swappable independently of graph/state logic. Auto-detects backend and
  picks the model accordingly: CUDA uses the proposal's pre-quantized
  `unsloth/Llama-3.2-1B-Instruct-bnb-4bit` (needs `bitsandbytes`+`accelerate`
  device_map dispatch); Apple Silicon MPS or CPU uses the non-quantized
  `unsloth/Llama-3.2-1B-Instruct` in fp16 (`bitsandbytes` has no Metal/CPU
  backend), loaded via a pipeline built directly rather than
  `from_model_id(device=...)` (which only accepts legacy CUDA indices, not
  device strings like `"mps"`).
- **`travel_assistant/app.py`** — Gradio `ChatInterface` wiring. Each
  browser session gets its own LangGraph thread (`uuid4` `thread_id`,
  persisted via `InMemorySaver`) so concurrent users don't cross-talk.
  Unlike the old fixed flow, the **first message is real content** — it's
  passed in as `last_user_input` (`graph.invoke({"last_user_input": message})`),
  not discarded. `respond()` checks for the `"__interrupt__"` key in the
  result dict; if present, `_render_interrupt()` dispatches on the
  payload's `"kind"` (`initial_prompt`, `missing_fields`,
  `single_field_fallback`, `summary_confirmation`, `plan_review`) to
  render the right card. If absent, the graph ran to completion and
  `format_summary()` renders the five generated sections into one message.

## Design docs

- `docs/design_document.md` — the single design doc, matching the
  **current code**: architecture diagram, full workflow/state diagrams
  (all 15 nodes, all conditional edges), state schema, interaction
  sequences, UI mockups, and HCI rationale for the adaptive extraction
  flow as actually implemented (§1–§8), plus a "Design History" appendix
  (§9) condensing the rationale from two earlier documents that have
  since been merged into it and deleted — the original adaptive-extraction
  redesign proposal, and the team's official docx design document. §9
  also records which docx-proposed capabilities were **not** implemented
  (`user_name`/personalization, a `recommend_destination` branch,
  `destination_recommendations` state) — don't assume any capability
  described only in §9 exists in the current graph without checking
  `graph.py`/`state.py`.
- `docs/understanding_the_code.md` — plain-English, no-programming-
  background walkthrough of what each file does, updated for the adaptive
  flow.

## Scope

Matches the project proposal: destination selection, preference collection
(via natural-language extraction, not a fixed Q&A), itinerary generation,
activity recommendations, transportation suggestions, a budget estimate,
and a packing checklist — all conversational, with a pre-generation
confirmation step and a post-generation review/edit loop that regenerates
only the affected section(s). Explicitly out of scope: flight/hotel
booking, payment processing, visa guidance, live/real-time pricing, and
persisting user data/history across sessions.
