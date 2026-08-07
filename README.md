# AI-Powered Travel Assistant Planner

Group 4 — MSAI-631-B01. A conversational assistant that collects trip
preferences and generates a personalized itinerary, activity picks,
transportation guidance, a budget estimate, and a packing checklist.

See [`docs/design_document.md`](docs/design_document.md) for the detailed
design doc (architecture diagram, workflow/state diagrams, interaction
sequence, UI design, and HCI heuristics). Not a coder? Start with
[`docs/understanding_the_code.md`](docs/understanding_the_code.md) instead —
a plain-English walkthrough of what each file does, no programming
background required. For a plain-English tour of the design decisions and
the reasoning behind each one — why routing stays in Python, why the budget
is computed rather than generated, why a null extraction counts as a
success — see [`docs/interview_prep.md`](docs/interview_prep.md).

## Architecture

Built with **LangGraph** instead of calling `transformers` directly. The
conversation is modeled as an explicit `StateGraph` (`travel_assistant/graph.py`)
with adaptive, extraction-based preference collection rather than a fixed
sequential questionnaire:

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

- The user describes their trip in free text ("7-day family trip to Japan,
  we love food and theme parks, moderate budget"); `extract_preferences`
  makes one LLM call to pull structured fields out of that message, and
  `validate_preferences` (plain Python, no LLM call) checks which of the
  five required fields are still missing.
- `request_missing_fields` uses LangGraph's `interrupt()` / `Command(resume=...)`
  human-in-the-loop mechanism to ask only for what's missing, and accepts
  any number of fields in a single reply — a user can answer everything at
  once or one detail at a time. No LLM-driven routing is involved anywhere
  in the graph: conditional edges read plain state (e.g. "is `missing_fields`
  empty?"), so a small local model only ever has to extract or generate
  text, never decide what happens next.
- Before generating anything, `show_summary`/`confirm_preferences` shows a
  ✅/❓ card of everything collected and lets the user confirm or correct it.
- Each `generate_*` node calls the LLM once to produce its section.
  The budget estimate combines a deterministic rate-table calculation with
  an LLM-written summary, so the number itself doesn't depend on the model
  getting arithmetic right.
- After the plan is generated, `review_plan` lets the user request changes
  (e.g. "increase my budget," "add more outdoor activities"); `apply_plan_edit`
  figures out which section(s) are affected and `regenerate_affected`
  reruns only those, leaving the rest of the plan untouched.
- State and routing are separate from the model, so the same graph works
  with any LangChain chat model dropped into `travel_assistant/llm.py`.

The model itself is still the one from the original proposal —
`unsloth/Llama-3.2-1B-Instruct-bnb-4bit`, a pre-quantized 4-bit model that
needs no API key and runs on Colab's free T4 GPU — just wrapped with
LangChain's `HuggingFacePipeline` + `ChatHuggingFace` (`travel_assistant/llm.py`)
instead of calling `transformers` directly.

`travel_assistant/llm.py` auto-detects the backend and picks a model
accordingly:

- **CUDA** (Colab's free T4): the proposal's pre-quantized
  `unsloth/Llama-3.2-1B-Instruct-bnb-4bit`, loaded via `bitsandbytes` + `accelerate`.
- **Apple Silicon (MPS) or CPU**: `bitsandbytes` has no Metal/CPU backend, so
  this loads the same model family's non-quantized weights
  (`unsloth/Llama-3.2-1B-Instruct`) in fp16 directly on the GPU via PyTorch's
  MPS backend. Verified working end-to-end on an Apple M4 Max (64 GB) — the
  full graph (preference extraction/confirmation + itinerary/activities/
  transportation/budget/packing generation) runs locally with no cloud
  dependency at all.

## Project layout

```
travel_assistant_project.zip   # ready-to-upload snapshot of this project, for Colab
travel_assistant/
  state.py    # TravelState TypedDict — the graph's shared state schema
  llm.py      # loads the local model as a LangChain chat model
  graph.py    # the LangGraph StateGraph and its nodes
  app.py      # Gradio ChatInterface wiring
evals/
  extraction_cases.json   # ~30 labelled utterances with expected fields
  extraction_eval.py      # scores the real model's extraction quality
tests/
  test_graph.py            # graph control-flow tests using a fake chat model (no GPU)
  test_app.py              # session lifecycle + interrupt rendering
  test_llm.py              # model-selection resolution (no weights loaded)
  test_extraction_eval.py  # tests the eval scorer itself (also fake-model only)
notebooks/
  travel_assistant_colab.ipynb   # thin notebook to run the Gradio demo on Colab
```

## Measuring extraction quality

`pytest` uses a fake chat model, so it verifies graph *logic* and says
nothing about whether the real ~1B model can actually read "my wife and
son" as `group_type="family"`. The eval harness supplies that number:

```bash
python -m evals.extraction_eval              # all cases, real local model
python -m evals.extraction_eval --limit 5    # quick smoke run
python -m evals.extraction_eval --tag grounding --verbose
```

It reports, per field, four outcomes:

| outcome | meaning |
|---|---|
| **hit** | expected a value, got a matching one |
| **miss** | expected a value, got nothing |
| **wrong** | expected a value, got a different one |
| **hallucination** | expected *nothing*, got a value |

The last one is the one to watch. A model that invents `group_type="solo"`
because the sentence never said who is travelling is far more damaging than
one returning null: null triggers a follow-up question, whereas an invented
value flows silently into the itinerary, the transport advice, and the
budget multiplier. `--min-accuracy 0.9` exits non-zero for CI gating.

### Trying a different model

The checkpoint is selectable without a code change, so "would a bigger
model help?" is a measurement rather than an argument:

```bash
TRAVEL_ASSISTANT_MODEL=unsloth/Llama-3.2-3B-Instruct python -m evals.extraction_eval
TRAVEL_ASSISTANT_MODEL=unsloth/Llama-3.2-3B-Instruct python -m travel_assistant.app
```

Resolution order is the `model_id` argument → `TRAVEL_ASSISTANT_MODEL` →
the per-device default. The 1B default is what the proposal specifies and
what the architecture is built around, so a larger model is an upgrade
rather than a prerequisite — routing stays in Python, the budget figure is
computed deterministically, and extraction is backstopped in code either
way. **Keep the backstops regardless of model size**: they are cheap and
deterministic, and a bigger model reduces how often they fire without
making them unnecessary. The `WARNING` emitted when the budget backstop
fires is how you measure that.

### Current baseline

33 cases, `unsloth/Llama-3.2-1B-Instruct` in fp16 on Apple M4 Max (MPS):

| metric | value |
|---|---|
| field accuracy (hit / expected-a-value) | **95.2 %** (99/104) |
| hallucination rate (invented / should-be-null) | **0.0 %** (0/44) |
| fully correct cases | 28/33 |
| mean extraction latency | 0.90 s |
| wrong values (extracted, but not the right one) | 0 |

Two things worth reading off that table. **Zero "wrong"** — when this
pipeline produces a value it is right; every failure is a *miss*, i.e. it
declined rather than guessed. That is `_is_grounded` plus the normalizers
doing their job, and it is the behaviour you want, because a miss becomes a
follow-up question while a wrong value becomes a silently bad plan.

The eval also paid for itself immediately: the first run scored 93.9 % with
one hallucination — "Group of 4 going to Tokyo" read as a 4-day trip,
because `_coerce_trip_length` scanned the message for *any* digit.
`_find_duration_days` now requires the number to sit next to a duration
noun, which killed that hallucination and picked up "a week" / "two weeks"
as a bonus: 93.9 % → 94.9 %, 2.3 % → 0 % hallucinations, 25 → 27 perfect.

The five remaining misses are honest known-hard cases and are left in
deliberately, so the number doesn't flatter itself: a typo'd budget cue
("budget is tight"), a destination buried behind a dollar figure, an
adjectival `travel_style`, a negated preference ("I *don't* want museums —
nightlife instead"), and one `must_visit_attractions` lost among a long
list of interests.

## Running in Google Colab

This is the easiest way for any team member to try the full chatbot —
no local Python setup, no GPU of your own required. Google Colab gives you
a free, temporary GPU in your browser.

1. **Turn on the free GPU.** Menu bar: **Runtime → Change runtime type →
   Hardware accelerator → T4 GPU → Save.** (Skipping this step means the
   model runs on CPU only, which is much slower.)
2. **Get the code into Colab.** Two ways to do this — pick one:

   **Option A — upload the zip (simplest):**
   - Download `travel_assistant_project.zip` from this repo (on GitHub,
     click the file → **Download raw file**).
   - In Colab, click the folder icon in the left sidebar, then the upload
     icon (a page with an up arrow) near the top of that panel, and select
     `travel_assistant_project.zip`. It lands in Colab's working directory.
   - Add a new code cell and run:
     ```python
     !unzip -q travel_assistant_project.zip
     ```
     (If you have Colab Pro and a terminal panel, running
     `unzip travel_assistant_project.zip` there does the same thing. Free
     Colab has no terminal panel, so the cell above is the version that
     works for everyone.)
   - Then open `travel_assistant_colab.ipynb` from the file browser (or
     upload it too, if you only grabbed the zip) so you have the rest of
     the cells to run.

   **Option B — clone from GitHub instead:** open
   `notebooks/travel_assistant_colab.ipynb` and run its `!git clone ...`
   cell. This always gets the latest code from `main`, instead of a
   point-in-time zip.
3. **Run every remaining cell, top to bottom** (`Shift+Enter` on each one,
   or **Runtime → Run all**). The first real run downloads the AI model
   (a couple of minutes) — this only happens once per Colab session.
4. **Wait for a public link.** The last cell prints a message ending in
   something like `Running on public URL: https://xxxxx.gradio.live` —
   click that link to open the chat window in a new tab.
5. **Chat with it.** Describe your trip in your own words to start (e.g.
   "5-day family trip to Japan, we love food and theme parks, moderate
   budget") — the assistant extracts what it can and only asks for
   whatever's still missing. Once everything's collected, it shows a
   summary card to confirm before generating the full trip plan, and lets
   you request changes ("increase my budget," "make it 10 days") afterward.
6. **When you're done**, go to **Runtime → Disconnect and delete runtime**
   to free up the shared GPU quota for the next team member.

**Troubleshooting:**

- *"Session crashed" / out of memory:* **Runtime → Restart runtime**, then
  run the cells again from the top — Colab's free GPU occasionally needs a
  fresh session.
- *No public link shows up:* scroll to the bottom of the last cell's
  output — it only appears after the model has finished loading.
- *The assistant seems to repeat its question:* wait for its message to
  fully finish printing before typing your reply.

**Keeping `travel_assistant_project.zip` up to date:** it's a snapshot,
not a live link to the code — if you change anything under
`travel_assistant/`, regenerate it before teammates rely on it again:

```bash
zip -FSrq travel_assistant_project.zip . \
  -x ".git/*" -x ".venv/*" -x "*__pycache__*" -x ".pytest_cache/*" \
  -x ".idea/*" -x ".claude/*" -x "*.DS_Store" -x "travel_assistant_project.zip" \
  -x ".playwright-mcp/*" -x "docs/_build/*" -x ".ruff_cache/*"
```

`-FS` (filesync) is deliberate: plain `-r` only adds and replaces, so
files you have since deleted would linger in the archive forever.

## Running locally (logic only, no GPU needed)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt   # runtime deps + pytest + ruff
pytest
ruff check . && ruff format --check .
```

The test suite never loads the real model, and importing `travel_assistant.app`
doesn't either — the model is built lazily on the first chat message
(`app.get_graph`). That's what lets CI run the whole suite on a CPU-only
runner. `.github/workflows/ci.yml` asserts it with an explicit import step,
so moving `build_chat_model()` back to module scope fails the build.

Dependency versions carry upper bounds on purpose: `HuggingFacePipeline`
reads invoke-time generation overrides only from a nested `pipeline_kwargs`
dict, and `from_model_id(device=...)` wants a legacy CUDA index rather than
a device string. A major bump would change either silently. Verified
against langgraph 1.2, langchain 1.3, transformers 5.14, gradio 6.20,
torch 2.13.

## Running the full demo locally

For running on your own machine instead of Colab — works on an Apple
Silicon Mac (MPS), a CUDA GPU, or CPU (slow, but functional); the backend
is auto-detected in `travel_assistant/llm.py`.

```bash
pip install -r requirements.txt
python -m travel_assistant.app
```

## Scope

Matches the project proposal: destination selection, preference collection,
itinerary generation, activity recommendations, transportation suggestions,
a budget estimate, and a packing checklist, all through a conversational
interface. It does not book flights/hotels, process payments, give visa
advice, or guarantee real-time pricing.
