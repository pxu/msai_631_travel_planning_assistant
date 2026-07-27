# AI-Powered Travel Assistant Planner

Group 4 — MSAI-631-B01. A conversational assistant that collects trip
preferences and generates a personalized itinerary, activity picks,
transportation guidance, a budget estimate, and a packing checklist.

See [`docs/design_document.md`](docs/design_document.md) for the detailed
design doc (architecture diagram, workflow/state diagrams, interaction
sequence, UI design, and HCI heuristics). Not a coder? Start with
[`docs/understanding_the_code.md`](docs/understanding_the_code.md) instead —
a plain-English walkthrough of what each file does, no programming
background required.

## Architecture

Built with **LangGraph** instead of calling `transformers` directly. The
conversation is modeled as an explicit `StateGraph` (`travel_assistant/graph.py`):

```
START -> collect_preferences -> generate_itinerary -> generate_activities
      -> generate_transportation -> generate_budget -> generate_packing -> END
```

- `collect_preferences` uses LangGraph's `interrupt()` / `Command(resume=...)`
  human-in-the-loop mechanism to ask one question at a time (destination,
  trip length, traveler group, interests, budget level) and pause until the
  user answers — no LLM-driven routing involved, so a small local model only
  ever has to generate text, never decide what happens next.
- Each `generate_*` node calls the LLM once to produce its section.
  The budget estimate combines a deterministic rate-table calculation with
  an LLM-written summary, so the number itself doesn't depend on the model
  getting arithmetic right.
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
  full graph (all 5 preference questions + itinerary/activities/transportation/
  budget/packing generation) runs locally with no cloud dependency at all.

## Project layout

```
travel_assistant_project.zip   # ready-to-upload snapshot of this project, for Colab
travel_assistant/
  state.py    # TravelState TypedDict — the graph's shared state schema
  llm.py      # loads the local model as a LangChain chat model
  graph.py    # the LangGraph StateGraph and its nodes
  app.py      # Gradio ChatInterface wiring
tests/
  test_graph.py   # graph control-flow tests using a fake chat model (no GPU needed)
notebooks/
  travel_assistant_colab.ipynb   # thin notebook to run the Gradio demo on Colab
```

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
   `notebooks/travel_assistant_colab.ipynb`'s first code cell and follow
   the comments there (`!git clone ...`). This always gets the latest
   code instead of a point-in-time zip, at the cost of one extra step
   (editing in the branch name, until PR #1 is merged).
3. **Run every remaining cell, top to bottom** (`Shift+Enter` on each one,
   or **Runtime → Run all**). The first real run downloads the AI model
   (a couple of minutes) — this only happens once per Colab session.
4. **Wait for a public link.** The last cell prints a message ending in
   something like `Running on public URL: https://xxxxx.gradio.live` —
   click that link to open the chat window in a new tab.
5. **Chat with it.** Type anything to start (e.g. "hi"), then answer each
   question the assistant asks in order (destination, trip length, who's
   traveling, interests, budget). After the fifth answer, it generates and
   shows the full trip plan.
6. **When you're done**, go to **Runtime → Disconnect and delete runtime**
   to free up the shared GPU quota for the next team member.

**Troubleshooting:**

- *"Session crashed" / out of memory:* **Runtime → Restart runtime**, then
  run the cells again from the top — Colab's free GPU occasionally needs a
  fresh session.
- *No public link shows up:* scroll to the bottom of the last cell's
  output — it only appears after the model has finished loading.
- *The assistant seems to repeat a question:* wait for its question to
  fully finish printing before typing your reply.

**Keeping `travel_assistant_project.zip` up to date:** it's a snapshot,
not a live link to the code — if you change anything under
`travel_assistant/`, regenerate it before teammates rely on it again:

```bash
zip -rq travel_assistant_project.zip . \
  -x ".git/*" -x ".venv/*" -x "*__pycache__*" -x ".pytest_cache/*" \
  -x ".idea/*" -x ".claude/*" -x "*.DS_Store" -x "travel_assistant_project.zip"
```

## Running locally (logic only, no GPU needed)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest tests/
```

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
