# Understanding the Code (No Coding Background Needed)

This is for Jane, Varsha, Monir, Akshay — anyone on the team who wants to
follow along with what the code does, speak to it in the presentation or
Results Report, or review it without being a Python developer. It assumes
no programming experience. If you already know what a function or a
dictionary is, skim past the "tiny bit of Python" section.

If you just want to **see the chatbot work** without reading any code, jump
to [Trying it yourself](#trying-it-yourself) near the bottom.

## 1. The big picture, in plain English

The whole project is one conversation loop, but it's an *adaptive* one, not
a fixed script:

1. The chatbot opens with a friendly greeting explaining what it can do,
   then asks the user to describe their trip in their own words, instead
   of firing off a list of individual questions.
2. It reads that message and pulls out whatever details it can find
   (destination, trip length, who's traveling, interests, budget, and a
   few optional extras) — this is called **extraction**.
3. It checks which of the five *required* details are still missing, and
   if any are, asks for just those — the user can answer one, several, or
   all of them in a single reply, in any order.
4. Once everything required is known, it summarizes what it has in a
   plain sentence and asks the user to confirm or change something before
   generating anything.
5. Once confirmed, it hands the collected details to an AI language model
   five separate times — once to write an itinerary, once for activity
   ideas, once for transportation advice, once for a budget summary, once
   for a packing list — and stitches all five into one final message.
6. It then asks if the user wants any changes. If the user asks for one
   (e.g. "increase my budget" or "add more outdoor activities"), only the
   affected part(s) of the plan get regenerated — not the whole thing.
   This can repeat as many times as the user likes before they say
   "finalize."

That's it. Everything in the codebase is either (a) the plumbing that
manages that back-and-forth conversation, (b) the piece that talks to the
AI model, or (c) the chat window itself. There's no database, no user
accounts, no payment processing — nothing beyond what's needed for that one
loop.

## 2. A tiny bit of Python (just enough to follow along)

You'll see these patterns constantly in every file. You don't need to be
able to write them — just recognize what they mean when reading.

| You'll see | It means |
|---|---|
| `# some text` | A comment. Ignored by the computer, written for humans reading the code. |
| `"""some text"""` | A longer comment (called a "docstring"), usually explaining what a whole function or file does. |
| `def some_name(...):` | "Here's a reusable recipe called `some_name`." Everything indented underneath is the steps of that recipe. |
| `{"key": "value"}` | A dictionary — basically a labeled box. `{"destination": "Tokyo"}` means "the box labeled `destination` holds `Tokyo`." |
| `if something:` | "Only do the next indented lines if `something` is true." |
| `for x in list:` | "Do the next indented lines once for every item in `list`." |
| `import X` | "I'm borrowing tool `X` that someone else already built." |
| `f"Hello {name}"` | A template string — `{name}` gets swapped for whatever `name` actually holds. |
| `None` | Python's word for "nothing" / "empty" / "not answered yet." |

## 3. Folder tour

```
travel_assistant/
  state.py    <- the list of "boxes" (fields) the conversation fills in
  llm.py      <- connects to the AI language model
  graph.py    <- the actual conversation flow / decision logic
  app.py      <- the chat window the user actually sees and types into
tests/
  test_graph.py   <- automated checks that the conversation flow behaves correctly
notebooks/
  travel_assistant_colab.ipynb   <- a runnable copy of the demo for Google Colab
docs/
  design_document.md          <- the formal architecture write-up (diagrams, HCI heuristics)
  understanding_the_code.md   <- this document
```

Reading order that will make the most sense: **`state.py` → `graph.py` →
`llm.py` → `app.py`**. That's the order below.

## 4. Jargon glossary

Skim this once, then refer back as needed while reading Section 5.

| Term | Plain-English meaning |
|---|---|
| **LLM** | "Large Language Model" — the AI that actually writes sentences (itinerary text, packing list, etc) and also pulls structured details out of free-form messages. In our case, a small open-source model called Llama 3.2. |
| **Extraction** | Asking the AI to read a message and hand back a structured answer (e.g. "destination: Japan, budget: moderate") instead of a written paragraph. |
| **LangChain** | A toolkit that gives a consistent, simple way to "talk to" different AI models, so the rest of the code doesn't need to know the model's specific technical details. |
| **LangGraph** | A toolkit (built by the same people as LangChain) for describing a conversation as a flowchart — boxes ("nodes") connected by arrows ("edges"). This is what `graph.py` uses. |
| **Node** | One box in the flowchart — one step of the conversation (e.g., "extract details from the last message" or "write the itinerary"). |
| **Conditional edge** | An arrow in the flowchart whose destination depends on what's currently in the notebook (state) — e.g. "if anything's still missing, go ask for it; otherwise, show the summary." This is what makes the flowchart adaptive instead of a straight line. |
| **State** | The shared notebook every step reads from and writes to — literally just the answers and generated text collected so far. |
| **Interrupt / Resume** | The mechanism that lets the flowchart "pause" to wait for the user to type an answer, then "resume" exactly where it left off once they do. |
| **Checkpointer** | The thing that remembers where each user's conversation currently is (so two different people chatting at the same time don't get mixed up). |
| **Gradio** | The toolkit used to build the actual chat window/web page. |
| **Pipeline** | A packaged, ready-to-use bundle that takes in text and produces text out — used here to wrap the AI model. |
| **GPU / CUDA / MPS / CPU** | The hardware the AI model runs on. GPU = graphics chip, much faster for AI than the regular processor (CPU). CUDA = NVIDIA's GPU tech (used on Colab). MPS = Apple's equivalent on Mac computers. |
| **Quantization / 4-bit** | A trick to shrink the AI model so it needs less memory, at a small cost to quality. Not something you need to understand deeply — just know it's a memory-saving technique. |
| **Token** | Roughly, a chunk of a word. AI models generate text one token at a time; `max_new_tokens` just caps how long a generated answer can be. |

## 5. Walking through each file

### `state.py` — the shared notebook

```python
class TravelState(TypedDict, total=False):
    last_user_input: Optional[str]

    destination: Optional[str]
    trip_length_days: Optional[int]
    group_type: Optional[str]
    interests: Optional[str]
    budget_level: Optional[str]

    travel_style: Optional[str]
    travel_season: Optional[str]
    must_visit_attractions: Optional[str]

    collected_fields: Dict[str, str]
    missing_fields: List[str]
    conversation_stage: ConversationStage
    confirmation_status: Optional[ConfirmationStatus]
    extraction_attempts: int
    regeneration_targets: Optional[List[str]]

    itinerary: Optional[str]
    activities: Optional[str]
    transportation: Optional[str]
    budget_estimate: Optional[str]
    packing_list: Optional[str]
```

Read this as: *"Every conversation has these labeled boxes."* The five
required trip-detail boxes (`destination` through `budget_level`) and
three optional ones (`travel_style`, `travel_season`,
`must_visit_attractions`) get filled in by extracting from what the user
types. A handful of bookkeeping boxes (`collected_fields`,
`missing_fields`, `conversation_stage`, `confirmation_status`,
`extraction_attempts`, `regeneration_targets`) track where the
conversation currently stands — more on those in Section 5's walk through
`graph.py`. The last five boxes (`itinerary` through `packing_list`) get
filled in by the AI. `Optional[str]` just means "this box holds text, or
is empty (`None`) until it's filled in." That's the entire file — it
doesn't *do* anything, it just defines what the notebook looks like.

### `graph.py` — the conversation flowchart

This is the file worth spending the most time on, since it's the actual
"design" of the chatbot. Unlike a fixed list of questions asked in order,
this flowchart branches based on what's already known.

**Step 1 — turn free text into structured details:**

```python
def _make_extract_preferences_node(llm):
    def extract_preferences(state):
        updates = _merge_extracted(llm, state, overwrite=False)
        ...
        return updates
    return extract_preferences
```

Plain English: *"Send whatever the user just typed to the AI, ask it to
pull out any of the eight trip details it can find, and fill in only the
boxes that were still empty."* A field that's already filled in from an
earlier message is never overwritten — so if the user later says "moderate
budget," that doesn't erase a destination they mentioned two messages ago.

**Step 2 — check what's still missing (no AI call, just checking the notebook):**

```python
def validate_preferences(state):
    missing = [field for field in REQUIRED_PREFERENCE_FIELDS if not state.get(field)]
    ...
    return {"missing_fields": missing, "collected_fields": collected, ...}
```

Plain English: *"Look at the five required boxes. Which ones are still
empty? Write that list down, along with a readable snapshot of everything
that IS filled in."* This is the piece that decides — via the flowchart's
arrows — whether to ask more questions or move on. It never asks the AI to
decide this; it's a plain checklist.

**Step 3 — ask only for what's missing, freeze, and wait:**

```python
def request_missing_fields(state):
    ...
    answer = interrupt({"kind": "missing_fields", "collected_fields": ..., "missing_fields": ...})
    return {"last_user_input": answer}
```

Plain English: *"Tell the user what's known and what's still missing, in
a normal sentence, then freeze right here until they reply — however much
or little they give in that one reply."* That freeze-and-wait is the `interrupt(...)` call — it's the
single most important idea in the whole codebase. The reply goes right
back into Step 1 (`extract_preferences`), so answering with "5 days,
moderate budget, food and nature" all in one message works exactly like
answering one detail at a time — the flowchart loops between "ask for
what's missing" and "extract from the reply" until nothing's left.

**Step 4 — show a summary and get confirmation before generating anything:**

```python
def confirm_preferences(state):
    answer = interrupt({"kind": "summary_confirmation", "collected_fields": ...})
    if _is_confirmation(answer):
        return {"confirmation_status": "confirmed", ...}
    return {"last_user_input": answer, "confirmation_status": "pending"}
```

Plain English: *"Show everything collected so far as one tidy card. If the
user says something like 'yes' or 'looks good,' move on to generating the
plan. Otherwise, treat their reply as a correction and re-extract from
it."* `_is_confirmation` is just a small checklist of words/phrases like
"yes," "confirm," "looks good," "finalize."

**Five recipes that each call the AI once to write a section**, e.g.:

```python
def _itinerary_update(llm, state):
    prompt = (
        f"Destination: {state['destination']}\n"
        f"Trip length: {state['trip_length_days']} days\n"
        ...
        "Write a day-by-day itinerary for this trip..."
    )
    text = _generate(llm, ..., prompt)
    return {"itinerary": text}
```

Plain English: *"Build a text prompt out of whatever's in the notebook so
far, send it to the AI, and write the AI's answer into the `itinerary`
box."* The activities, transportation, and packing sections all follow
this exact same pattern — only the prompt wording changes. The budget
section is slightly different: the dollar number is calculated with plain
arithmetic (a rate-per-day table), not asked from the AI — only the
sentence describing that number is AI-written. This is deliberate: it
means the number itself can't be an AI mistake.

**Step 5 — after the plan is shown, offer to make changes:**

```python
def review_plan(state):
    answer = interrupt({"kind": "plan_review", "prompt": "Would you like any changes, or shall I finalize this plan?"})
    if _is_confirmation(answer):
        return {"confirmation_status": "confirmed", ...}
    return {"last_user_input": answer, ...}
```

If the user asks for a change (e.g. "increase my budget to luxury"), the
flowchart figures out which of the five sections that change actually
affects — changing the budget only touches the budget section; changing
the destination touches four of the five — and regenerates *only* those,
leaving the rest of the plan exactly as it was. This loops back to asking
"anything else?" so the user can make several rounds of small changes
before saying "finalize."

**The flowchart itself, at the bottom of the file**, wires all of the
above together with `add_edge` (a fixed next-step) and
`add_conditional_edges` (a next-step chosen by checking the notebook, e.g.
"go here if fields are still missing, otherwise go there"). See the
workflow diagram in [`design_document.md`](design_document.md) for the
same thing drawn as a picture — it's no longer a straight line top to
bottom; it has three places where the flowchart branches and loops.

### `llm.py` — connecting to the AI model

You can skim this one. The only thing worth knowing:

- It automatically figures out what hardware it's running on (a Mac's own
  graphics chip, a cloud GPU like Colab's, or just the regular processor)
  and loads a slightly different version of the model depending on which
  one it finds — this is invisible to the user and doesn't change how the
  chatbot behaves, just how fast it runs.
- The model is `Llama-3.2-1B-Instruct` — an open-source, free AI model
  small enough to run without a paid subscription or API key, as required
  by the project proposal. It's used both for extracting structured
  details and for writing the final plan's sections.

### `app.py` — the actual chat window

This connects everything above to an on-screen chat box (via a tool called
Gradio). The core idea, in plain English:

- The **first** thing the user types is real content — their free-form
  trip description — and gets sent straight into the flowchart for
  extraction, not thrown away.
- Every message after that answers whatever the chatbot most recently
  asked (missing details, a confirmation, or a change request).
- Depending on where the conversation is, the chatbot's reply is one of a
  few card types: a "here's what's missing" card, a "here's everything,
  please confirm" card, the full finished plan, or a plain "anything else?"
  question. `app.py` picks which one to show based on a label
  (`"kind"`) attached to whatever the flowchart just paused on.

## 6. `test_graph.py` — the safety net

This file doesn't run the real AI model at all (that would be slow and
need a GPU every time). Instead it swaps in a fake, instant "pretend AI":
for extraction, a small stand-in that reads the message with simple
pattern matching instead of a real AI call; for the five writing steps, a
model that just echoes back a snippet of its prompt. This lets the tests
check that the conversation flow itself is correct — the right fields get
extracted, nothing gets asked twice, editing one detail doesn't erase
another, and asking to change one section of the finished plan only
regenerates that section — without needing a GPU or an internet
connection. If someone changes `graph.py` and breaks the flow, these tests
catch it immediately. You don't need to read this file in detail; just
know that "tests passing" means "the conversation logic still works as
designed."

## 7. Small, safe edits a non-coder can make

These are low-risk, easy to spot-check, and don't require understanding
the rest of the code:

- **Change the opening prompt** the bot shows: edit the `description=` text
  in `app.py` (currently "Tell me about the trip you would like to
  plan.").
- **Change the confirmation wording the user sees**: edit the sentences
  inside `_render_interrupt` in `app.py` (e.g. "Please confirm, or tell me
  what to change.").
- **Change the wording an AI-generation step is instructed to follow**:
  edit the quoted instruction text inside `_itinerary_update`,
  `_activities_update`, etc. in `graph.py` (e.g., "Write a day-by-day
  itinerary..." could become "Write a half-day-by-half-day itinerary...").
- **Add or remove a confirmation word**: edit the `_CONFIRM_WORDS` or
  `_CONFIRM_PHRASES` list near the top-middle of `graph.py` if the bot
  isn't recognizing a phrase as a "yes."

**Don't touch** (ask Peng Fei first): anything in `llm.py`; the
`add_node`/`add_edge`/`add_conditional_edges` lines in `graph.py`; the
extraction system prompts (`_EXTRACTION_SYSTEM_PROMPT`,
`_EDIT_EXTRACTION_SYSTEM_PROMPT`); or `state.py` — these control how the
pieces are wired together and how the AI is instructed to return
structured data, and a small typo there can break the whole app in ways
that are hard to spot without running it.

## 8. Trying it yourself

You don't need to read or understand any code to see the chatbot in
action:

1. Open `notebooks/travel_assistant_colab.ipynb` in Google Colab.
2. Menu bar: **Runtime → Change runtime type → T4 GPU**, then **Save**.
3. Run each cell top to bottom (the play button next to each cell, or
   **Runtime → Run all**).
4. The last cell prints a public URL — open it and chat with the
   assistant. Try describing your whole trip in one message (e.g. "5-day
   solo trip to Lisbon, I love food and beaches, moderate budget") and
   notice it skips straight to a summary card instead of asking five
   separate questions.

For a plain-English tour of *why* it's built this way (with pictures),
see [`design_document.md`](design_document.md).
