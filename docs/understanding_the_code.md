# Understanding the Code (No Coding Background Needed)

This is for Jane, Varsha, Monir, Akshay — anyone on the team who wants to
follow along with what the code does, speak to it in the presentation or
Results Report, or review it without being a Python developer. It assumes
no programming experience. If you already know what a function or a
dictionary is, skim past the "tiny bit of Python" section.

If you just want to **see the chatbot work** without reading any code, jump
to [Trying it yourself](#trying-it-yourself) near the bottom.

## 1. The big picture, in plain English

The whole project is one conversation loop:

1. The chatbot asks the user five quick questions (destination, trip
   length, who's traveling, interests, budget).
2. Once it has all five answers, it hands those answers to an AI language
   model five separate times — once to write an itinerary, once for
   activity ideas, once for transportation advice, once for a budget
   summary, once for a packing list.
3. It stitches those five pieces of text together into one final message
   and shows it to the user.

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
| **LLM** | "Large Language Model" — the AI that actually writes sentences (itinerary text, packing list, etc). In our case, a small open-source model called Llama 3.2. |
| **LangChain** | A toolkit that gives a consistent, simple way to "talk to" different AI models, so the rest of the code doesn't need to know the model's specific technical details. |
| **LangGraph** | A toolkit (built by the same people as LangChain) for describing a conversation as a flowchart — boxes ("nodes") connected by arrows ("edges"). This is what `graph.py` uses. |
| **Node** | One box in the flowchart — one step of the conversation (e.g., "ask for destination" or "write the itinerary"). |
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
    destination: Optional[str]
    trip_length_days: Optional[int]
    group_type: Optional[str]
    interests: Optional[str]
    budget_level: Optional[str]

    itinerary: Optional[str]
    activities: Optional[str]
    transportation: Optional[str]
    budget_estimate: Optional[str]
    packing_list: Optional[str]
```

Read this as: *"Every conversation has these ten labeled boxes. The first
five get filled in by the user's answers; the last five get filled in by
the AI."* `Optional[str]` just means "this box holds text, or is empty
(`None`) until it's filled in." That's the entire file — it doesn't *do*
anything, it just defines what the notebook looks like.

### `graph.py` — the conversation flowchart

This is the file worth spending the most time on, since it's the actual
"design" of the chatbot. It has two kinds of things in it:

**A list of the five questions, in order:**

```python
PREFERENCE_QUESTIONS = (
    ("destination", "Where would you like to travel to?"),
    ("trip_length_days", "How many days will your trip be?"),
    ("group_type", "Who's traveling — solo, a couple, family, friends, or students?"),
    ("interests", "What are you most interested in on this trip ..."),
    ("budget_level", "What's your budget level — budget, moderate, or luxury?"),
)
```

If you ever want to reword a question, **this is the one place to change
it** — see [Section 7](#7-small-safe-edits-a-non-coder-can-make).

**A recipe (`collect_preferences`) that asks each question and waits:**

```python
def collect_preferences(state: TravelState) -> dict:
    for field, question in PREFERENCE_QUESTIONS:
        if state.get(field) or updates.get(field):
            continue          # already answered — skip it
        answer = interrupt({"field": field, "question": question})
        ...
```

In plain English: *"Go through the five questions in order. If one's
already answered, skip it. Otherwise, ask it and freeze right here until
the user replies."* That freeze-and-wait is the `interrupt(...)` call —
it's the single most important idea in the whole codebase, so it's worth
re-reading this paragraph once.

**Five recipes that each call the AI once**, e.g.:

```python
def generate_itinerary(state: TravelState) -> dict:
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
box."* `generate_activities`, `generate_transportation`, and
`generate_packing` all follow this exact same pattern — only the prompt
wording changes. `generate_budget` is slightly different: the dollar
number is calculated with plain arithmetic (a rate-per-day table), not
asked from the AI — only the sentence describing that number is
AI-written. This is deliberate: it means the number itself can't be an AI
mistake.

**The flowchart itself, at the bottom of the file:**

```python
builder.add_edge(START, "collect_preferences")
builder.add_edge("collect_preferences", "generate_itinerary")
builder.add_edge("generate_itinerary", "generate_activities")
builder.add_edge("generate_activities", "generate_transportation")
builder.add_edge("generate_transportation", "generate_budget")
builder.add_edge("generate_budget", "generate_packing")
builder.add_edge("generate_packing", END)
```

This is literally just: ask questions, then itinerary, then activities,
then transportation, then budget, then packing, then done — read top to
bottom, it's a straight line. See the workflow diagram in
[`design_document.md`](design_document.md) for the same thing drawn as a
picture.

### `llm.py` — connecting to the AI model

You can skim this one. The only thing worth knowing:

- It automatically figures out what hardware it's running on (a Mac's own
  graphics chip, a cloud GPU like Colab's, or just the regular processor)
  and loads a slightly different version of the model depending on which
  one it finds — this is invisible to the user and doesn't change how the
  chatbot behaves, just how fast it runs.
- The model is `Llama-3.2-1B-Instruct` — an open-source, free AI model
  small enough to run without a paid subscription or API key, as required
  by the project proposal.

### `app.py` — the actual chat window

This connects everything above to an on-screen chat box (via a tool called
Gradio). The core idea, in plain English:

- The first thing the user types just "wakes up" the conversation and gets
  thrown away — the bot's first real reply is the first question.
- Every message after that is treated as the answer to whatever question
  the bot most recently asked.
- Once all five questions are answered and the AI has generated all five
  sections, the bot's next reply is the entire finished trip plan, nicely
  formatted with headings.

## 6. `test_graph.py` — the safety net

This file doesn't run the real AI model at all (that would be slow and
need a GPU every time). Instead it swaps in a fake, instant "pretend AI"
and checks that the conversation flow itself is correct — the right
questions get asked in the right order, answers land in the right boxes,
and nothing gets asked twice. If someone changes `graph.py` and breaks the
flow, these tests catch it immediately. You don't need to read this file
in detail; just know that "tests passing" means "the conversation logic
still works as designed."

## 7. Small, safe edits a non-coder can make

These are low-risk, easy to spot-check, and don't require understanding
the rest of the code:

- **Reword a question** the bot asks: edit the second item in each tuple
  inside `PREFERENCE_QUESTIONS` in `graph.py` (Section 5 above).
- **Change the chat window's title or description**: edit the `title=` and
  `description=` text in `app.py`.
- **Change the wording an AI-generation step is instructed to follow**:
  edit the quoted instruction text inside `_make_itinerary_node`,
  `_make_activities_node`, etc. in `graph.py` (e.g., "Write a day-by-day
  itinerary..." could become "Write a half-day-by-half-day itinerary...").

**Don't touch** (ask Peng Fei first): anything in `llm.py`, the
`add_node`/`add_edge` lines in `graph.py`, or `state.py` — these control
how the pieces are wired together, and a small typo there can break the
whole app in ways that are hard to spot without running it.

## 8. Trying it yourself

You don't need to read or understand any code to see the chatbot in
action:

1. Open `notebooks/travel_assistant_colab.ipynb` in Google Colab.
2. Menu bar: **Runtime → Change runtime type → T4 GPU**, then **Save**.
3. Run each cell top to bottom (the play button next to each cell, or
   **Runtime → Run all**).
4. The last cell prints a public URL — open it and chat with the
   assistant.

For a plain-English tour of *why* it's built this way (with pictures),
see [`design_document.md`](design_document.md).
