# Interview Prep — Talking About This Project

Simple-English scripts for an AI Engineer interview.

- **Part A — Technical** (§1–§11): architecture, the tech stack, the
  trade-offs, your stories, senior-depth topics, the numbers.
- **Part B — Non-technical** (§12–§20): you, the team, mistakes, honesty
  about AI tools, and questions to ask them.

Everything factual here is true of the code in this repo. Numbers come from
`python -m evals.extraction_eval` and `pytest`. Where a section needs
**your own experience**, it says so and gives you a fill-in prompt — do not
recite a story that isn't yours.

**The one thing to internalize:** your strongest material is *not* "I built
a chatbot." Many candidates built a chatbot. Your strongest material is
that you **measured** your model, **found a real bug with that
measurement**, and **designed around a weak model instead of hoping it
would behave**. Lead with that.

---
---

# PART A — TECHNICAL

---

## 1. The 60-second pitch

> I built a travel planning assistant. You describe your trip in plain
> English, and it produces a day-by-day itinerary, activity suggestions,
> transport advice, a budget, and a packing list.
>
> The interesting part is the constraint. It runs a **1-billion-parameter
> model locally** — no API, no key, no cloud. A model that small is not
> reliable. So the whole design is about **giving the model as little
> responsibility as possible**.
>
> The conversation is a **LangGraph state machine**. Every routing
> decision — is a field still missing? did the user confirm? — is plain
> Python reading state. The model is only ever asked to do one small thing:
> pull structured fields out of one message, or write one section of text.
> It never decides what happens next.
>
> And I don't assume it does its job well — I measure it. There's an eval
> set of 33 labelled sentences. Current score: **95.2% field accuracy and
> 0% hallucination rate**.

Then **stop talking** and let them ask. Resist the urge to keep going.

---

## 2. "Walk me through the architecture"

### 2a. The 45-second version (say this first)

> A user message goes into a LangGraph state machine. First node extracts
> preferences — one model call, returns JSON. Second node is pure Python:
> it checks which of the five required fields are still empty. That check
> drives a conditional edge — if something is missing we loop back and ask
> only for that; if everything is there we show a summary and ask the user
> to confirm.
>
> After they confirm, five generation nodes run in order, one model call
> each. Then a review step. If the user asks for a change, I work out which
> fields actually changed, map those to the sections they affect, and
> regenerate **only those sections**. So "make it 5 days instead of 3"
> reruns the itinerary, budget, and packing list — but not the transport
> advice, because trip length doesn't change how you get around.

Then stop. If they want more, they'll ask — and 2b/2c is what you give
them.

### 2b. The layers, and why each boundary exists

Draw this if you have a whiteboard. Five layers, each with one job:

```
Browser
   │  chat messages
┌──▼──────────────────────────────────────────┐
│ Presentation   app.py (Gradio)              │  renders cards, owns no logic
├─────────────────────────────────────────────┤
│ Conversation   graph.py + state.py          │  ALL routing decisions (Python)
│                LangGraph StateGraph         │  + checkpointer, one thread/user
├─────────────────────────────────────────────┤
│ AI processing  llm.py (LangChain wrapper)   │  one interface: invoke(msgs)->msg
│                Llama 3.2 1B, local          │  extract fields | write one section
├─────────────────────────────────────────────┤
│ Business logic _estimate_budget, formatters │  the arithmetic, in plain Python
└─────────────────────────────────────────────┘
```

The line to say about each boundary is **"what breaks if it isn't there":**

| Boundary | What it buys | What breaks without it |
|---|---|---|
| UI ↔ graph | UI holds no conversation logic; it renders whatever card the graph paused on | Conversation state ends up half in the browser and half in Python — untestable |
| Graph ↔ model | The model is called, never consulted about flow | A 1B model decides control flow, and the conversation becomes non-deterministic and untestable |
| Model ↔ business logic | Money is computed in Python | The number users are most likely to check is produced by the component least able to compute it |
| `llm.py` wrapper | Swap the model without touching graph code | Model details leak into 15 nodes; changing model becomes a rewrite |

### 2c. One turn, end to end (the concrete trace)

Interviewers love a specific trace far more than a diagram. Use this one:

> Say the user types *"a relaxed week in Kyoto with my wife, we love
> temples and food."*
>
> 1. `app.py` looks at the graph's checkpoint to decide: is this a new
>    conversation or a resume? New → invoke with the message as
>    `last_user_input`.
> 2. `extract_preferences` — **one model call**, deterministic decoding,
>    asks for JSON with eight fields.
> 3. The raw output is parsed by a JSON reader that can repair truncation,
>    because a small model often stops before the closing brace.
> 4. Then code-level checks run on each field: is this value actually
>    *grounded* in what the user typed, or did the model invent it? "week"
>    becomes 7 days. "my wife" becomes `couple`. Anything unmappable
>    becomes `None`.
> 5. `validate_preferences` — **no model call**. Recomputes what's missing
>    from scratch. Here, only budget.
> 6. Conditional edge → `request_missing_fields` → the graph **interrupts**
>    and control returns to the UI, which renders "I have destination, trip
>    length, group, interests. What's your budget?"
>
> The user's next message resumes the graph at exactly that point. State
> lives in the checkpointer, keyed by a per-session thread ID, so two
> people using it at once never collide.

**The sentence that lands:** *"Notice the only thing the model did was turn
one sentence into JSON. Everything else — what's missing, what to ask,
where to go next — was Python."*

---

## 3. The tech stack: what each piece does, and why that one

Have an opinion about every dependency. "It's what the tutorial used" is a
bad answer; so is silence.

| Technology | What it does *here* | Why this one | What I'd consider instead |
|---|---|---|---|
| **LangGraph** | The conversation as an explicit `StateGraph`: 15 nodes, 3 conditional edges, plus checkpointing and interrupt/resume | I needed three things together: explicit control flow, the ability to **pause mid-graph** for user input and resume exactly there, and per-user state. Writing that by hand is a state machine plus a persistence layer — that's the library's whole job | Plain `if/else` for something this small; a durable-workflow engine (Temporal) if steps had to survive restarts |
| **LangChain** (`langchain-huggingface`) | Wraps the local model so everything else sees `invoke(messages) -> AIMessage` | It's the seam. Because of it, swapping the model is one env var and no graph changes. I use a *fake* model implementing the same interface for all 163 tests | Calling `transformers` directly — fewer layers, but then every node knows about tokenizers and I lose the fake-model trick |
| **HuggingFace `transformers`** | Loads and runs the model locally | It's the standard for open-weight local inference, and it's what supports both the 4-bit CUDA path and the fp16 Apple Silicon path from one codebase | vLLM or llama.cpp if throughput mattered — much faster serving, more setup |
| **Llama 3.2 1B Instruct** | Extraction + text generation | Project constraint: free, local, no API key. Small enough for a free Colab T4 or a laptop | A 3B/8B for better instruction-following; an API model if cost and privacy allowed |
| **Gradio** (`ChatInterface`) | The chat UI, plus a public share link for Colab | Zero frontend code for a working chat window, and the share link is how non-technical teammates could try it | FastAPI + a real frontend if I needed control over rendering — see the trade-off in §4.7 |
| **`TypedDict`** for state | Declares the ~20 state fields and their types | One file is the single source of truth for the schema. `Literal` types make the closed vocabularies (`budget_level`, `group_type`) checkable | Pydantic if I needed runtime validation and coercion — heavier, but validates at the boundary |
| **`InMemorySaver`** | Per-thread checkpoint store | Matches the scope: no cross-session persistence was in scope | Postgres/Redis checkpointer — same LangGraph interface, so the graph code wouldn't change |
| **pytest / ruff / GitHub Actions** | 163 tests, lint+format, CI on 3.11–3.13 | Tests run with a fake model so CI needs no GPU and finishes in seconds | — |

**If they ask "why not just call the API directly, why all the layers?"**

> For a project this size you could. The layers earn their place in two
> specific ways: the LangChain seam is what lets 122 tests run without a
> GPU, because I can substitute a fake model behind the same interface; and
> the LangGraph structure is what makes selective regeneration possible,
> because the field-to-section mapping needs an explicit graph to act on. If
> neither of those mattered, I'd agree it's over-built.

---

## 4. Trade-offs — the section that separates senior from junior

This is the highest-value part of Part A. The format that sounds senior is
always the same four beats: **what I chose · what I gained · what I gave
up · when I'd revisit it.** Never skip the third beat — a candidate who
can't name the cost of their own decision hasn't really made one.

### 4.1 Explicit state graph vs. an agent loop

- **Gained:** deterministic, testable control flow. I can assert that
  editing "trip length" regenerates exactly three sections.
- **Gave up:** flexibility. The assistant can only do what I drew. It can't
  handle "actually, book me a flight" — it has no concept of that.
- **Revisit when:** the task space is genuinely open-ended, *and* the model
  is strong enough to plan. Neither was true here.

### 4.2 Small local model vs. a hosted API model

- **Gained:** zero cost, no API key, works offline, no user data leaves the
  machine. And honestly it forced better engineering.
- **Gave up:** quality. It misses things a bigger model wouldn't — my eval
  shows 5 misses out of 33 cases, including a negated preference ("I *don't*
  want museums").
- **Revisit when:** quality per dollar matters more than privacy and cost.
  I made this a one-line change (`TRAVEL_ASSISTANT_MODEL`) specifically so
  it can be re-measured rather than re-argued.

### 4.3 Deterministic budget vs. letting the model compute it

- **Gained:** the number is always arithmetically right and reproducible.
- **Gave up:** nuance. A real model could reason about Kyoto being pricier
  than Lisbon; my rate table can't. My number is *consistent*, not *smart*.
- **Revisit when:** I have real pricing data to ground it. The fix is a
  data problem, not a model problem.

### 4.4 Closed vocabularies + raise, vs. free text + defaults

- **Gained:** failures are loud. An unrecognized budget level goes back to
  "missing" and gets re-asked instead of silently costing at a default.
- **Gave up:** the system now sometimes asks a question it used to answer.
  That's a slightly worse experience in exchange for never being quietly
  wrong.
- **Revisit when:** never, honestly. For anything involving money I'll take
  a loud failure over a quiet wrong answer every time.

### 4.5 Free-text extraction vs. a fixed questionnaire

- **Gained:** far fewer turns. Front-load everything in one sentence and
  you go straight to the summary.
- **Gave up:** a much larger failure surface. A fixed form can't
  misunderstand you. Extraction can — and most of the hardening code in the
  repo exists to pay this bill.
- **Revisit when:** the fields become high-stakes or legally binding, where
  an explicit form is the right interface.

### 4.6 Five separate model calls vs. one big call

- **Gained:** each call is a small, focused task the 1B model can handle,
  and I can regenerate one section without touching the others.
- **Gave up:** latency. Five sequential calls means the user waits in
  silence — the single worst thing about the current UX.
- **Revisit:** immediately, and I know the fix. Stream the first section as
  soon as it's ready, and run the independent sections in parallel —
  activities and packing don't depend on each other.

### 4.7 Gradio vs. a custom frontend

- **Gained:** a working chat UI in a few lines, and a shareable link that
  let non-technical teammates try it.
- **Gave up:** rendering control. And it cost me a real bug: because the UI
  only ever displays "whatever the graph paused on," it was easy to write a
  pause that returned the *question* without the *plan* — so users were
  asked to approve something invisible. A hand-written frontend would
  probably have made that obvious sooner.
- **Revisit when:** the UI needs anything beyond a message list.

### 4.8 Fake model in tests vs. testing against the real one

- **Gained:** 163 tests in ~3 seconds, no GPU, CI on free runners.
- **Gave up:** those tests say **nothing** about real model quality.
- **How I covered it:** that gap is exactly why the eval harness exists.
  Two layers answering two different questions.

### 4.9 Hand-written keyword and regex backstops vs. trusting the model

- **Gained:** deterministic recovery from known failure modes — "my wife
  and son" → family, "a week" → 7 days.
- **Gave up:** generality, and there's a real risk of **overfitting to my
  own eval set** — I fix a case, the number goes up, and I've learned
  nothing about unseen input.
- **How I manage it:** I only add a rule when it generalizes as English
  ("a number next to a duration word"), not when it patches one sentence.
  Naming this risk unprompted is itself a strong signal.

---

## 5. Your five strongest technical stories

Use **Situation → Problem → What I did → Result**. Keep each under 90
seconds.

### Story A — "I measured the model instead of trusting it" ⭐ best one

> My unit tests all used a fake model. That was correct for testing the
> conversation logic, but it meant I had **zero evidence** the real model
> could actually do its job. I had written a lot of defensive code —
> checks for the model inventing values, keyword fallbacks — and I couldn't
> say whether any of it helped.
>
> So I built an eval set: 33 real sentences a traveler might type, each
> labelled with what a correct reading should produce. I scored every field
> into four buckets: hit, miss, wrong, and **hallucination**.
>
> Hallucination is the one I care about. If the sentence never says who is
> travelling and the model answers "solo," nothing downstream can tell that
> was invented — it silently changes the itinerary, the transport advice,
> and the price. Whereas if the model says "I don't know," the assistant
> just asks. So **declining to guess is a success, not a failure**, and I
> score it separately.
>
> First run: 93.9% accuracy, one hallucination. The eval showed me exactly
> what it was — "Group of 4 going to Tokyo" was being read as a **4-day
> trip**, because my number parser grabbed any digit in the sentence. I
> fixed it to require the number be next to a duration word. That removed
> the hallucination and also picked up "a week" and "two weeks," which a
> digit scan can never find. Score went to 95.2%, hallucinations to zero.

**If they ask "what does 0% hallucination really mean?"** — be honest:

> On this 33-case set, with this model. It's not a guarantee. It's a
> regression baseline — if I change a prompt and hallucinations appear, I
> find out immediately instead of in front of a user.

**Also mention the zero:**

> One number I'm proud of: **zero "wrong" values**. Every failure is a
> miss, never a wrong answer. When this system states something, it's
> right. That's the shape you want, because a miss becomes a follow-up
> question and a wrong value becomes a silently bad plan.

### Story B — "I kept the math away from the model"

> Budget is the one output a user will actually check. A 1B model cannot
> do arithmetic reliably. So the dollar figure is computed in **plain
> Python** — a rate table times group size times number of days — and the
> model only writes the sentence around it.
>
> That wasn't enough. When I ran it, the model was told a $5,000 budget and
> wrote "approximately $1,200." Another time it wrote "$1,800 to $2,200 per
> night for 3 nights, so approximately $540 total" — which contradicts
> itself. My prompt said "don't invent numbers" and it ignored that.
>
> So I added a check: scan the generated text for dollar amounts, and if
> any of them disagree with the authoritative figure, throw the whole
> paragraph away and use a deterministic sentence instead. And log a
> warning, so I can measure how often it happens.
>
> It fired in my last test run — the model quoted $10, $20, $200, and $300
> against a real total of $2,450. The user never saw any of that.

**The principle to say out loud:** *"A prompt is a request, not a
guarantee. If something must be true, enforce it in code."*

### Story C — "I found bugs by actually running it"

> I had over a hundred passing tests and the app was still broken in ways the tests
> couldn't see. I only found them by opening the browser and using it like
> a user.
>
> The worst one: after you confirm your preferences, it generated all five
> sections — and then showed you only the question "would you like any
> changes?" **It never showed you the plan.** You were being asked to
> approve something invisible. The sections existed in state, but the code
> only rendered them after you'd already said yes.
>
> Second one: once you finalized a plan, every message after that replayed
> the same plan forever. Ask for Rome after finishing Japan, get Japan
> back. The cause was a boolean flag that said "this conversation has
> started" and never got reset. I replaced it — now the graph's own
> checkpoint decides whether a message continues the old conversation or
> starts a new one. That also fixed the Clear button, which had the same
> problem.
>
> Both now have regression tests.

**The lesson to say:** *"Tests tell you the code does what you wrote. They
don't tell you what you wrote is the right thing. You still have to use the
product."*

### Story D — "I made it possible to run in CI"

> The model was being loaded at import time — one line at the top of the
> file. That meant you couldn't even *import* the app without downloading
> and loading a 1B model. No CI, no fast test, no lint pass.
>
> I made it lazy: the graph is built on the first real request and cached.
> Import is now free. CI runs the whole suite on a CPU-only runner in
> seconds, and there's an explicit "import the app" step in the pipeline so
> that if anyone moves it back, the build fails.

Short, but a very legible signal that you've shipped software before.

### Story E — "A code review found the same class of bug in my own fix" ⭐ best senior story

This is the most senior-sounding story you have, because it's about
**intellectual honesty and generalizing from a defect**, not about being
clever.

> I'd built a normalization layer whose entire purpose was to stop the
> system quietly accepting a value the user never gave. I was pleased with
> it. Then I put the change through review, and the review found that my
> normalizer did **exactly the thing it existed to prevent**.
>
> I was matching budget words by substring. So "mid" matched inside
> "pyramids," and "low" matched inside "flower." A message saying *"a
> 7-day trip to Egypt to see the pyramids"* silently acquired a **moderate
> budget** the user never mentioned — and because the field then looked
> filled, the system never asked, and costed the whole trip at that
> invented rate.
>
> Three things I'd point out about how I handled it. First, I **reproduced
> every finding before fixing anything** — I ran each claimed input through
> the real function rather than trusting the report. Second, I fixed the
> *class*, not the instances: boundary-anchored matching, plus a separate,
> stricter vocabulary for scanning free text than for reading the model's
> own answer — because "mid September" is a date and "low season" is a
> season, and those words only mean a budget tier when the model is
> answering that specific field. Third, I checked the fix didn't cost
> recall: the eval score was identical afterwards, 95.2%, so I'd removed
> false positives without losing true ones.
>
> The review found three more in the same pass — a dollar figure like
> "$200 a night" being read as the whole trip budget, and an exception in a
> node permanently trapping the session because the checkpoint kept
> pointing at the failed step. All four are fixed with regression tests.

**The line that makes this senior:**

> The lesson wasn't "be more careful." It was that I'd written *validation*
> code and never validated it. The layer that's supposed to catch bad input
> is exactly the layer where a silent bug is most dangerous, because
> everything downstream now trusts it.

**If they ask "was this an AI review or a human review?"** — answer
honestly (§13 covers the general version). The interesting part is
unaffected: what matters is that you sought review, reproduced before
believing, and generalized the fix.

---

## 6. Depth topics for a senior interview

Part A up to here handles most interviews. This section is for when they
push — a staff/senior loop, or an interviewer who keeps asking "and then
what?" Have an opinion ready on each.

### 6.1 Failure-mode taxonomy — the framing that impresses

Most candidates list bugs. Classify them instead:

> I think about failures in this system in three buckets, and they need
> completely different treatment.
>
> **Loud failures** — the model call throws, the process dies. Easy: catch,
> log, show the user something useful, retry if transient.
>
> **Quiet wrong answers** — the model invents a group type, or a budget
> word matches inside an unrelated word. These are the dangerous ones,
> because nothing downstream can tell. Every code-level check I have exists
> for this bucket, and the eval's hallucination metric exists to measure it.
>
> **Absent answers** — the model returns null. These look like failures and
> are actually the *good* outcome, because they turn into a follow-up
> question instead of a bad plan. A lot of my design is deliberately
> converting bucket two into bucket three.

That last sentence is the thesis of the whole project. If you say one
sophisticated thing in the interview, say that.

### 6.2 Evaluation methodology — beyond "I have an eval set"

Be ready to defend the design of the eval, not just its existence:

- **Why four buckets and not accuracy?** Because accuracy hides the
  distinction that matters. A miss and a hallucination are both "not a
  hit," but one costs a follow-up question and the other corrupts the
  output. Collapsing them into one number would have hidden the "Group of
  4 → 4-day trip" bug entirely.
- **Why is a null expectation a real expectation?** Because most of the
  risk is in fields the user *didn't* mention. 44 of my ~104 scored slots
  are "should be null." If I only scored fields that had answers, I'd be
  measuring half the problem.
- **How do you avoid overfitting to the eval?** Honest answer: I can't
  fully, with 33 cases. What I do is only add a rule when it generalizes as
  English — "a number must be adjacent to a duration noun" is a language
  fact; "special-case the word Tokyo" would be overfitting. And I'd want a
  held-out set before trusting the number for anything real.
- **What's missing?** Generation quality. I measure extraction, not whether
  the itinerary is factually correct. That needs either human rating or an
  LLM-as-judge setup, and I'd want to validate the judge against human
  labels before trusting it.

### 6.3 Observability

> Every model call logs latency, prompt size and response size at INFO, and
> the full prompt and completion at DEBUG — split deliberately, because the
> prompts contain the user's trip details and shouldn't be on by default.
> The budget verifier logs a WARNING when it rejects a narration, which
> gives me a **rate** rather than an anecdote: that's how I'd know whether
> a bigger model still needs the backstop.
>
> What's missing for production is tracing. Right now I have per-call logs
> but no per-turn trace tying five generation calls to one user request. I'd
> add OpenTelemetry spans, or one of the LLM-specific tracing tools, so I
> could see the whole turn as a tree with token counts attached.

### 6.4 Latency and cost

Know your numbers and the shape of the fix:

> Extraction is about 0.9 seconds. A generated section is 2–6 seconds, and
> there are five of them in series, so worst case the user waits close to
> 25 seconds seeing nothing. That's the dominant UX problem, and it's an
> architecture problem, not a model problem.
>
> Fixes in the order I'd do them: stream the first section as soon as it
> exists so time-to-first-token is a second, not twenty-five; run the
> independent sections concurrently — activities and packing don't depend
> on each other, though itinerary and budget both depend on trip length;
> then cache, because regenerating after an edit already only touches the
> affected sections. Only after all that would I look at a smaller or
> quantized model.

### 6.5 Concurrency and state

> State is per-conversation, keyed by a thread ID, held by the checkpointer
> — so concurrent users are isolated by construction, not by locking. The
> graph nodes are pure functions of state: they take state and return a
> partial update, which is what makes them individually testable and what
> would make them safe to run on different machines.
>
> The two things that aren't production-ready: state is in memory, so it
> dies with the process; and the model is loaded in-process, so the web
> tier and the GPU scale together whether you want that or not. Both are
> deliberate scope decisions, and both are swaps rather than rewrites — the
> checkpointer is an interface, and inference behind an HTTP call would only
> change `llm.py`.

### 6.6 "How would you productionize this?" — a concrete path

Give an ordered plan, not a wish list:

> 1. **Persistence** — swap `InMemorySaver` for a Postgres checkpointer.
>    Same interface, graph code unchanged. Now conversations survive a
>    deploy.
> 2. **Split inference out** — put the model behind its own service (vLLM
>    or a hosted endpoint). Only `llm.py` changes. Now web and GPU scale
>    independently, and one model server serves many app instances.
> 3. **Stream** — the biggest single UX win, and it needs the graph to
>    yield partial results rather than return once.
> 4. **Tracing and an eval gate in CI** — I already have the eval; I'd wire
>    `--min-accuracy` into the pipeline so a prompt change that drops
>    extraction quality fails the build the same way a failing test does.
> 5. **Then** the product work — grounding activities against a real places
>    API so the content is checkable, not just fluent.

**The framing sentence:** *"Notice steps 1 and 2 don't touch the state
machine at all. That's the return on drawing the boundaries early."*

### 6.7 Security and abuse (be honest — see §7 for the full answer)

The one-liner worth having ready:

> The model can't take actions — no tools, no spending, no control flow. So
> the worst case is bad text, not bad behaviour. That's a design property,
> not luck, and it's the main reason I'm comfortable running a small
> instruction-following model at all.

---

## 7. Technical questions they will probably ask

**"Why LangGraph and not an agent that decides for itself?"**

> Because the model is too small to be trusted with control flow. An agent
> loop asks the model "what should I do next?" — a 1B model answers that
> badly. My graph asks it only "what fields are in this sentence?" and
> "write this one section." Every decision is Python reading state. It's
> also easier to test and to reason about, and I can regenerate just one
> section on an edit, which an agent loop makes hard.

**"Why such a small model?"**

> It was a project constraint — run locally, free, no API key. But I'd
> defend it as a design exercise. Building for a weak model forced the
> discipline that makes the system reliable: deterministic routing,
> deterministic math, verified output, measured extraction. Those are all
> good even with a strong model.

**"Would a bigger model fix your problems?"** *(very likely follow-up)*

> Probably it would follow instructions better, yes. But I'd want to
> measure it rather than assume it, so I made the model selectable with an
> environment variable and pointed the eval harness at it. Then it's a
> number, not an argument.
>
> The thing I'd **not** do is remove the safety checks. They're cheap and
> deterministic. A bigger model makes them fire less often; it doesn't make
> them unnecessary. And the warning log tells me the actual rate.

**"How would you scale this?"**

Be honest about current limits — pretending otherwise is worse than
admitting it:

> Right now state is in memory, so it dies with the process, and the model
> is loaded in-process, so it's one machine. To scale, I'd swap the
> checkpointer for a database-backed one — LangGraph supports that behind
> the same interface, so the graph code wouldn't change — and move
> inference behind a separate serving layer so the web tier could scale
> independently. The state machine itself doesn't need to change; that
> separation is the point of the layering.

**"How do you handle prompt injection / a malicious user?"**

Honest answer, and the honesty is the point:

> I haven't hardened against it, and I should say that clearly. The
> mitigating factor is the architecture: the model can't take actions. It
> can't call tools, spend money, or change control flow — it only fills in
> fields and writes text. So the blast radius is bad output, not bad
> behaviour. If I were putting this in production I'd add input length
> limits, output filtering, and treat extracted fields as untrusted data
> rather than as instructions.

**"What would you do next?"**

> Three things. Stream the output — right now you wait in silence while
> five sections generate, and that's the worst part of the experience.
> Grow the eval set and add generation-quality evals, because right now I
> only measure extraction, not whether the itinerary is factually right.
> And add tracing so I can see per-turn latency and token counts properly.

**"What's the weakest part?"**

> The generated content isn't fact-checked. The budget number is
> deterministic and I verify the text around it, but if the model claims a
> museum opens at 9am, nothing checks that. For a real product that's the
> next thing I'd fix — probably by grounding activities against a real
> places API rather than trusting the model's memory.

---

## 8. If they ask you to open the code

Have these four ready. Say what it does and *why* — the why is the point.

| File | What to say |
|---|---|
| `graph.py` — `build_graph()` | "This is the whole conversation in about 30 lines. Nodes and edges. You can read the entire flow here." |
| `graph.py` — `validate_preferences()` | "This is the routing decision. No model call. Pure Python. This is what makes it reliable." |
| `graph.py` — `_estimate_budget()` | "The math. And notice it **raises** instead of defaulting — let me explain why." |
| `evals/extraction_eval.py` | "The scorer. Four buckets, and hallucination is separate on purpose." |

**The `_estimate_budget` story is a good one:**

> This used to have defaults — if the budget level wasn't recognized, use
> "moderate." That looks safe and it's actually dangerous. Someone says
> "mid-range" and it doesn't match my table, so it silently costs the trip
> at the default rate and prints a confident, wrong number with nothing on
> screen to hint at it.
>
> Now the value is normalized to a fixed set when it's extracted, and if it
> still doesn't match, the field goes back to "missing" and we just ask
> again. And this function raises. By the time it runs, every input is
> required and validated — so an unexpected value is a bug in my pipeline,
> not user error, and it should be loud.

---

## 9. If they give you a live coding or design exercise

You won't be asked to rebuild this. But they may ask you to **extend** it.
Common ones, and how to think out loud:

**"Add hotel recommendations."**

> I'd add it as another generation node, not another responsibility for an
> existing one. New optional state field for the hotel preference, a new
> node after transportation, and an entry in the field-to-section map so an
> edit to budget or destination regenerates it. The graph makes this a
> small change, which is the payoff of the explicit structure.

**"The model is too slow. What do you do?"**

> First measure where the time goes — I already log per-call latency, and
> it's about 0.9 seconds for extraction and several seconds per generated
> section. Five sections in series is the bottleneck. Three options, in
> order of how much I'd trust them: stream so the user sees the first
> section immediately; run the independent sections in parallel, since
> activities and packing don't depend on each other; then look at a smaller
> or quantized model. I'd do the first two before touching the model.

**Say your reasoning out loud.** In these exercises the thinking is what's
being graded, not the answer.

---

## 10. Words to use, and words to avoid

**Use these** — accurate, and they sound senior:

- "The model never decides control flow."
- "Deterministic where it matters, generative where it helps."
- "A prompt is a request, not a guarantee."
- "Declining to answer is better than guessing."
- "I measured it" — the single most valuable sentence in an AI interview.

**Avoid these:**

- ❌ "It works perfectly." — Nothing does. Say what it does and doesn't do.
- ❌ "The AI figures it out." — Vague. Say exactly which call does what.
- ❌ "I used LangChain and LangGraph" as if the tools are the achievement.
  The design decisions are the achievement.
- ❌ Overclaiming the eval. It's 33 cases, one model, one machine. Say so.

---

## 11. Numbers to memorize

| Thing | Number |
|---|---|
| Field accuracy | **95.2%** (99 of 104) |
| Hallucination rate | **0%** (0 of 44 chances) |
| Wrong values | **0** |
| Fully correct cases | 28 of 33 |
| Mean extraction latency | ~0.9 s |
| Tests | 163, no GPU needed |
| Model | Llama 3.2, 1B params, local |
| Graph size | 15 nodes, 3 conditional edges |
| Improvement from the eval | 93.9% → 95.2%, hallucinations 2.3% → 0% |

If you only remember two: **95.2% and 0%**.

---
---

# PART B — NON-TECHNICAL

This part matters as much as Part A. Strong engineers lose offers here.

---

## 12. "Tell me about yourself"

Structure: **now → how you got here → why this role**. About 60–90
seconds. Do not narrate your whole CV.

Template — fill the brackets with your own truth:

> I'm currently [your role] at [company], and I'm finishing a Master's in
> AI at the University of the Cumberlands. My background is [X], and over
> the last [N] years I've moved toward [data / backend / ML] work.
>
> What pulled me toward AI engineering specifically is that I like the part
> where a model meets real engineering constraints — reliability, cost,
> testing, what happens when the model is wrong. My recent project is a
> good example: a travel planning assistant built on a small local model,
> where most of the work was designing so the system stays correct even
> when the model isn't.
>
> I'm looking for a role where I'm building AI systems that real users
> depend on, not just prototypes.

**Practise saying this out loud five times.** It should sound spoken, not
read. The first 20 seconds set the tone for the whole interview.

---

## 13. "Did you use AI to write this code?" ⚠️ read this carefully

You will likely be asked. Many candidates handle it badly — either they
deny it, or they get defensive. **This project was substantially built
with AI assistance, and pretending otherwise is both dishonest and easy to
catch** (they may ask you to explain any line, or change something live).

The good news: handled honestly, this is a **strength** in 2026. Using AI
tools well is part of the job now. What they're really testing is whether
you understand your own system.

**Script:**

> Yes, I used AI coding tools heavily, the same way I would on the job.
> What I'd point to is what I did with them. I made the architectural
> calls — keeping routing in Python rather than letting the model decide,
> computing the budget deterministically, treating a null extraction as a
> success rather than a failure. And I did the verification work: I built
> the eval set, I ran the app and found bugs the tests missed, and I
> rejected a few things the tool suggested because they were wrong.
>
> Concretely — the tool was happy to leave a default in the budget
> function. I took it out and made it raise, because a silent default there
> produces a confident wrong number. That's the kind of judgement I think
> the tools don't have yet.

**Then be ready to prove you understand it.** They may follow up with
"explain this function" or "what happens if I do X?" Re-read Part A §5
before the interview. If you can explain every design decision and why the
alternative was rejected, the AI question is completely defused.

**What NOT to say:**

- ❌ "No, I wrote it all myself." Risky and probably untrue.
- ❌ "AI wrote it, I just reviewed." Undersells you and sounds passive.
- ❌ Getting defensive. It's a normal question, not an accusation.

**One extra credit line if it fits:**

> I also documented it — there's an AI acknowledgement in the design doc.
> I think being explicit about that is just professional practice now.

---

## 14. The team story (this was a group project)

Be accurate: this is a **5-person group project** for MSAI-631, and you
were the one who owned the implementation. Do not claim you did everything
alone, and do not undersell what you actually did.

**"Tell me about working on a team."**

> It was a five-person team for a graduate course. I owned the
> implementation and the technical design; the others contributed to the
> proposal, the written design document, and the presentation.
>
> The hardest part wasn't the code — it was the gap between what the design
> document described and what the system actually did. The written design
> said the chatbot would ask for the user's name and could recommend
> destinations if you hadn't picked one. I built something different,
> because a fixed question-by-question flow tested badly and I moved to
> free-text extraction instead.
>
> So we had a document describing one system and a codebase doing another.
> I went through the document section by section against the code, listed
> every place they disagreed, and brought the list to the team rather than
> just changing things unilaterally. Some things we moved to "future work"
> because they genuinely weren't built; others I corrected in the doc.

**Why this story works:** it shows ownership, honesty, and that you don't
let documentation rot. It's also completely true.

**Fill in your own detail:** how did the team react? Did anyone disagree?
If someone pushed back and you changed your mind, that's an even better
version — say so.

---

## 15. "Tell me about a mistake"

Do **not** pick a fake weakness. Pick a real one with a real fix. You have
excellent true options:

**Option 1 — the silent default (best):**

> I wrote a budget calculation with a fallback: if the budget level wasn't
> recognized, use "moderate." I thought I was being defensive. I was
> actually hiding a bug. Someone types "mid-range," it doesn't match my
> list, and the system quotes a confident, completely wrong number with
> nothing to indicate anything went wrong.
>
> I found it later while reviewing the design document against the code. I
> fixed it in two places: the value is now normalized to a fixed set when
> it's extracted, and if it can't be, we ask the user again instead of
> guessing. And the calculation raises instead of defaulting.
>
> What I took from it is that "safe" defaults often aren't safe — they turn
> a loud failure into a quiet wrong answer, and quiet wrong answers are
> much worse in a system users trust.

**Option 2 — I documented something I hadn't verified:**

> While updating the design document, I wrote that the chatbot asks the
> user to clarify when a budget can't be understood. Then I checked the
> code and it did no such thing. I had described the behaviour I *intended*
> instead of the behaviour that existed.
>
> I corrected the document, and then fixed the code so the behaviour
> actually matched. But the lesson stuck: when I write documentation now, I
> check each claim against the code rather than from memory.

Both are honest, both have a concrete fix, both show self-correction. That
is exactly what the question is testing.

---

## 16. "Explain something technical to a non-technical person"

You have a real, verifiable story here — use it.

> Four of my five teammates weren't programmers, but they had to present
> this work and answer questions about it. So I wrote a separate document
> for them that explains the whole system with no code background assumed —
> what each file does, what the jargon means, and which small changes were
> safe for them to make versus which ones to ask me about first.
>
> The part I found hardest was explaining why the system sometimes
> *refuses* to fill in a detail. It looks like a weakness. So I framed it
> the way I'd explain it to a user: if the assistant guesses who you're
> travelling with and gets it wrong, that wrong guess quietly changes your
> whole plan and your budget. If it just asks, you lose five seconds. I
> found that once people had a concrete consequence, they got it
> immediately.

Point them at `docs/understanding_the_code.md` if they want to see it.

**Why this works:** most candidates answer this hypothetically. You have an
artifact.

---

## 17. Other behavioural questions — short scripts

**"How do you handle feedback / disagreement?"**

Use a real example. If someone reviewed your work and you disagreed, say
what you did:

> I try to separate "I think this is wrong" from "I don't like it." If
> someone tells me something is broken, my first move is to reproduce it,
> because being right about the diagnosis matters more than being fast. On
> this project a teammate said the assistant looked like it was ignoring
> them — I assumed it was a phrasing issue at first, but when I actually
> reproduced it, it turned out the plan was being generated and never
> displayed. They were completely right and my first instinct was wrong.

**"How do you prioritize when you can't do everything?"**

> I ask which failures the user actually feels. On this project I had a
> long list — no CI, no linting, no logging, no evals. I did the evals
> first, because "the model might be quietly wrong and I have no way to
> know" was the only item on that list that could make the product wrong
> rather than just make my life harder.

**"Tell me about a time you were under pressure / a deadline."**
*(Your story — fill in.)* Structure it: what was the deadline, what did you
cut, what did you protect, what was the outcome. The interesting part is
always **what you chose not to do**.

**"Why do you want to work here?"**
Research the company. Name something specific — a product, a problem they
have, a paper or blog post they published. Generic answers hurt you.

**"What are you looking for in a role?"**
Be honest and specific. "Systems real users depend on," "a team where code
review is serious," "mentorship from people stronger than me" are all
good, if true.

---

## 18. Questions to ask them

Ask 2–3. Good questions signal seniority. These are tailored to AI roles:

**About the work:**
- "How do you evaluate your models today? Is it offline evals, online
  metrics, human review, or a mix?"
- "When a model gives a wrong answer in production, how do you find out —
  does a user report it, or do you catch it?"
- "How much of the work is model-side versus the engineering around the
  model?"

**About the team:**
- "How do AI features get from idea to production here? Who signs off?"
- "What's the code review culture like?"
- "What does someone who's great in this role do differently from someone
  who's just okay?"

**About you:**
- "What would you want me to have accomplished in the first three months?"
- "What's the biggest gap on the team right now?"

**Avoid:** salary and vacation in a first technical round; anything
answered on their careers page.

---

## 19. Practical checklist

**The day before:**
- [ ] Run `pytest` and `python -m evals.extraction_eval` so the numbers are
      fresh in your head, and so nothing is broken if you screen-share.
- [ ] Re-read Part A §8 — be able to explain any of those four files.
- [ ] Say the 60-second pitch out loud 5 times.
- [ ] Have the repo open, plus the app already running if you can — the
      model takes a minute to load the first time.
- [ ] Prepare 3 questions from §18 for this specific company.

**During:**
- **If you don't know something, say so.** "I haven't worked with that —
  here's how I'd approach it" is a strong answer. Bluffing is the fastest
  way to lose a technical interviewer.
- **Answer the question asked**, then stop. Don't fill silence.
- **Think out loud** on design questions. The reasoning is being graded.
- **It's fine to pause.** "Let me think about that for a second" is fine.

**If you get stuck:** ask a clarifying question. It buys time and it's
what good engineers do anyway.

---

## 20. One-line summary if you only get one sentence

> I built a conversational travel planner on a 1B local model, and the
> engineering is all about not trusting that model: Python owns the routing
> and the math, the model only extracts and writes, and I have an eval set
> that proves it gets 95% of fields right with zero hallucinations.
