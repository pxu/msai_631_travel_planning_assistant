# Review — `docs/project Design Doc final.docx`

Reviewed 2026-08-07 against the implemented system in `travel_assistant/`
and the scope recorded in `README.md` / `CLAUDE.md` / `docs/design_document.md`.

---

> ## STATUS — documentation-only fixes applied 2026-08-07
>
> All of §A (A1–A7 and the three smaller drifts), plus C3 and C4, have been
> applied in place to `docs/project Design Doc final.docx`. The pre-edit file
> is preserved as **`project Design Doc final.BACKUP-20260807.docx`**.
>
> ### Correction to the A4 fix (made 2026-08-07, same day)
>
> The A4 replacement text originally written into §7.3 was itself wrong. It
> claimed the chatbot "asks the user to clarify which level best describes
> the trip" when a budget can't be mapped to `budget`/`moderate`/`luxury`.
> **The code does no such thing.** `budget_level` is a free-text string;
> `validate_preferences` (`graph.py:363`) only checks truthiness, and
> `_estimate_budget` (`graph.py:554`) does `.get(key, 175)` — so `"$4000"`,
> `"mid-range"` and `"zzz"` all silently cost out at the *moderate* rate:
>
> ```
> $4000 -> $4900   moderate -> $4900   mid-range -> $4900   zzz -> $4900
> ```
>
> §7.3 now reads: *"If a required travel detail cannot be interpreted from
> the user's message, the chatbot re-requests only that detail rather than
> restarting the conversation."* — which the extraction-attempt fallback
> (`_MAX_EXTRACTION_ATTEMPTS` → `single_field_fallback`) actually does.
>
> The underlying silent-fallback behavior was a **real code bug**. It has
> since been fixed: `budget_level`/`group_type` are now closed vocabularies
> normalized at extraction time, an unmappable value returns to
> `missing_fields`, a stated total is captured in `budget_total_usd` and
> used verbatim, and `_estimate_budget` raises instead of defaulting. §7.2
> of the docx was updated to match ("If the traveler states a total
> outright, that figure is used as-is…").
>
> **Three things still need a human:**
>
> 1. **Both figures are unchanged and now contradict the text.** Figure 1
>    still lists "1. User Name" and "(or Destination Discovery)"; Figure 2
>    still has the "Destination in Mind?" diamond, the whole Scenario B
>    column, and "Request User's Name" in Stage 1. They are embedded PNGs
>    and must be redrawn. Figure 2's Stage 1 box should read "Welcome User /
>    Introduce Travel Assistant / Invite Free-Form Trip Description," and
>    the diamond and its right-hand column should be deleted so Stage 2 is a
>    single "Collect / Extract Travel Details → Ask Only for Missing Fields"
>    path.
> 2. **Refresh the TOC in Word** — Ctrl+A, then F9, then "update entire
>    table." Paragraphs were deleted and added, so page numbers are stale.
>    The stale "Table 1…" entry has already been removed from the TOC and
>    its outline level cleared, so it will not come back.
> 3. **C1, C2, C5, C6 were not touched** — see §C. C1 (a placeholder name
>    in the author list) needs the real roster; C2 (in-text citations) needs
>    the team's call on placement.
>
> Everything below is the original review, kept as the record of what was
> found and why.

## Assumption about "requirements"

**No course rubric, assignment brief, or grading criteria exists in this
repository.** `Group 4_Project Proposal.pdf` is referenced by
`docs/design_document.md:9` but is not present in the working tree either.
So this review does **not** check the docx against a course rubric — it
checks the two things that are verifiable here:

1. **Internal completeness/consistency** of the document itself.
2. **Fidelity to the system that was actually built.**

If a rubric or the proposal PDF exists elsewhere, re-run the completeness
check against it; findings in §C below are the ones most likely to be
rubric-graded.

---

## Verdict

The document is **structurally complete and well written** — all thirteen
sections are present, both figures are drawn and captioned, the layered
architecture and the design-decision rationale are solid, and §9/§10
correctly bound scope and future work.

It is **not accurate to the delivered system.** Six substantive claims
describe behavior that does not exist in the code, and one of them (the
"sequence of questions" framing) describes the *opposite* of the project's
defining architectural decision. These must be resolved before submission —
either by softening the doc to proposed/future framing, or by implementing
the features.

---

## A. Blocking — document describes a different system than the one built

| # | Doc claim | Where | Code reality |
|---|---|---|---|
| A1 | Chatbot asks the user's name and greets them by name (e.g. "Nice to meet you, <name>!"); "Personalization" listed as an HCI principle | §6.1, §6.2 Stage 1, §6.3, §11, Table 1 row 1, Figure 1 ("1. User Name"), Figure 2 ("Request User's Name") | No `user_name` field in `TravelState` (`state.py:26-56`). The graph starts at `collect_initial_request` (`graph.py:704`) and the first user message is treated as trip content, not a name. |
| A2 | Scenario B — destination-recommendation branch for undecided users; sample output "Colorado / Banff / Yellowstone" | §6.1, §6.2 Stage 2, §7.2, §9.3, Figure 2 (the "Destination in Mind?" diamond and its whole right-hand column) | No such node or edge exists in `build_graph()` (`graph.py:684-722`). `destination` is an ordinary required field that stays in `missing_fields` until stated. Confirmed as "Not built" in `docs/design_document.md` §9.3. |
| A3 | **"question sequencing"**, "responses are collected in sequence", "collected gradually through a sequence of simple questions" | §4 ¶2, §5.1 (Conversation Management Layer), §6.3 (Progressive Information Collection), §7.4 | This is the one design choice the project deliberately rejected. The system uses *adaptive extraction*: one LLM call parses all eight fields from free text, a pure-Python gap analysis picks what's still missing, and only that is asked for — in any order, any number per turn (`extract_preferences` → `validate_preferences` → `request_missing_fields`). §6.3's own "Intelligent Information Extraction" bullet describes the real behavior and **contradicts** the three bullets above it. |
| A4 | "If a user enters an invalid budget format, the chatbot prompts for a valid numerical value." | §7.3 | `budget_level` is categorical, not numeric — `_DAILY_RATE_BY_BUDGET = {"budget": 75, "moderate": 175, "luxury": 400}` (`graph.py:549`). There is no numeric budget entry and no such validation path. Wrong on both the data type and the behavior. |
| A5 | "Weather Preference (Optional)" as a system input | Table 1 row 8; Figure 2 Scenario B box | The three optional fields are `travel_style`, `travel_season`, `must_visit_attractions` (`state.py:69-73`). No weather preference is collected anywhere. |
| A6 | "An internet connection is available to communicate with the AI language model." | §9.1 Assumptions | The model runs **locally** — `unsloth/Llama-3.2-1B-Instruct` loaded through a local HuggingFace `transformers` pipeline (`llm.py:27-28`, `build_chat_model` at `llm.py:39-87`). No API call is made. Internet is needed only for the one-time weight download. |
| A7 | "The chatbot informs the user that the travel plan is being prepared and **provides status updates while the request is processed**"; "Continuous Feedback: …informs users whenever the travel plan is being generated or updated" | §6.2 Stage 3, §6.3 | `respond()` (`app.py:88`) is a plain `-> str` function: it calls `graph.invoke(...)` and returns one finished message. No `yield`, no `gr.Progress`, no streaming anywhere in `app.py`. With a 1B model running five sequential `generate_*` nodes, the user gets a long silent wait and then the whole plan at once — the most user-visible gap in this list, since a grader running the demo will notice it. |

**Three smaller drifts in the same category:**

- §7.2 / Table 1: "Estimated Travel Budget: An approximate budget calculated
  according to the user's **specified spending limit**." The estimate is
  `daily_rate(budget_level) × group_multiplier(group_type) × days`
  (`graph.py:553-560`) — derived from a categorical budget *level*, group
  size and duration, not from a limit the user names.
- Figure 1's "Collect User Preference" box lists seven items (User Name,
  Destination or Destination Discovery, Trip Duration, Travel Group, Travel
  Interests, Budget, Optional Must-Visit Attractions). The real field set is
  five required (`destination`, `trip_length_days`, `group_type`,
  `interests`, `budget_level`) plus three optional (`travel_style`,
  `travel_season`, `must_visit_attractions`).
- §5.1 AI Processing Layer lists four things the language model generates
  (itinerary, attractions/activities, transportation, packing) and omits the
  budget. The doc is right that the Business Logic layer owns the *number*
  (`_estimate_budget`, `graph.py:553-560`), but the LLM does narrate that
  figure into prose (`_budget_update` calls `_generate`, `graph.py:562-576`).
  Add budget narration to the §5.1 list — the hybrid split is a genuinely
  good design decision and currently goes unstated.

### Recommended resolution

Both figures need redrawing regardless of which path is chosen, because A1
and A2 are baked into them.

- **Fastest path (documentation-only):** delete Stage 1's name-collection
  interaction and Scenario B entirely; retitle §6.3's "Personalization"
  bullet to "Context Retention"; replace the three "sequence of questions"
  phrasings with the extraction wording already used in §6.3's
  "Intelligent Information Extraction" bullet; rewrite §7.3's budget
  example as a categorical-budget clarification; drop the Weather
  Preference row; fix the §9.1 internet assumption; drop or soften the two
  "status updates while processing" claims (A7). Move name
  personalization and destination recommendation into §10.2 Future
  Enhancements, where they read as deliberate scope decisions rather than
  as unmet claims.
- **Feature path:** implement `user_name` and a `recommend_destination`
  branch. `docs/design_document.md` §9.3 records why these were dropped
  (and why `vacation_type` and a destination-known boolean were rejected as
  redundant) — worth reading before reversing the decision.

**This is your call, not mine** — I have not edited the .docx.

---

## B. Two design documents now disagree

`CLAUDE.md` states the team's docx design document "has since been merged
into [`docs/design_document.md` §9] and deleted." But this file exists, is
named "final," and is dated after that note. The repo now carries two design
documents describing different systems:

- `docs/design_document.md` — matches the current code.
- `docs/project Design Doc final.docx` — describes the name/recommendation flow.

Decide which is the graded submission, and either delete or clearly label
the other. If the docx is the submission, `CLAUDE.md`'s "since deleted"
claim in the Design docs section is now stale and should be corrected.

---

## C. Academic / presentation issues

| # | Issue | Detail |
|---|---|---|
| C1 | **A stock placeholder name in the author list** | One entry on the title page is a well-known filler name rather than a real teammate. Verify the list against the real roster — highest-embarrassment, lowest-effort fix in the document. |
| C2 | **Zero in-text citations** | Seven references in §12 (Nielsen 1994; Norman 2013; ISO 9241-11:2018; Gradio; LangGraph; Llama 3; Python), none cited anywhere in the body. APA requires every reference to be cited in text. §6.3 "Conversation Design Principles" and §7.4 are exactly where Nielsen/Norman/ISO 9241-11 belong; §8.2–§8.5 are where the four tool-doc citations belong. Standard rubric deduction. |
| C3 | **Table 1 has no header row** | Ten data rows, three columns, no `Input \| Purpose \| Output` header. A reader has to infer the third column's meaning. |
| C4 | **Table 1 caption appears in the Table of Contents** | "Table 1. System Inputs and Corresponding Outputs" is styled with an outline level, so it renders in the TOC as a level-2 entry between §7.1 and §7.2, as if it were a subsection. Clear its outline level (set the caption paragraph to Body Text / Caption, outline level "Body Text"), then update the TOC field. |
| C5 | **Date reads "July 26, 2026"** | Confirm this is the intended final-submission date rather than a leftover from the draft. |
| C6 | **No List of Figures / List of Tables** | Two figures and one table are captioned but not indexed. Optional unless the rubric asks for it. |

Not an issue: the run-together text in Table 1 ("Refines / destination
recommendations", "Suitable / destination suggestions", "Attractions /
Included in itinerary") and on the title page is separate paragraphs and
`<w:br/>` line breaks inside cells, not missing spaces. It renders correctly
in Word.

---

## D. What is correct and complete — no action needed

- All 13 sections present and in a sensible order; TOC generated from real heading styles.
- §1–§4 (Introduction, Problem Statement, Objectives, System Overview) are complete and consistent with each other.
- §5 layered architecture and §8.6's five-layer breakdown match the real module split: `app.py` (presentation) / `graph.py` + `state.py` (conversation management) / `llm.py` (AI processing) / `_estimate_budget` + `format_summary` (business logic).
- §8.2–§8.5 technology-selection rationale (Gradio, LangGraph, Llama 3.2, Python) is sound and matches what was built.
- §9.2 Limitations and §9.3 Scope Boundary correctly state the out-of-scope list — no booking, no payments, no real-time data, no cross-session persistence — matching `README.md` and `CLAUDE.md`.
- §10 Future Scope is proportionate and consistent with the architecture.
- §6.2 Stage 4 "updates only the selected information while preserving the rest of the travel plan" is **accurate** — this is `apply_plan_edit` → `_FIELD_TO_REGEN_NODES` → `regenerate_affected` (`graph.py:611-683`). Good, and worth keeping prominent.
- §13 AI Acknowledgement is present and appropriately specific.

---

## Priority order

1. A3 (sequence-of-questions contradiction) — most substantive, easiest to miss because it reads as innocuous prose.
2. A1 + A2 + both figures — largest visible gap between doc and demo.
3. B — decide the canonical document.
4. A7 — a grader running the demo will hit the silent wait; either add a
   streaming/progress emission in `app.py` or drop the claim.
5. A4, A5, A6, and the three smaller drifts.
6. C1, C2 — quick, and C2 is likely graded.
7. C3, C4, C5, C6.
