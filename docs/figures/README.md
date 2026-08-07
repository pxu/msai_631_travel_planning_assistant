# Corrected figures for the .docx design document

The two figures embedded in `project Design Doc final.docx` describe a
system that was never built — Figure 1 lists "User Name" as a collected
preference and "(or Destination Discovery)"; Figure 2 has a "Destination in
Mind?" decision diamond, a whole Scenario B destination-recommendation
column, and "Request User's Name" in Stage 1. The document text was
corrected for all of this; the figures are embedded PNGs and could not be.

These replacements match the implemented system and keep the original
visual style (plain black-on-white boxes, serif type, layer brackets).

| File | Replaces | Size |
|---|---|---|
| `figure1-system-architecture.png` | Figure 1. System Architecture | 1620 × 3197 |
| `figure2-user-interaction-flow.png` | Figure 2. User Interaction Flow | 1968 × 4958 |

Rendered at 3× so they stay sharp in print. Source: `../_build/figures.html`
— edit that and re-render if anything changes again.

## What changed, and why

**Figure 1**
- Removed `1. User Name` — there is no `user_name` field in `TravelState`.
- Removed `(or Destination Discovery)` — no such branch exists.
- Replaced the old 7-item list with the real schema: five required fields
  (destination, trip duration, travel group, travel interests, budget
  level) and three optional ones (travel style, travel season, must-visit
  attractions).
- Added the **Gap Analysis (plain Python)** box. This is the point of the
  architecture — routing is decided by code reading state, not by the
  model — and the original figure didn't show it at all.
- Labelled the budget calculator **(deterministic)** to match §5.1.

**Figure 2**
- Stage 1 no longer asks for a name; it invites a free-form trip
  description, which is what `collect_initial_request` actually does.
- Deleted the "Destination in Mind?" diamond and the entire Scenario B
  column (collect preferences → recommend Colorado/Banff/Yellowstone →
  user selects). None of it is implemented.
- Stage 2 is now the real loop: extract from free text → check what's
  missing → ask **only** for the missing fields → repeat.
- Added the pre-generation confirmation gate (`show_summary` →
  `confirm_preferences`), which the original figure omitted even though
  the code has it.
- Stage 4 now says **"Regenerate Only the Affected Section(s)"** rather
  than implying a full regeneration — this is the `_FIELD_TO_REGEN_NODES`
  behaviour and it's one of the better things to point at in a demo.

## Putting them into Word

1. Right-click the old figure → **Change Picture** → **From a File…** →
   pick the replacement. Doing it this way keeps the existing caption,
   numbering and cross-references intact. (Deleting and re-inserting
   breaks the "Figure 1"/"Figure 2" auto-numbering.)
2. If the image lands too large: drag a **corner** handle, not a side one,
   or set an exact width under **Picture Format → Size**. About 3.5 in
   wide suits Figure 1; Figure 2 is tall, so fit it to the page height and
   let the width follow.
3. Check the caption still reads "Figure 1. System Architecture of the
   AI-Powered Travel Assistant Planner" and "Figure 2. User Interaction
   Flow".

## Refreshing the Table of Contents

The TOC page numbers are stale — paragraphs were added and removed during
the text corrections.

1. `Ctrl` + `A` (Windows) or `Cmd` + `A` (Mac) to select the whole document.
2. Press **F9**. On a Mac laptop this is usually `fn` + `F9`.
3. When prompted, choose **Update entire table** — not "page numbers only",
   since section titles changed too (§10.2 gained two entries, and the
   "Table 1…" line was removed from the TOC).
4. Click once outside the TOC to deselect.

The stale "Table 1. System Inputs and Corresponding Outputs" entry has
already had its outline level cleared, so it will not reappear.
