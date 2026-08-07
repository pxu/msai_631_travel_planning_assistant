"""Offline evaluation of preference extraction quality.

The unit tests in ``tests/`` use a fake chat model, so they verify graph
*logic* and deliberately say nothing about whether the real ~1B model can
actually read "my wife and son" as ``group_type="family"``. Everything in
``graph.py`` that exists to compensate for a small model — ``_is_grounded``,
``_infer_group_type``, ``_looks_like_specific_place``, the JSON truncation
repair, the budget/group normalizers — was previously unmeasured: there was
no number to say whether any of it helped.

This module supplies that number. It runs ``_merge_extracted`` over a fixed
set of utterances with known-correct fields and reports, per field:

* **hit**     — expected a value, got a matching one
* **miss**    — expected a value, got nothing
* **wrong**   — expected a value, got a different one
* **halluc.** — expected *nothing*, got a value

That last column is the one worth watching. A model that invents
``group_type="solo"`` because the sentence did not say who is travelling is
far more damaging than one that returns null: null triggers a follow-up
question, an invented value flows silently into the itinerary, the transport
advice and the budget multiplier.

Usage::

    python -m evals.extraction_eval                 # real local model
    python -m evals.extraction_eval --limit 5       # quick smoke run
    python -m evals.extraction_eval --tag grounding # one mechanism only
    python -m evals.extraction_eval --verbose       # per-case detail
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from travel_assistant.graph import _merge_extracted
from travel_assistant.state import ALL_PREFERENCE_FIELDS

CASES_PATH = Path(__file__).with_name("extraction_cases.json")

#: Fields scored by exact equality — closed vocabularies and integers, where
#: "close enough" is not a meaningful notion because they index into tables.
EXACT_FIELDS = frozenset({"trip_length_days", "group_type", "budget_level", "budget_total_usd"})

#: Free-text fields. A model writing "food and temples" where the case says
#: "food, temples" is correct; demanding string equality would measure
#: phrasing, not extraction. Scored as: does the extraction cover the
#: content words the case asked for?
_OVERLAP_THRESHOLD = 0.5

_TOKEN_STOPWORDS = frozenset(
    {"a", "an", "the", "and", "or", "of", "in", "on", "with", "to", "for", "some", "my", "our"}
)

SCORED_FIELDS = (*ALL_PREFERENCE_FIELDS, "budget_total_usd")


def _tokens(value: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", str(value).lower()) if t not in _TOKEN_STOPWORDS}


def matches(field_name: str, expected, actual) -> bool:
    """Is ``actual`` an acceptable extraction for ``expected``?"""
    if field_name in EXACT_FIELDS:
        return actual == expected
    exp_tokens, act_tokens = _tokens(expected), _tokens(actual)
    if not exp_tokens:
        return not act_tokens
    if field_name == "destination":
        # Either direction counts: "Kyoto, Japan" for "Kyoto" is right, and
        # so is "Kyoto" for "Kyoto, Japan".
        return bool(exp_tokens & act_tokens)
    return len(exp_tokens & act_tokens) / len(exp_tokens) >= _OVERLAP_THRESHOLD


# Outcome labels, ordered worst-first for reporting.
HALLUCINATION, WRONG, MISS, HIT, TRUE_NULL = (
    "hallucination",
    "wrong",
    "miss",
    "hit",
    "true_null",
)


def classify(field_name: str, expected, actual) -> str:
    if expected is None:
        return TRUE_NULL if actual is None else HALLUCINATION
    if actual is None:
        return MISS
    return HIT if matches(field_name, expected, actual) else WRONG


@dataclass
class CaseResult:
    case_id: str
    utterance: str
    tags: list[str]
    elapsed_s: float
    outcomes: dict[str, str] = field(default_factory=dict)
    extracted: dict = field(default_factory=dict)
    expected: dict = field(default_factory=dict)
    error: str | None = None

    @property
    def perfect(self) -> bool:
        """Every scored field either matched or was correctly left empty."""
        return self.error is None and all(
            outcome in (HIT, TRUE_NULL) for outcome in self.outcomes.values()
        )


def load_cases(path: Path = CASES_PATH, *, tag: str | None = None, limit: int | None = None):
    cases = json.loads(path.read_text(encoding="utf-8"))["cases"]
    if tag:
        cases = [c for c in cases if tag in c.get("tags", [])]
    return cases[:limit] if limit else cases


def run_case(llm, case: dict) -> CaseResult:
    started = time.perf_counter()
    result = CaseResult(
        case_id=case["id"],
        utterance=case["utterance"],
        tags=case.get("tags", []),
        elapsed_s=0.0,
        expected=case["expected"],
    )
    try:
        extracted = _merge_extracted(llm, {"last_user_input": case["utterance"]}, overwrite=False)
    except Exception as exc:  # a crash is a result, not a reason to stop the run
        result.elapsed_s = time.perf_counter() - started
        # Include the cause: graph.py wraps backend failures in
        # LLMInvocationError, whose message says nothing about *why* the
        # model died, which is the only thing worth reading in a failed run.
        detail = f"{type(exc).__name__}: {exc}"
        if exc.__cause__ is not None:
            detail += f" (caused by {type(exc.__cause__).__name__}: {exc.__cause__})"
        result.error = detail
        return result

    result.elapsed_s = time.perf_counter() - started
    result.extracted = extracted
    for name in SCORED_FIELDS:
        # A field the case does not mention is unscored: the set was written
        # incrementally and absent != expected-null.
        if name not in case["expected"]:
            continue
        result.outcomes[name] = classify(name, case["expected"][name], extracted.get(name))
    return result


def summarize(results: list[CaseResult]) -> dict:
    per_field: dict[str, Counter] = {}
    for r in results:
        for name, outcome in r.outcomes.items():
            per_field.setdefault(name, Counter())[outcome] += 1
    return {
        "cases": len(results),
        "perfect_cases": sum(r.perfect for r in results),
        "errors": sum(r.error is not None for r in results),
        "per_field": per_field,
        "total": Counter(o for r in results for o in r.outcomes.values()),
        "mean_latency_s": (sum(r.elapsed_s for r in results) / len(results)) if results else 0.0,
    }


def format_report(results: list[CaseResult], summary: dict) -> str:
    lines: list[str] = []
    scored = summary["total"]
    graded = sum(scored[k] for k in (HIT, WRONG, MISS))
    accuracy = scored[HIT] / graded if graded else 0.0
    null_opportunities = scored[TRUE_NULL] + scored[HALLUCINATION]
    halluc_rate = scored[HALLUCINATION] / null_opportunities if null_opportunities else 0.0

    lines.append("")
    lines.append(f"{'field':<24}{'hit':>5}{'miss':>6}{'wrong':>7}{'halluc':>8}{'ok-null':>9}")
    lines.append("-" * 59)
    for name in SCORED_FIELDS:
        counts = summary["per_field"].get(name)
        if not counts:
            continue
        lines.append(
            f"{name:<24}{counts[HIT]:>5}{counts[MISS]:>6}{counts[WRONG]:>7}"
            f"{counts[HALLUCINATION]:>8}{counts[TRUE_NULL]:>9}"
        )
    lines.append("-" * 59)
    lines.append(
        f"{'TOTAL':<24}{scored[HIT]:>5}{scored[MISS]:>6}{scored[WRONG]:>7}"
        f"{scored[HALLUCINATION]:>8}{scored[TRUE_NULL]:>9}"
    )
    lines.append("")
    lines.append(
        f"field accuracy (hit / expected-a-value) : {accuracy:6.1%}  ({scored[HIT]}/{graded})"
    )
    lines.append(
        f"hallucination rate (invented / should-be-null): {halluc_rate:6.1%}  "
        f"({scored[HALLUCINATION]}/{null_opportunities})"
    )
    lines.append(
        f"fully correct cases                          : "
        f"{summary['perfect_cases']}/{summary['cases']}"
    )
    lines.append(f"mean extraction latency                      : {summary['mean_latency_s']:.2f}s")
    if summary["errors"]:
        lines.append(f"cases that raised                            : {summary['errors']}")
    return "\n".join(lines)


def format_details(results: list[CaseResult]) -> str:
    lines: list[str] = []
    for r in results:
        if r.perfect:
            continue
        lines.append(f"\n[{r.case_id}] {'/'.join(r.tags)}  ({r.elapsed_s:.2f}s)")
        lines.append(f"  utterance: {r.utterance}")
        if r.error:
            lines.append(f"  ERROR: {r.error}")
            continue
        for name, outcome in r.outcomes.items():
            if outcome in (HIT, TRUE_NULL):
                continue
            lines.append(
                f"  {outcome:<14} {name:<22} "
                f"expected={r.expected[name]!r} got={r.extracted.get(name)!r}"
            )
    return "\n".join(lines) if lines else "\nAll cases fully correct."


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--limit", type=int, help="only run the first N cases")
    parser.add_argument("--tag", help="only run cases carrying this tag")
    parser.add_argument("--verbose", action="store_true", help="show every imperfect case")
    parser.add_argument("--json", action="store_true", help="emit machine-readable results")
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=None,
        help="exit non-zero if field accuracy falls below this (0-1), for CI gating",
    )
    args = parser.parse_args(argv)

    cases = load_cases(tag=args.tag, limit=args.limit)
    if not cases:
        print("no cases matched", file=sys.stderr)
        return 2

    # Imported lazily: loading the model takes far longer than everything
    # else here, and --help should not pay for it.
    from travel_assistant.llm import build_chat_model

    print(f"loading model and running {len(cases)} extraction cases...", file=sys.stderr)
    llm = build_chat_model()
    results = [run_case(llm, case) for case in cases]
    summary = summarize(results)

    if args.json:
        print(
            json.dumps(
                {
                    "summary": {
                        k: (
                            dict(v)
                            if isinstance(v, Counter)
                            else {f: dict(c) for f, c in v.items()}
                            if k == "per_field"
                            else v
                        )
                        for k, v in summary.items()
                    },
                    "results": [
                        {
                            "id": r.case_id,
                            "tags": r.tags,
                            "outcomes": r.outcomes,
                            "extracted": r.extracted,
                            "error": r.error,
                            "elapsed_s": r.elapsed_s,
                        }
                        for r in results
                    ],
                },
                indent=2,
            )
        )
    else:
        print(format_report(results, summary))
        if args.verbose:
            print(format_details(results))

    if args.min_accuracy is not None:
        total = summary["total"]
        graded = sum(total[k] for k in (HIT, WRONG, MISS))
        accuracy = total[HIT] / graded if graded else 0.0
        if accuracy < args.min_accuracy:
            print(
                f"\nFAIL: field accuracy {accuracy:.1%} < required {args.min_accuracy:.1%}",
                file=sys.stderr,
            )
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
