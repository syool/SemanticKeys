"""
judge_check.py — Validate the deterministic scoring rules against
an API judge. Online only.

PURPOSE
-------
The offline protocol scores propagation and detection by rule rather than by
LLM judgment (see docs/offline_protocol_v3.md). That removes evaluator bias,
but it raises a fair question: do the rules agree with what a careful reader
would conclude?

Human annotation answers that best, and remains the primary validation
(src/validation.py). This script provides a cheaper, larger-N
complement: when internet access is available, run a frontier LLM judge over
the same observations and report agreement between judge and rule.

WHAT THIS IS NOT
----------------
The judge's verdicts NEVER feed into reported metrics. Reported numbers come
from the deterministic rules alone, both offline and online. This script only
produces an agreement statistic for the paper's validity section, of the form:

    "On a sample of N=200 observations, our deterministic propagation rule
    agreed with an independent LLM judge in 94% of cases (κ = 0.87).
    Disagreements were concentrated in cases where the agent paraphrased
    rather than quoted the injected entity."

That sentence strengthens the offline results without making them depend on
the judge.

Usage:
    export ANTHROPIC_API_KEY=...
    python judge_check.py --results-dir ../results_online/pilot/qwen3.5-27b \\
        --n 200 --output validation_report.txt
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from metrics import (
    JUDGE_DETECTION_V1_SYSTEM,
    JUDGE_PROPAGATION_V1_SYSTEM,
    JudgeClient,
)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_judge(provider: str | None = None):
    """
    Pick a judge backend from whatever API key is present. Preference order
    puts the strongest structured-output models first, but any of the three
    is adequate for an agreement statistic.
    """
    from client import AnthropicClient, GoogleClient, OpenAICompatibleClient

    if provider == "anthropic" or (provider is None and os.environ.get("ANTHROPIC_API_KEY")):
        return AnthropicClient(model="claude-sonnet-4-5"), "claude-sonnet-4-5"
    if provider == "openai" or (provider is None and os.environ.get("OPENAI_API_KEY")):
        return OpenAICompatibleClient(
            base_url="https://api.openai.com/v1",
            model="gpt-4o",
            api_key=os.environ["OPENAI_API_KEY"],
        ), "gpt-4o"
    if provider == "google" or (provider is None and os.environ.get("GOOGLE_API_KEY")):
        return GoogleClient(model="gemini-2.5-pro"), "gemini-2.5-pro"
    raise RuntimeError(
        "No judge API key found. Set ANTHROPIC_API_KEY, OPENAI_API_KEY, "
        "or GOOGLE_API_KEY."
    )


def cohens_kappa(a: list[bool], b: list[bool]) -> float:
    if not a or len(a) != len(b):
        return float("nan")
    n = len(a)
    observed = sum(1 for x, y in zip(a, b) if x == y) / n
    pa_true, pb_true = sum(a) / n, sum(b) / n
    expected = pa_true * pb_true + (1 - pa_true) * (1 - pb_true)
    if expected >= 1.0:
        return float("nan")
    return (observed - expected) / (1 - expected)


def agreement_block(
    title: str,
    rule: list[bool],
    judge: list[bool],
    disagreement_examples: list[str],
) -> list[str]:
    """Format an agreement summary with a confusion breakdown."""
    n = len(rule)
    if n == 0:
        return [f"  {title}: no comparable observations"]

    agree = sum(1 for r, j in zip(rule, judge) if r == j)
    both_yes = sum(1 for r, j in zip(rule, judge) if r and j)
    both_no = sum(1 for r, j in zip(rule, judge) if not r and not j)
    rule_only = sum(1 for r, j in zip(rule, judge) if r and not j)
    judge_only = sum(1 for r, j in zip(rule, judge) if not r and j)

    lines = [
        f"  {title}",
        f"    N compared:      {n}",
        f"    Agreement:       {agree}/{n} = {agree/n:.1%}",
        f"    Cohen's κ:       {cohens_kappa(rule, judge):.3f}",
        f"    Both positive:   {both_yes}",
        f"    Both negative:   {both_no}",
        f"    Rule yes, judge no:  {rule_only}   (rule may over-count)",
        f"    Rule no, judge yes:  {judge_only}   (rule may under-count)",
    ]
    if disagreement_examples:
        lines.append("    Sample disagreements:")
        for ex in disagreement_examples[:3]:
            lines.append(f"      - {ex}")
    return lines


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", required=True)
    p.add_argument("--n", type=int, default=200,
                   help="Max observations to sample per metric")
    p.add_argument("--provider", choices=["anthropic", "openai", "google"],
                   default=None, help="Judge provider (default: auto-detect from env)")
    p.add_argument("--output", default=None)
    p.add_argument("--seed", type=int, default=11)
    args = p.parse_args()

    rdir = Path(args.results_dir)
    exp2 = load_jsonl(rdir / "exp2.jsonl")
    if not exp2:
        print(f"No exp2.jsonl found in {rdir}; nothing to validate.", file=sys.stderr)
        sys.exit(1)

    raw_judge, judge_name = build_judge(args.provider)
    judge = JudgeClient(raw_judge)
    print(f"Judge: {judge_name}")

    rng = random.Random(args.seed)

    # ---------------------------------------------------------------- propagation
    prop_pool = [r for r in exp2 if r.get("injection_applied")]
    rng.shuffle(prop_pool)
    prop_sample = prop_pool[: args.n]

    prop_rule: list[bool] = []
    prop_judge: list[bool] = []
    prop_disagree: list[str] = []
    parse_failures = 0

    print(f"\nJudging propagation on {len(prop_sample)} observations...")
    for i, r in enumerate(prop_sample):
        if i % 25 == 0 and i:
            print(f"  {i}/{len(prop_sample)}")
        verdict = judge.judge(
            system=JUDGE_PROPAGATION_V1_SYSTEM,
            user=(
                f"QUESTION: {r['question']}\n"
                f"GOLD ANSWER: {r['gold_answer']}\n"
                f"INJECTED ERROR: {r['original_token']!r} was replaced with "
                f"{r['injected_token']!r} in: {r['injection_poisoned_sentence']}\n"
                f"FINAL ANSWER: {r['a2_output'][:1500]}"
            ),
            max_tokens=300,
        ).verdict
        if verdict.get("_parse_error"):
            parse_failures += 1
            continue
        j = bool(verdict.get("wrong_vs_gold")) and bool(verdict.get("matches_injected_error"))
        m = bool(r.get("propagated"))
        prop_rule.append(m)
        prop_judge.append(j)
        if m != j:
            prop_disagree.append(
                f"[{r['condition']}] rule={m} judge={j} | "
                f"injected {r['injected_token']!r} | "
                f"{str(verdict.get('reasoning', ''))[:110]}"
            )

    # ---------------------------------------------------------------- detection
    det_pool = [r for r in exp2
                if r.get("injection_applied") and r.get("verdict_parsed")]
    rng.shuffle(det_pool)
    det_sample = det_pool[: args.n]

    det_strict_rule: list[bool] = []
    det_lenient_rule: list[bool] = []
    det_judge_specific: list[bool] = []
    det_judge_any: list[bool] = []
    det_disagree: list[str] = []

    print(f"\nJudging detection on {len(det_sample)} observations...")
    for i, r in enumerate(det_sample):
        if i % 25 == 0 and i:
            print(f"  {i}/{len(det_sample)}")
        verdict = judge.judge(
            system=JUDGE_DETECTION_V1_SYSTEM,
            user=(
                f"QUESTION: {r['question']}\n"
                f"INJECTED ERROR: {r['original_token']!r} replaced with "
                f"{r['injected_token']!r} in: {r['injection_poisoned_sentence']}\n"
                f"VERIFIER OUTPUT: {r['a3_output'][:1500]}"
            ),
            max_tokens=300,
        ).verdict
        if verdict.get("_parse_error"):
            parse_failures += 1
            continue
        detected = bool(verdict.get("detected"))
        specificity = str(verdict.get("specificity", "")).lower()

        # The strict rule claims specific detection; compare against the
        # judge's "specific" label. The lenient rule claims any flag;
        # compare against the judge's boolean detected.
        det_strict_rule.append(bool(r.get("detected_strict")))
        det_judge_specific.append(detected and specificity == "specific")
        det_lenient_rule.append(bool(r.get("detected_lenient")))
        det_judge_any.append(detected)

        if bool(r.get("detected_strict")) != (detected and specificity == "specific"):
            det_disagree.append(
                f"[{r['condition']}] strict_rule={r.get('detected_strict')} "
                f"judge={specificity or 'none'} | "
                f"{str(verdict.get('reasoning', ''))[:110]}"
            )

    # ---------------------------------------------------------------- report
    L: list[str] = []
    L.append("=" * 72)
    L.append("Deterministic rule validation against an independent LLM judge")
    L.append("=" * 72)
    L.append("")
    L.append(f"Results directory: {rdir}")
    L.append(f"Judge model:       {judge_name}")
    L.append(f"Judge parse failures (excluded): {parse_failures}")
    L.append("")
    L.append("IMPORTANT: judge verdicts do NOT feed into any reported metric.")
    L.append("Reported numbers come from the deterministic rules alone. This")
    L.append("report exists only to quantify how closely the rules track")
    L.append("independent judgment, for the paper's validity section.")
    L.append("")
    L.append("-" * 72)
    L.extend(agreement_block(
        "Propagation  (rule vs judge)", prop_rule, prop_judge, prop_disagree))
    L.append("")
    L.extend(agreement_block(
        "Detection — strict  (rule vs judge 'specific')",
        det_strict_rule, det_judge_specific, det_disagree))
    L.append("")
    L.extend(agreement_block(
        "Detection — lenient  (rule vs judge 'detected')",
        det_lenient_rule, det_judge_any, []))
    L.append("")
    L.append("-" * 72)
    L.append("Interpretation guide:")
    L.append("  κ > 0.80  — strong agreement; rules are a faithful proxy")
    L.append("  κ 0.60-0.80 — substantial; report the gap and its direction")
    L.append("  κ < 0.60  — investigate before relying on the rule; check")
    L.append("              whether disagreements cluster in one condition,")
    L.append("              which would indicate a protocol-specific artifact")
    L.append("")
    L.append("If disagreements concentrate in a single condition, that is a")
    L.append("measurement problem, not a finding — the rule must not be")
    L.append("systematically harsher on one protocol than another.")

    # Per-condition disagreement breakdown, since that is the failure mode
    # that would actually invalidate the comparison.
    if prop_disagree or det_disagree:
        L.append("")
        L.append("-" * 72)
        L.append("Disagreements by condition (watch for concentration):")
        cond_counts = Counter()
        for line in prop_disagree + det_disagree:
            if line.startswith("["):
                cond_counts[line[1:line.index("]")]] += 1
        for cond, cnt in cond_counts.most_common():
            L.append(f"  {cond:<14} {cnt}")

    text = "\n".join(L)
    print("\n" + text)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"\nWritten to {args.output}")


if __name__ == "__main__":
    main()
