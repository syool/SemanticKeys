"""
analyze.py — Pre-registered hypothesis testing on saved experiment outputs.

This module is deliberately decoupled from the earlier judge-based runner so that
(1) stats can be rerun without repeating API calls, and (2) the analysis
plan is transparent and auditable on its own.

Hypotheses (from docs/methodology_v2.md §6):

  H1a  out_tokens(SK) < out_tokens(NL-free)             — Wilcoxon paired
  H1b  out_tokens(SK) < out_tokens(NL-concise)          — Wilcoxon paired
  H1c  F1(SK) ≥ F1(NL-free) within Δ=0.05               — TOST equivalence
  H2a  propagated(SK) < propagated(NL-free)             — McNemar paired
  H2b  detected(SK)   > detected(NL-free)               — McNemar paired
  H3a  drift(SK)      < drift(NL-anchor)                — Wilcoxon paired
  H3b  goal_em(SK)    > goal_em(NL-anchor)              — McNemar paired

All p-values are reported with Holm-Bonferroni correction across this
seven-test family. Point estimates and 95% bootstrap CI are reported for
every metric regardless of test outcome.

Usage:
    python analyze.py --results-dir ./results_v2 --output ./results_v2/analysis.txt
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import sys
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from metrics import (
    bootstrap_ci,
    holm_bonferroni,
    mcnemar_paired,
    paired_bootstrap_ci,
    wilcoxon_paired,
)


def load_jsonl(path: str | Path) -> list[dict]:
    out = []
    p = Path(path)
    if not p.exists():
        return out
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def pair_by_sample(
    rows: list[dict],
    condition_a: str,
    condition_b: str,
    metric: Callable[[dict], float | int | bool | None],
) -> tuple[list, list]:
    """
    Return paired (a_values, b_values) for samples that have both conditions
    and for which the metric function returns a non-None value in both.
    """
    a_by_sample: dict[str, float] = {}
    b_by_sample: dict[str, float] = {}
    for r in rows:
        if r["condition"] == condition_a:
            v = metric(r)
            if v is not None:
                a_by_sample[r["sample_id"]] = float(v)
        elif r["condition"] == condition_b:
            v = metric(r)
            if v is not None:
                b_by_sample[r["sample_id"]] = float(v)
    common = sorted(set(a_by_sample) & set(b_by_sample))
    return ([a_by_sample[s] for s in common], [b_by_sample[s] for s in common])


@dataclass
class HypothesisResult:
    name: str
    description: str
    n_pairs: int
    point_a: float
    point_b: float
    diff: float
    ci_low: float
    ci_high: float
    pvalue: float
    test: str
    # Extra fields populated depending on test type.
    b01: int | None = None
    b10: int | None = None
    median_a: float | None = None
    median_b: float | None = None


# Direction each hypothesis predicts for (A − B), where A is the SK condition.
# A two-sided test only says the two differ; it does not say which is larger.
# Without this mapping a result showing SK using MORE tokens prints as
# "REJECT H0" and reads like confirmation, which is the opposite of the truth.
EXPECTED_DIRECTION: dict[str, str] = {
    "H1a": "negative",   # SK should emit fewer tokens than NL-free
    "H1b": "negative",   # SK should emit fewer tokens than NL-concise
    "H1c": "positive",   # SK F1 should be no worse (non-inferiority)
    "H2a": "negative",   # SK should propagate errors less often
    "H2b": "positive",   # SK should detect errors more often
    "H3a": "negative",   # SK should drift less than NL-anchor
    "H3b": "positive",   # SK should maintain the goal more often
    # --- Ablation: does field count drive SK's verbosity? -------------------
    # Pre-specified in Amendment 1. Reported outside the H1a-H3b correction
    # family because they address a separate question about the protocol's
    # own specification rather than SK versus a baseline.
    "A1": "positive",    # full SK should use MORE tokens than SK-minimal
    "A2": "positive",    # full SK F1 should be at least as good
    # --- SK-v2 variants vs the six-key original and each other -------------
    "V1": "positive",    # six-key SK should use more tokens than SK-cite
    "V2": "positive",    # six-key SK should use more tokens than SK-index
    "V3": "positive",    # SK-cite should use more tokens than SK-index
    "V4": "negative",    # SK-index F1 should not fall below SK-cite
}

# Tested outside the Holm-Bonferroni family for H1a-H3b. Their p-values are
# uncorrected within their own two-test set.
SEPARATE_FAMILY = {"A1", "A2", "V1", "V2", "V3", "V4"}

# Designated in Amendment 1 as the result the paper is organised around if it
# holds. Marked in output so a reader can see which hypothesis carried the
# most prior weight, without that changing its test or threshold.
PRIMARY_INTEREST = {"H3b"}


def _direction_verdict(name: str, diff: float) -> tuple[str, bool]:
    """
    Return (label, matches_hypothesis) for an observed difference.

    A near-zero difference is reported as such rather than being forced into
    a direction, since sign is meaningless at that magnitude.
    """
    expected = EXPECTED_DIRECTION.get(name)
    if expected is None:
        return ("direction not specified", True)
    if abs(diff) < 1e-9:
        return ("no difference", False)
    observed = "positive" if diff > 0 else "negative"
    if observed == expected:
        return (f"as predicted ({observed})", True)
    return (f"OPPOSITE to hypothesis ({observed}, expected {expected})", False)


def _fmt(r: HypothesisResult, reject: bool) -> str:
    dir_label, dir_ok = _direction_verdict(r.name, r.diff)
    separate = r.name in SEPARATE_FAMILY

    # Significance and direction are separate facts and must be shown as such.
    # "Significant but backwards" is a real and important outcome; collapsing
    # it into a single REJECT/— column hides it.
    if reject and dir_ok:
        verdict = "SUPPORTED"
    elif reject and not dir_ok:
        verdict = "CONTRADICTED"
    else:
        verdict = "no evidence"
    if separate:
        verdict += "  [ablation — separate test family]"
    if r.name in PRIMARY_INTEREST:
        verdict += "  [primary hypothesis of interest]"

    if separate:
        sig_note = f"uncorrected, {'significant' if reject else 'not significant'}"
    else:
        sig_note = f"{'significant' if reject else 'not significant'} after correction"

    median_note = ""
    if r.median_a is not None and r.median_b is not None:
        median_note = (
            f"\n        A median = {r.median_a:.1f}  B median = {r.median_b:.1f}"
            f"  (medians disagree with means when outliers dominate)"
            if (r.median_a - r.median_b) * (r.point_a - r.point_b) < 0
            else f"\n        A median = {r.median_a:.1f}  B median = {r.median_b:.1f}"
        )

    base = (
        f"  {r.name}: {r.test:<13} n={r.n_pairs:<4} "
        f"{r.description}\n"
        f"        A mean = {r.point_a:.4f}  B mean = {r.point_b:.4f}  "
        f"diff = {r.diff:+.4f}  95%CI = [{r.ci_low:+.4f}, {r.ci_high:+.4f}]\n"
        f"        p = {r.pvalue:.4g}  ({sig_note})  "
        f"direction: {dir_label}"
        + median_note + "\n"
        f"        → {verdict}"
    )
    if r.test == "McNemar":
        base += f"\n        off-diagonals: b10(A+,B−)={r.b10}  b01(A−,B+)={r.b01}"
    return base


def run_hypotheses(
    exp1: list[dict],
    exp2: list[dict],
    exp3: list[dict],
    alpha: float = 0.05,
) -> list[tuple[HypothesisResult, bool]]:
    """
    Run all seven pre-registered hypothesis tests. Returns list of
    (HypothesisResult, reject_decision) after Holm-Bonferroni correction.
    Returns empty list if not enough data is available.
    """
    results: list[HypothesisResult] = []

    # --- H1a: out_tokens SK < NL-free (Wilcoxon) ---
    if exp1:
        a, b = pair_by_sample(exp1, "SK", "NL-free", lambda r: r["out_tokens"])
        if a:
            pt, lo, hi = paired_bootstrap_ci(a, b)  # SK - NL-free
            w = wilcoxon_paired(a, b)
            results.append(HypothesisResult(
                name="H1a",
                description="out_tokens(SK) vs out_tokens(NL-free)",
                n_pairs=len(a),
                point_a=sum(a) / len(a),
                point_b=sum(b) / len(b),
                diff=pt, ci_low=lo, ci_high=hi,
                pvalue=w["pvalue"], test="Wilcoxon",
            ))

    # --- H1b: out_tokens SK < NL-concise (Wilcoxon) ---
    if exp1:
        a, b = pair_by_sample(exp1, "SK", "NL-concise", lambda r: r["out_tokens"])
        if a:
            pt, lo, hi = paired_bootstrap_ci(a, b)
            w = wilcoxon_paired(a, b)
            results.append(HypothesisResult(
                name="H1b",
                description="out_tokens(SK) vs out_tokens(NL-concise)",
                n_pairs=len(a),
                point_a=sum(a) / len(a),
                point_b=sum(b) / len(b),
                diff=pt, ci_low=lo, ci_high=hi,
                pvalue=w["pvalue"], test="Wilcoxon",
            ))

    # --- H1c: F1 equivalence SK ≥ NL-free within Δ=0.05 (TOST) ---
    #   Implemented as a two-sided Wilcoxon on (F1_SK - F1_NL-free). Reject the
    #   non-equivalence null if the 90% CI of the diff lies within (-0.05, +∞).
    #   Full TOST (two one-sided tests) requires slightly more machinery;
    #   we report the p-value of a one-sided Wilcoxon testing SK ≥ NL-free − 0.05
    #   as a reasonable approximation, flagging this in the report.
    if exp1:
        a, b = pair_by_sample(exp1, "SK", "NL-free", lambda r: r.get("f1"))
        if a:
            pt, lo, hi = paired_bootstrap_ci(a, b)
            # Shift NL-free down by 0.05 to test the non-inferiority margin.
            b_shifted = [x - 0.05 for x in b]
            w = wilcoxon_paired(a, b_shifted)
            results.append(HypothesisResult(
                name="H1c",
                description="F1(SK) non-inferior to F1(NL-free) by Δ=0.05",
                n_pairs=len(a),
                point_a=sum(a) / len(a),
                point_b=sum(b) / len(b),
                diff=pt, ci_low=lo, ci_high=hi,
                pvalue=w["pvalue"], test="Wilcoxon-NI",
            ))

    # --- H2a: propagation SK < NL-free (McNemar) ---
    if exp2:
        a, b = pair_by_sample(
            exp2, "SK", "NL-free",
            lambda r: (1 if r["propagated"] else 0) if r.get("propagated") is not None else None,
        )
        if a:
            outcomes = [(bool(x), bool(y)) for x, y in zip(a, b)]
            m = mcnemar_paired(outcomes)
            diff = sum(a) / len(a) - sum(b) / len(b)
            lo_diff, hi_diff = paired_bootstrap_ci(a, b)[1:]
            results.append(HypothesisResult(
                name="H2a",
                description="propagation rate SK vs NL-free (lower is better)",
                n_pairs=len(a),
                point_a=sum(a) / len(a),
                point_b=sum(b) / len(b),
                diff=diff, ci_low=lo_diff, ci_high=hi_diff,
                pvalue=m["pvalue"], test="McNemar",
                b01=m["b01"], b10=m["b10"],
            ))

    # --- H2b: detection SK > NL-free (McNemar) ---
    if exp2:
        a, b = pair_by_sample(
            exp2, "SK", "NL-free",
            lambda r: (1 if r["detected"] else 0) if r.get("detected") is not None else None,
        )
        if a:
            outcomes = [(bool(x), bool(y)) for x, y in zip(a, b)]
            m = mcnemar_paired(outcomes)
            diff = sum(a) / len(a) - sum(b) / len(b)
            lo_diff, hi_diff = paired_bootstrap_ci(a, b)[1:]
            results.append(HypothesisResult(
                name="H2b",
                description="detection rate SK vs NL-free (higher is better)",
                n_pairs=len(a),
                point_a=sum(a) / len(a),
                point_b=sum(b) / len(b),
                diff=diff, ci_low=lo_diff, ci_high=hi_diff,
                pvalue=m["pvalue"], test="McNemar",
                b01=m["b01"], b10=m["b10"],
            ))

    # --- H3a: drift SK < NL-anchor (Wilcoxon) ---
    if exp3:
        a, b = pair_by_sample(exp3, "SK", "NL-anchor", lambda r: r["mean_drift"])
        if a:
            pt, lo, hi = paired_bootstrap_ci(a, b)
            w = wilcoxon_paired(a, b)
            results.append(HypothesisResult(
                name="H3a",
                description="mean drift SK vs NL-anchor (lower is better)",
                n_pairs=len(a),
                point_a=sum(a) / len(a),
                point_b=sum(b) / len(b),
                diff=pt, ci_low=lo, ci_high=hi,
                pvalue=w["pvalue"], test="Wilcoxon",
            ))

    # --- H3b: goal_em SK > NL-anchor (McNemar) ---
    if exp3:
        a, b = pair_by_sample(
            exp3, "SK", "NL-anchor",
            lambda r: r.get("final_em"),
        )
        if a:
            outcomes = [(bool(x), bool(y)) for x, y in zip(a, b)]
            m = mcnemar_paired(outcomes)
            diff = sum(a) / len(a) - sum(b) / len(b)
            lo_diff, hi_diff = paired_bootstrap_ci(a, b)[1:]
            results.append(HypothesisResult(
                name="H3b",
                description="goal-maintenance EM SK vs NL-anchor (higher is better)",
                n_pairs=len(a),
                point_a=sum(a) / len(a),
                point_b=sum(b) / len(b),
                diff=diff, ci_low=lo_diff, ci_high=hi_diff,
                pvalue=m["pvalue"], test="McNemar",
                b01=m["b01"], b10=m["b10"],
            ))

    # --- Ablation: six-field SK vs three-field SK-minimal -----------------
    # Pre-specified in preregistration Amendment 1 after the pilot showed the
    # extra three fields cost tokens with no accuracy return. Kept outside the
    # Holm-Bonferroni family for H1a–H3b: these ask a different question, and
    # folding them in would raise the correction threshold for the
    # confirmatory tests purely because we asked more.
    ablation: list[HypothesisResult] = []
    if exp1:
        a, b = pair_by_sample(exp1, "SK", "SK-minimal", lambda r: r["out_tokens"])
        if a:
            pt, lo, hi = paired_bootstrap_ci(a, b)
            w = wilcoxon_paired(a, b)
            ablation.append(HypothesisResult(
                name="A1",
                description="out_tokens(SK) vs out_tokens(SK-minimal)",
                n_pairs=len(a),
                point_a=sum(a) / len(a),
                point_b=sum(b) / len(b),
                diff=pt, ci_low=lo, ci_high=hi,
                pvalue=w["pvalue"], test="Wilcoxon",
            ))

        a, b = pair_by_sample(exp1, "SK", "SK-minimal", lambda r: r.get("f1"))
        if a:
            pt, lo, hi = paired_bootstrap_ci(a, b)
            w = wilcoxon_paired(a, b)
            ablation.append(HypothesisResult(
                name="A2",
                description="F1(SK) vs F1(SK-minimal)",
                n_pairs=len(a),
                point_a=sum(a) / len(a),
                point_b=sum(b) / len(b),
                diff=pt, ci_low=lo, ci_high=hi,
                pvalue=w["pvalue"], test="Wilcoxon",
            ))

    # --- SK-v2 variant comparisons ----------------------------------------
    # Which structured formulation costs least while holding accuracy. Median
    # matters as much as mean here: a pilot found one variant lower on the
    # median but higher on the mean, driven by a few long outliers.
    variant_pairs = [
        ("V1", "SK", "SK-cite", "out_tokens",
         "out_tokens(SK) vs out_tokens(SK-cite)"),
        ("V2", "SK", "SK-index", "out_tokens",
         "out_tokens(SK) vs out_tokens(SK-index)"),
        ("V3", "SK-cite", "SK-index", "out_tokens",
         "out_tokens(SK-cite) vs out_tokens(SK-index)"),
        ("V4", "SK-index", "SK-cite", "f1",
         "F1(SK-index) vs F1(SK-cite)"),
    ]
    for name, cond_a, cond_b, metric, desc in variant_pairs:
        if not exp1:
            break
        a, b = pair_by_sample(exp1, cond_a, cond_b, lambda r: r.get(metric))
        if not a:
            continue
        pt, lo, hi = paired_bootstrap_ci(a, b)
        w = wilcoxon_paired(a, b)
        import statistics
        ablation.append(HypothesisResult(
            name=name, description=desc, n_pairs=len(a),
            point_a=sum(a) / len(a), point_b=sum(b) / len(b),
            diff=pt, ci_low=lo, ci_high=hi,
            pvalue=w["pvalue"], test="Wilcoxon",
            median_a=statistics.median(a), median_b=statistics.median(b),
        ))

    if not results:
        # Ablation and variant tests alone can still be reported.
        return [(r, r.pvalue < alpha) for r in ablation]

    pvalues = [r.pvalue for r in results]
    decisions = holm_bonferroni(pvalues, alpha=alpha)
    # Ablation p-values are uncorrected: they sit outside the pre-registered
    # family, so folding them in would raise the correction threshold for the
    # confirmatory tests purely because we asked an extra question.
    return list(zip(results, decisions)) + \
           [(r, r.pvalue < alpha) for r in ablation]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=str, required=True)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--alpha", type=float, default=0.05)
    args = p.parse_args()

    exp1 = load_jsonl(f"{args.results_dir}/exp1.jsonl")
    exp2 = load_jsonl(f"{args.results_dir}/exp2.jsonl")
    exp3 = load_jsonl(f"{args.results_dir}/exp3.jsonl")

    print(f"Loaded Exp1: {len(exp1)} rows  Exp2: {len(exp2)} rows  Exp3: {len(exp3)} rows")

    tested = run_hypotheses(exp1, exp2, exp3, alpha=args.alpha)

    if not tested:
        msg = "No hypotheses could be tested — insufficient data."
        print(msg)
        if args.output:
            Path(args.output).write_text(msg, encoding="utf-8")
        return

    lines = [
        "=" * 72,
        f"Pre-registered hypothesis tests (Holm-Bonferroni, α = {args.alpha})",
        "=" * 72,
        "",
    ]
    for res, reject in tested:
        lines.append(_fmt(res, reject))
        lines.append("")

    out = "\n".join(lines)
    print(out)
    if args.output:
        Path(args.output).write_text(out, encoding="utf-8")


if __name__ == "__main__":
    main()