"""
compare.py — Cross-model side-by-side + pooled analysis.

Takes a results root directory containing per-model subdirectories
(e.g., results_full_multi/qwen27/, results_full_multi/gemma31/) and
produces:

  1. Side-by-side per-model summary tables (same metrics, both models)
  2. Cross-model agreement report — for each pre-registered hypothesis,
     does it replicate across both models?
  3. Pooled paired test — treating model as a random-effect-ish stratum,
     is the SK effect significant when we combine evidence across models?

The pooled test uses a simple stratified approach: we compute per-model
Wilcoxon/McNemar statistics, then combine p-values via Fisher's method
(rather than naively pooling data, which would inflate N and confound
model variance with sample variance). This is the standard approach
for "same hypothesis, different populations" meta-analysis.

Usage:
    python compare.py --root ./results_full_multi
    python compare.py --root ./results_full_multi \
        --models qwen27 gemma31 --output comparison.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from analyze import HypothesisResult, load_jsonl, run_hypotheses
from metrics import bootstrap_ci

try:
    from scipy.stats import combine_pvalues
except ImportError:
    combine_pvalues = None


def discover_model_dirs(root: Path) -> list[Path]:
    """Find subdirectories that contain exp{1,2,3}.jsonl."""
    out = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        if any((sub / f"exp{i}.jsonl").exists() for i in (1, 2, 3)):
            out.append(sub)
    return out


def load_model_results(model_dir: Path) -> tuple[list[dict], list[dict], list[dict]]:
    exp1 = load_jsonl(model_dir / "exp1.jsonl")
    exp2 = load_jsonl(model_dir / "exp2.jsonl")
    exp3 = load_jsonl(model_dir / "exp3.jsonl")
    return exp1, exp2, exp3


def per_condition_summary(rows: list[dict], metric: str) -> dict[str, tuple[float, float, float]]:
    """{condition: (mean, ci_lo, ci_hi)} with bootstrap CI."""
    buckets: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        v = r.get(metric)
        if v is None:
            continue
        try:
            buckets[r["condition"]].append(float(v))
        except (TypeError, ValueError):
            continue
    return {c: bootstrap_ci(vals) for c, vals in buckets.items()}


def build_side_by_side_table(
    per_model: dict[str, dict[str, tuple[float, float, float]]],
    metric_name: str,
) -> list[str]:
    """
    Build a side-by-side table for one metric across models.
    per_model is {model_name: {condition: (mean, lo, hi)}}.
    """
    lines = [f"  {metric_name}:"]
    # All conditions across all models
    conditions = sorted({c for m in per_model.values() for c in m})
    model_names = list(per_model.keys())

    # Header
    header = f"    {'condition':<14}"
    for m in model_names:
        header += f" | {m:<28}"
    lines.append(header)
    lines.append("    " + "-" * (14 + (31 * len(model_names))))

    for cond in conditions:
        row = f"    {cond:<14}"
        for m in model_names:
            if cond in per_model[m]:
                mn, lo, hi = per_model[m][cond]
                row += f" | {mn:.3f} [{lo:.3f},{hi:.3f}]   "
            else:
                row += f" | {'n/a':<28}"
        lines.append(row)
    return lines


def cross_model_hypothesis_check(
    per_model_tested: dict[str, list[tuple[HypothesisResult, bool]]],
    alpha: float = 0.05,
) -> list[str]:
    """
    For each hypothesis, check whether it's replicated across models.
    Returns a list of printable lines with a per-hypothesis verdict.
    """
    lines: list[str] = []
    # Collect per-model decisions keyed by hypothesis name
    hyp_keys: list[str] = []
    seen: set[str] = set()
    for model_name, results in per_model_tested.items():
        for res, _ in results:
            if res.name not in seen:
                seen.add(res.name)
                hyp_keys.append(res.name)

    lines.append(f"    {'hypothesis':<6} | " + " | ".join(
        f"{m:<18}" for m in per_model_tested
    ) + " | pooled p (Fisher)  | Replication")
    lines.append("    " + "-" * (8 + 21 * len(per_model_tested) + 22 + 15))

    for hyp_name in hyp_keys:
        row_parts = [f"    {hyp_name:<6}"]
        pvalues = []
        rejects = []
        for model_name, results in per_model_tested.items():
            found = None
            for res, reject in results:
                if res.name == hyp_name:
                    found = (res, reject)
                    break
            if found is None:
                row_parts.append(f"{'n/a':<18}")
                continue
            res, reject = found
            mark = "✓" if reject else "✗"
            row_parts.append(f"p={res.pvalue:.4g} {mark:<4}")
            pvalues.append(res.pvalue)
            rejects.append(reject)

        if len(pvalues) >= 2 and combine_pvalues is not None:
            # Fisher's combined test: appropriate when hypotheses are the
            # same but populations differ. Gives one pooled p-value.
            _, pooled_p = combine_pvalues(pvalues, method="fisher")
            row_parts.append(f"p={pooled_p:.4g}")
        else:
            row_parts.append("n/a")

        # Replication verdict
        if len(rejects) == 0:
            verdict = "n/a"
        elif all(rejects):
            verdict = "REPLICATED ✓"
        elif not any(rejects):
            verdict = "neither model"
        else:
            verdict = "PARTIAL (1/N)"
        row_parts.append(verdict)

        lines.append(" | ".join(row_parts))

    return lines


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, required=True,
                   help="Root directory containing per-model subdirectories.")
    p.add_argument("--models", nargs="+", default=None,
                   help="Optional: specific model subdirs to compare "
                        "(default: auto-discover).")
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--alpha", type=float, default=0.05)
    args = p.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"ERROR: root {root} does not exist", file=sys.stderr)
        sys.exit(2)

    if args.models:
        model_dirs = [root / m for m in args.models]
        for d in model_dirs:
            if not d.exists():
                print(f"ERROR: model dir {d} does not exist", file=sys.stderr)
                sys.exit(2)
    else:
        model_dirs = discover_model_dirs(root)
        if not model_dirs:
            print(f"ERROR: no model subdirs with exp*.jsonl found under {root}",
                  file=sys.stderr)
            sys.exit(2)

    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("Cross-model comparison — Semantic Keys v2 experiments")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Models included: {[d.name for d in model_dirs]}")
    lines.append("")

    # -------------------------------------------------------------------------
    # Per-model raw data
    # -------------------------------------------------------------------------
    per_model_data: dict[str, tuple[list[dict], list[dict], list[dict]]] = {}
    for d in model_dirs:
        per_model_data[d.name] = load_model_results(d)
        e1, e2, e3 = per_model_data[d.name]
        lines.append(f"  {d.name}: exp1={len(e1)}  exp2={len(e2)}  exp3={len(e3)}")
    lines.append("")

    # -------------------------------------------------------------------------
    # Side-by-side summaries by experiment
    # -------------------------------------------------------------------------
    lines.append("-" * 80)
    lines.append("[Experiment 1] Compression + reconstruction fidelity")
    lines.append("-" * 80)
    per_model_summary_e1 = {
        m: per_condition_summary(e[0], "out_tokens") for m, e in per_model_data.items()
    }
    lines.extend(build_side_by_side_table(per_model_summary_e1, "out_tokens (↓ better)"))
    per_model_summary_f1 = {
        m: per_condition_summary(e[0], "f1") for m, e in per_model_data.items()
    }
    lines.append("")
    lines.extend(build_side_by_side_table(per_model_summary_f1, "F1 (↑ better)"))
    lines.append("")

    lines.append("-" * 80)
    lines.append("[Experiment 2] Cascading error detection")
    lines.append("-" * 80)
    # Propagation / detection are booleans; coerce to 0/1 floats.
    for e2_rows in (e[1] for e in per_model_data.values()):
        for r in e2_rows:
            if r.get("propagated") is not None:
                r["_propagated_num"] = 1.0 if r["propagated"] else 0.0
            if r.get("detected") is not None:
                r["_detected_num"] = 1.0 if r["detected"] else 0.0
    per_model_summary_prop = {
        m: per_condition_summary(e[1], "_propagated_num")
        for m, e in per_model_data.items()
    }
    per_model_summary_det = {
        m: per_condition_summary(e[1], "_detected_num")
        for m, e in per_model_data.items()
    }
    lines.extend(build_side_by_side_table(per_model_summary_prop, "P-rate (↓ better)"))
    lines.append("")
    lines.extend(build_side_by_side_table(per_model_summary_det, "D-rate (↑ better)"))
    lines.append("")

    lines.append("-" * 80)
    lines.append("[Experiment 3] Context drift prevention")
    lines.append("-" * 80)
    per_model_summary_drift = {
        m: per_condition_summary(e[2], "mean_drift") for m, e in per_model_data.items()
    }
    per_model_summary_goal_em = {
        m: per_condition_summary(e[2], "final_em") for m, e in per_model_data.items()
    }
    lines.extend(build_side_by_side_table(per_model_summary_drift, "drift (↓ better)"))
    lines.append("")
    lines.extend(build_side_by_side_table(per_model_summary_goal_em, "goal EM (↑ better)"))
    lines.append("")

    # -------------------------------------------------------------------------
    # Cross-model hypothesis replication
    # -------------------------------------------------------------------------
    lines.append("=" * 80)
    lines.append("Hypothesis replication across models")
    lines.append("=" * 80)
    lines.append("")
    lines.append("  Interpretation:")
    lines.append("    ✓ REPLICATED      — both models reject H0 at α=0.05 (Holm-Bonferroni")
    lines.append("                        corrected within each model's 7-test family)")
    lines.append("    ✗ neither         — neither model rejects (SK effect not supported)")
    lines.append("    PARTIAL           — only one model rejects (partial replication)")
    lines.append("    Pooled p (Fisher) — combined evidence across models; for reporting")
    lines.append("    Note: a raw p < 0.05 may still show ✗ if Holm-Bonferroni stepdown")
    lines.append("    rejected it due to other, larger p-values in the same family.")
    lines.append("")

    per_model_tested: dict[str, list[tuple[HypothesisResult, bool]]] = {}
    for name, (e1, e2, e3) in per_model_data.items():
        per_model_tested[name] = run_hypotheses(e1, e2, e3, alpha=args.alpha)

    if combine_pvalues is None:
        lines.append("  NOTE: scipy.stats.combine_pvalues not available — pooled p omitted.")
        lines.append("")

    lines.extend(cross_model_hypothesis_check(per_model_tested, alpha=args.alpha))
    lines.append("")

    # -------------------------------------------------------------------------
    # Output
    # -------------------------------------------------------------------------
    lines.append("=" * 80)
    lines.append("Notes:")
    lines.append("  - Pooled p-values use Fisher's method on per-model p-values.")
    lines.append("    This is appropriate when the same hypothesis is tested in")
    lines.append("    different strata (here: different main models). It is NOT the same")
    lines.append("    as pooling all samples into a single paired test, which would")
    lines.append("    incorrectly conflate model variance with sample variance.")
    lines.append("  - For the paper, we recommend reporting:")
    lines.append("      (a) per-model point estimates with CIs (side-by-side tables above)")
    lines.append("      (b) per-model p-values with Holm-Bonferroni correction")
    lines.append("      (c) cross-model replication verdict (REPLICATED / PARTIAL / neither)")
    lines.append("      (d) Fisher pooled p as a secondary summary")
    lines.append("")

    text = "\n".join(lines)
    print(text)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"\nWritten to {args.output}")


if __name__ == "__main__":
    main()
