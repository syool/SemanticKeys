"""
crossover.py — Locate the horizon at which structure starts paying off.

Reads a sweep directory produced by scripts/sweep.sh and answers the question
the revised design was built around: does structured communication overtake a
natural-language baseline as the conversation lengthens, and if so, where?

The crossover is defined in advance, in docs/revised_plan.md: the smallest
turn count at which SK's goal EM exceeds NL-anchor's under a paired test at
α=0.05. If no such point exists in the sweep, that is the finding — reported
plainly rather than softened into a trend.

Two things are plotted against turn count:

  benefit — goal EM and F1, whether the original task still gets done
  cost    — cumulative input and output tokens

Input tokens matter as much as output here, and are easy to overlook. NL
conditions re-send a growing transcript every turn while SK propagates a
bounded state tuple, so the input curves should diverge sharply even where
the output curves do not.

Usage:
    python crossover.py --root ../results/sweep/qwen3.5-27b
    python crossover.py --root ... --window 4k --output crossover.txt
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from metrics import mcnemar_paired

GRN = "\033[92m"
YEL = "\033[93m"
RED = "\033[91m"
DIM = "\033[2m"
RST = "\033[0m"

CELL_RE = re.compile(r"win(?P<window>[0-9]+k)_turns(?P<turns>[0-9]+)$")


def discover_cells(root: Path) -> dict[tuple[str, int], list[dict]]:
    """Map (window, turns) -> observations, from sweep subdirectories."""
    cells: dict[tuple[str, int], list[dict]] = {}
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        m = CELL_RE.search(sub.name)
        path = sub / "exp3.jsonl"
        if not m or not path.exists():
            continue
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        if rows:
            cells[(m.group("window"), int(m.group("turns")))] = rows
    return cells


def paired_by_sample(rows: list[dict], cond_a: str, cond_b: str, metric: str):
    """Return values for samples present in both conditions, in a stable order."""
    a_by = {r["sample_id"]: r.get(metric) for r in rows if r["condition"] == cond_a}
    b_by = {r["sample_id"]: r.get(metric) for r in rows if r["condition"] == cond_b}
    common = sorted(
        s for s in set(a_by) & set(b_by)
        if a_by[s] is not None and b_by[s] is not None
    )
    return [float(a_by[s]) for s in common], [float(b_by[s]) for s in common]


def condition_means(rows: list[dict], metric: str) -> dict[str, float]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        v = r.get(metric)
        if v is not None:
            buckets[r["condition"]].append(float(v))
    return {c: sum(v) / len(v) for c, v in buckets.items() if v}


def analyse_window(
    window: str,
    cells: dict[tuple[str, int], list[dict]],
    sk_condition: str,
    baseline: str,
    alpha: float,
) -> list[str]:
    """Build the report for one context-window condition."""
    turn_counts = sorted(t for (w, t) in cells if w == window)
    if not turn_counts:
        return [f"  no cells found for window {window}"]

    L: list[str] = []
    L.append(f"  Window: {window}   {sk_condition} vs {baseline}")
    L.append("")
    L.append(f"    {'turns':>5} {'SK EM':>8} {'base EM':>8} {'diff':>8} "
             f"{'p':>10} {'SK in-tok':>11} {'base in-tok':>12} {'dropped':>8}")
    L.append("    " + "-" * 78)

    crossover: int | None = None

    for turns in turn_counts:
        rows = cells[(window, turns)]
        a, b = paired_by_sample(rows, sk_condition, baseline, "final_em")
        if not a:
            L.append(f"    {turns:>5}  (no paired samples)")
            continue

        sk_em = sum(a) / len(a)
        base_em = sum(b) / len(b)
        diff = sk_em - base_em

        outcomes = [(bool(x), bool(y)) for x, y in zip(a, b)]
        try:
            p = mcnemar_paired(outcomes)["pvalue"]
        except ImportError as e:
            # A missing dependency must not read as "no significant difference".
            # Swallowing this turned a real crossover into an absent one during
            # development, because every p-value silently became nan.
            raise SystemExit(
                f"ERROR: cannot run the paired test — {e}\n"
                f"Install the analysis dependencies:\n"
                f"    pip install statsmodels scipy\n"
                f"Crossover detection depends on this test; without it the "
                f"report would understate the result."
            ) from e

        in_tok = condition_means(rows, "cumulative_in_tokens")
        sk_in = in_tok.get(sk_condition, 0.0)
        base_in = in_tok.get(baseline, 0.0)

        dropped = [
            r.get("turns_dropped", 0) for r in rows if r["condition"] == baseline
        ]
        n_dropped = sum(1 for d in dropped if d > 0)

        significant = p == p and p < alpha  # p == p excludes NaN
        if crossover is None and diff > 0 and significant:
            crossover = turns

        mark = ""
        if diff > 0 and significant:
            mark = f" {GRN}<-{RST}"
        elif diff > 0:
            mark = f" {DIM}+{RST}"

        L.append(
            f"    {turns:>5} {sk_em:>8.3f} {base_em:>8.3f} {diff:>+8.3f} "
            f"{p:>10.4g} {sk_in:>11,.0f} {base_in:>12,.0f} "
            f"{n_dropped:>4}/{len(dropped):<3}{mark}"
        )

    L.append("")
    if crossover is not None:
        L.append(f"    {GRN}Crossover at {crossover} turns{RST} — "
                 f"{sk_condition} exceeds {baseline} on goal EM, p < {alpha}")
    else:
        L.append(f"    {YEL}No crossover within {max(turn_counts)} turns.{RST}")
        L.append(f"    {DIM}This is a result, not a gap: within the horizons "
                 f"tested, structure did not overtake the baseline.{RST}")

    # Cost ratio at the longest horizon, where the divergence should be largest.
    longest = max(turn_counts)
    rows = cells[(window, longest)]
    in_tok = condition_means(rows, "cumulative_in_tokens")
    out_tok = condition_means(rows, "cumulative_out_tokens")
    if sk_condition in in_tok and baseline in in_tok and in_tok[baseline]:
        L.append("")
        L.append(f"    At {longest} turns:")
        L.append(f"      input tokens   {sk_condition}={in_tok[sk_condition]:>9,.0f}  "
                 f"{baseline}={in_tok[baseline]:>9,.0f}  "
                 f"ratio {in_tok[sk_condition]/in_tok[baseline]:.2f}x")
        if baseline in out_tok and out_tok[baseline]:
            L.append(f"      output tokens  {sk_condition}={out_tok[sk_condition]:>9,.0f}  "
                     f"{baseline}={out_tok[baseline]:>9,.0f}  "
                     f"ratio {out_tok[sk_condition]/out_tok[baseline]:.2f}x")
    return L


def variant_table(window: str, cells: dict) -> list[str]:
    """
    Goal EM for every SK variant against the baseline, at each turn count.

    The per-variant crossover analysis above answers "does this variant beat
    the baseline". This answers "which variant, and does the answer depend on
    horizon length" — which is the question that running SK-cite alongside
    SK-index exists to settle. If both three-field variants beat the six-field
    original, field count is what matters; if only SK-index does, the evidence
    representation is.
    """
    turn_counts = sorted(t for (w, t) in cells if w == window)
    if not turn_counts:
        return []

    present: set[str] = set()
    for turns in turn_counts:
        present |= {r["condition"] for r in cells[(window, turns)]}
    conditions = sorted(c for c in present if not c.startswith("SK")) + \
                 sorted(c for c in present if c.startswith("SK"))

    L = [f"  Goal EM by condition, window {window}", ""]
    L.append(f"    {'turns':>5}  " + "  ".join(f"{c:>12}" for c in conditions))
    L.append("    " + "-" * (7 + 14 * len(conditions)))
    for turns in turn_counts:
        rows = cells[(window, turns)]
        means = condition_means(rows, "final_em")
        best = max((means[c] for c in conditions if c in means), default=None)
        parts = []
        for c in conditions:
            if c not in means:
                parts.append(f"{'—':>12}")
            elif best is not None and means[c] == best:
                parts.append(f"{GRN}{means[c]:>12.3f}{RST}")
            else:
                parts.append(f"{means[c]:>12.3f}")
        L.append(f"    {turns:>5}  " + "  ".join(parts))
    L.append("")
    L.append(f"    {DIM}Best per row highlighted. Cumulative input tokens are "
             f"in the table above.{RST}")
    return L


def drift_table(window: str, cells: dict, alpha: float) -> list[str]:
    """Secondary metrics, reported but not used to define the crossover."""
    turn_counts = sorted(t for (w, t) in cells if w == window)
    if not turn_counts:
        return []

    # Read the conditions from the data. Hardcoding them meant a condition
    # added to a later sweep ran, was recorded, and then silently failed to
    # appear in this table.
    present: set[str] = set()
    for turns in turn_counts:
        present |= {r["condition"] for r in cells[(window, turns)]}
    # Baselines first, then SK variants, each group alphabetical.
    conditions = sorted(c for c in present if not c.startswith("SK")) + \
                 sorted(c for c in present if c.startswith("SK"))

    L = [f"  Secondary metrics, window {window}", ""]
    L.append(f"    {'turns':>5}  " + "  ".join(f"{c:>14}" for c in conditions))
    L.append("    " + "-" * (7 + 16 * len(conditions)))
    for turns in turn_counts:
        rows = cells[(window, turns)]
        means = condition_means(rows, "mean_drift")
        cells_txt = "  ".join(
            f"{means[c]:>14.3f}" if c in means else f"{'—':>14}"
            for c in conditions
        )
        L.append(f"    {turns:>5}  {cells_txt}")
    L.append("")
    L.append(f"    {DIM}Drift is cosine distance from the question — surface "
             f"proximity, not task success.{RST}")
    L.append(f"    {DIM}Goal EM above is the primary metric; these disagreed "
             f"in piloting.{RST}")
    return L


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="Sweep directory for one model")
    p.add_argument("--sk-condition", default="SK",
                   help="Which SK variant to test (SK, SK-index, ...)")
    p.add_argument("--baseline", default="NL-anchor")
    p.add_argument("--window", default=None, help="Restrict to one window")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"ERROR: {root} does not exist", file=sys.stderr)
        sys.exit(2)

    cells = discover_cells(root)
    if not cells:
        print(f"ERROR: no sweep cells found under {root}", file=sys.stderr)
        print("Expected subdirectories named win<W>_turns<N> containing exp3.jsonl",
              file=sys.stderr)
        sys.exit(2)

    windows = sorted({w for (w, _) in cells})
    if args.window:
        windows = [w for w in windows if w == args.window]

    L: list[str] = []
    L.append("=" * 82)
    L.append("Horizon sweep — cost and benefit versus conversation length")
    L.append("=" * 82)
    L.append("")
    L.append(f"Source: {root}")
    L.append(f"Cells:  {len(cells)}  ({', '.join(windows)})")
    L.append("")
    L.append("Crossover is pre-defined as the smallest turn count where the SK")
    L.append("condition's goal EM exceeds the baseline's, McNemar p < alpha.")
    L.append("")

    for w in windows:
        L.append("-" * 82)
        L.extend(analyse_window(w, cells, args.sk_condition, args.baseline, args.alpha))
        L.append("")
        L.extend(variant_table(w, cells))
        L.append("")
        L.extend(drift_table(w, cells, args.alpha))
        L.append("")

    text = "\n".join(L)
    print(text)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"\nWritten to {args.output}")


if __name__ == "__main__":
    main()