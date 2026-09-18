"""
compare_ladder.py — How cost saving and accuracy effect vary with model size.

Reads ladder directories (one per model, each holding turnN subdirectories)
and answers the two questions the ladder was run for.

**Does the cost saving track model size, or baseline verbosity?**

The 27B and 31B runs saved input tokens at different rates (0.15x vs 0.29x),
and the apparent reason was not capacity but how much the natural-language
baseline wrote: a terse baseline leaves less transcript for structure to avoid
re-sending. This script reports the saving alongside baseline output tokens
per turn, so the two explanations can be told apart. If saving tracks verbosity
rather than parameter count, that is the mechanism.

**Does structure start to matter for accuracy at the small end?**

At 27B and 31B it did not: removing the question barely moved EM even when
most 6-key SK turns emitted unparseable JSON, because the models recovered the
objective from the passages regardless. A smaller model may not manage that.
An accuracy gap that opens as size falls would mean structure earns its keep
precisely where models are weakest — a more useful claim than any of the
original three, and one the large-model results could not have shown.

Usage:
    python compare_ladder.py --root ../results/ladder
    python compare_ladder.py --root ../results/ladder --turns 96
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

from metrics import bootstrap_ci, mcnemar_paired

GRN = "\033[92m"
YEL = "\033[93m"
DIM = "\033[2m"
RST = "\033[0m"

TURN_RE = re.compile(r"turns(\d+)$")

# Parameter counts for ordering and for reading the size axis. Names are
# matched loosely because directory tags are derived from repo basenames.
SIZE_HINTS = [
    ("e2b", 2), ("e4b", 5), ("-4b", 4),
    ("-12b", 12), ("12b", 12),
    ("26b", 26), ("27b", 27),
    ("31b", 31), ("-31", 31),
]


def infer_size(tag: str) -> float:
    low = tag.lower()
    for hint, size in SIZE_HINTS:
        if hint in low:
            return float(size)
    return float("inf")   # unknown sizes sort last rather than crashing


def load_cells(root: Path) -> dict[str, dict[int, list[dict]]]:
    """model_tag -> {turns -> observations}"""
    out: dict[str, dict[int, list[dict]]] = {}
    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir():
            continue
        cells: dict[int, list[dict]] = {}
        for cell in sorted(model_dir.iterdir()):
            m = TURN_RE.search(cell.name)
            path = cell / "exp3.jsonl"
            if not m or not path.exists():
                continue
            rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
            if rows:
                cells[int(m.group(1))] = rows
        if cells:
            out[model_dir.name] = cells
    return out


def by_condition(rows: list[dict], field: str) -> dict[str, float]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        v = r.get(field)
        if v is not None:
            buckets[r["condition"]].append(float(v))
    return {c: sum(v) / len(v) for c, v in buckets.items() if v}


def paired(rows: list[dict], a: str, b: str, field: str):
    av = {r["sample_id"]: r.get(field) for r in rows if r["condition"] == a}
    bv = {r["sample_id"]: r.get(field) for r in rows if r["condition"] == b}
    common = sorted(s for s in set(av) & set(bv)
                    if av[s] is not None and bv[s] is not None)
    return [float(av[s]) for s in common], [float(bv[s]) for s in common]


def cost_section(models: dict, turns: int, sk: str, base: str) -> list[str]:
    L = [f"  Cost at {turns} turns — {sk} against {base}", ""]
    L.append(f"    {'model':<28} {'size':>5} {'SK in-tok':>11} {'base in-tok':>12} "
             f"{'ratio':>7} {'base out/turn':>14}")
    L.append("    " + "-" * 84)

    rows_out = []
    for tag in sorted(models, key=lambda t: infer_size(t)):
        cells = models[tag]
        if turns not in cells:
            continue
        rows = cells[turns]
        intok = by_condition(rows, "cumulative_in_tokens")
        outtok = by_condition(rows, "cumulative_out_tokens")
        if sk not in intok or base not in intok or not intok[base]:
            continue
        ratio = intok[sk] / intok[base]
        # Output tokens per turn for the baseline: the verbosity measure the
        # competing explanation turns on.
        verbosity = outtok.get(base, 0) / turns
        size = infer_size(tag)
        size_s = "?" if size == float("inf") else f"{size:.0f}B"
        L.append(f"    {tag[:28]:<28} {size_s:>5} {intok[sk]:>11,.0f} "
                 f"{intok[base]:>12,.0f} {ratio:>7.2f}x {verbosity:>14.1f}")
        rows_out.append((size, ratio, verbosity))

    if len(rows_out) >= 3:
        # Which explains the saving better: capacity, or how much the baseline
        # writes? Reported as correlations, with the caveat that three points
        # cannot settle it — the direction is what matters here.
        import statistics as st

        def corr(xs, ys):
            if len(set(xs)) < 2 or len(set(ys)) < 2:
                return float("nan")
            mx, my = st.mean(xs), st.mean(ys)
            num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
            den = (sum((x - mx) ** 2 for x in xs) *
                   sum((y - my) ** 2 for y in ys)) ** 0.5
            return num / den if den else float("nan")

        sizes = [r[0] for r in rows_out if r[0] != float("inf")]
        ratios = [r[1] for r in rows_out if r[0] != float("inf")]
        verbs = [r[2] for r in rows_out if r[0] != float("inf")]
        L.append("")
        L.append(f"    ratio vs model size:        r = {corr(sizes, ratios):+.3f}")
        L.append(f"    ratio vs baseline verbosity: r = {corr(verbs, ratios):+.3f}")
        L.append(f"    {DIM}A stronger correlation with verbosity than with size "
                 f"means the saving{RST}")
        L.append(f"    {DIM}comes from how much the baseline writes, not from "
                 f"capacity.{RST}")
        L.append(f"    {DIM}Three points cannot settle this; read the direction, "
                 f"not the value.{RST}")
    return L


def accuracy_section(models: dict, sk: str, base: str, alpha: float) -> list[str]:
    L = ["  Accuracy — does structure matter more as models shrink?", ""]
    turn_counts = sorted({t for cells in models.values() for t in cells})
    header = f"    {'model':<28} {'size':>5}"
    for t in turn_counts:
        header += f"  {'t=' + str(t):>16}"
    L.append(header)
    L.append("    " + "-" * (35 + 18 * len(turn_counts)))

    for tag in sorted(models, key=lambda t: infer_size(t)):
        cells = models[tag]
        size = infer_size(tag)
        size_s = "?" if size == float("inf") else f"{size:.0f}B"
        line = f"    {tag[:28]:<28} {size_s:>5}"
        for t in turn_counts:
            if t not in cells:
                line += f"  {'—':>16}"
                continue
            rows = cells[t]
            a, b = paired(rows, sk, base, "final_em")
            if not a:
                line += f"  {'—':>16}"
                continue
            diff = sum(a) / len(a) - sum(b) / len(b)
            try:
                p = mcnemar_paired([(bool(x), bool(y)) for x, y in zip(a, b)])["pvalue"]
            except Exception:
                p = float("nan")
            mark = "*" if p == p and p < alpha else " "
            line += f"  {diff:>+9.3f} {mark}{p:>5.2f}"
        L.append(line)

    L.append("")
    L.append(f"    {DIM}Each cell is EM({sk}) − EM({base}), with the McNemar "
             f"p-value; * marks p < {alpha}.{RST}")
    L.append(f"    {DIM}Positive values growing as size falls would mean "
             f"structure earns its keep{RST}")
    L.append(f"    {DIM}where models are weakest. Values near zero throughout "
             f"mean it does not.{RST}")
    return L


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="Ladder results directory")
    p.add_argument("--sk-condition", default="SK-index")
    p.add_argument("--baseline", default="NL-anchor")
    p.add_argument("--turns", type=int, default=96,
                   help="Turn count for the cost table")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"ERROR: {root} does not exist", file=sys.stderr)
        sys.exit(2)

    models = load_cells(root)
    if not models:
        print(f"ERROR: no ladder cells under {root}", file=sys.stderr)
        print("Expected <model>/turns<N>/exp3.jsonl", file=sys.stderr)
        sys.exit(2)

    L: list[str] = []
    L.append("=" * 86)
    L.append("Scale ladder — cost saving and accuracy effect versus model size")
    L.append("=" * 86)
    L.append("")
    L.append(f"Source: {root}")
    L.append(f"Models: {len(models)}  ({', '.join(sorted(models, key=infer_size))})")
    L.append("")
    L.extend(cost_section(models, args.turns, args.sk_condition, args.baseline))
    L.append("")
    L.append("-" * 86)
    L.extend(accuracy_section(models, args.sk_condition, args.baseline, args.alpha))
    L.append("")

    if any(infer_size(t) == float("inf") for t in models):
        L.append(f"  {YEL}Some model sizes could not be inferred from their "
                 f"directory names.{RST}")
        L.append(f"  {DIM}Add a hint to SIZE_HINTS in this file to place them "
                 f"on the size axis.{RST}")
        L.append("")

    text = "\n".join(L)
    print(text)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"\nWritten to {args.output}")


if __name__ == "__main__":
    main()