"""
figures.py — Generate the paper's figures from the result files.

Same reasoning as tables.py: numbers that are re-plotted by hand after every
re-run eventually disagree with the data. These read the JSONL directly.

Figures:

  cost      Cumulative input tokens against turn count, per condition. The
            divergence between a protocol that re-sends its transcript and
            one that propagates a bounded tuple is the clearest single
            picture in the paper.
  verbosity Saving ratio against baseline output per turn, one point per
            model. Four points — plotted, not fitted. A regression line
            through four points would claim more than the data supports.
  scale     Accuracy delta against model size, with the high-powered
            confirmation point marked distinctly from the underpowered ones.

Output is PDF by default, which is what Elsevier wants; PNG is available for
drafts.

Usage:
    python figures.py --figure cost --sweep ../results/sweep/qwen3.5-27b_
    python figures.py --figure verbosity --ladder ../results/ladder
    python figures.py --figure scale --ladder ../results/ladder
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

CELL_RE = re.compile(r"win(?P<window>[0-9]+k)_turns(?P<turns>[0-9]+)$")
TURN_RE = re.compile(r"turns(\d+)$")

MODEL_NAMES = {
    "gemma-4-e4b-it_": ("Gemma 4 E4B", 5),
    "gemma-4-12b-it-qat-w4a16-ct_": ("Gemma 4 12B", 12),
    "qwen3.5-27b_": ("Qwen 3.5 27B", 27),
    "gemma-4-31b-it-qat-w4a16-ct_": ("Gemma 4 31B", 31),
}

# Distinct linestyles as well as colours: Neurocomputing is printed, and a
# figure that only works in colour is a figure that fails for some readers.
STYLE = {
    "NL-append": {"color": "#999999", "ls": ":",  "marker": "v"},
    "NL-anchor": {"color": "#1f77b4", "ls": "-",  "marker": "o"},
    "SK":        {"color": "#d62728", "ls": "--", "marker": "s"},
    "SK-cite":   {"color": "#ff7f0e", "ls": "-.", "marker": "^"},
    "SK-index":  {"color": "#2ca02c", "ls": "-",  "marker": "D"},
}

LABEL = {
    "NL-append": "NL-append",
    "NL-anchor": "NL-anchor",
    "SK": "SK (6-key)",
    "SK-cite": "SK-cite",
    "SK-index": "SK-index",
}


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def mean_by_condition(rows: list[dict], field: str) -> dict[str, float]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        v = r.get(field)
        if v is not None:
            buckets[r["condition"]].append(float(v))
    return {c: sum(v) / len(v) for c, v in buckets.items() if v}


def setup(figsize):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
    })
    return plt.subplots(figsize=figsize)


# ---------------------------------------------------------------------------


def fig_cost(sweep_root: Path, window: str, out: Path) -> None:
    """Cumulative input tokens against turn count."""
    cells: dict[int, list[dict]] = {}
    for cell in sorted(sweep_root.iterdir()):
        m = CELL_RE.search(cell.name)
        if not m or m.group("window") != window:
            continue
        rows = load_jsonl(cell / "exp3.jsonl")
        if rows:
            cells[int(m.group("turns"))] = rows
    if not cells:
        print(f"no cells for window {window} under {sweep_root}", file=sys.stderr)
        return

    turns = sorted(cells)
    series: dict[str, list[float]] = defaultdict(list)
    for t in turns:
        means = mean_by_condition(cells[t], "cumulative_in_tokens")
        for cond, v in means.items():
            series[cond].append(v)

    fig, ax = setup((3.4, 2.6))
    order = [c for c in ("NL-append", "NL-anchor", "SK", "SK-cite", "SK-index")
             if c in series]
    # The two NL conditions carry near-identical transcripts, so one plots
    # exactly on top of the other and vanishes. Widening the lower line lets
    # both remain visible rather than silently dropping a condition.
    for i, cond in enumerate(order):
        if len(series[cond]) != len(turns):
            continue
        s = STYLE.get(cond, {})
        overlapped = any(
            other != cond and len(series[other]) == len(turns) and
            all(abs(a - b) / max(a, b, 1) < 0.05
                for a, b in zip(series[cond], series[other]))
            for other in order
        )
        ax.plot(turns, series[cond], label=LABEL.get(cond, cond),
                color=s.get("color"), linestyle=s.get("ls"),
                marker=s.get("marker"), markersize=3.5,
                linewidth=2.6 if overlapped and i == 0 else 1.3,
                alpha=0.85 if overlapped else 1.0)

    ax.set_xlabel("Conversation turns")
    ax.set_ylabel("Cumulative input tokens")
    ax.set_yscale("log")   # the gap spans an order of magnitude by 96 turns
    ax.set_xticks(turns)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


def fig_verbosity(ladder_root: Path, turns: int, sk: str, base: str,
                  out: Path) -> None:
    """Saving ratio against baseline verbosity, one point per model."""
    points = []
    for model_dir in sorted(ladder_root.iterdir()):
        if not model_dir.is_dir():
            continue
        rows = load_jsonl(model_dir / f"turns{turns}" / "exp3.jsonl")
        if not rows:
            continue
        intok = mean_by_condition(rows, "cumulative_in_tokens")
        outtok = mean_by_condition(rows, "cumulative_out_tokens")
        if sk not in intok or base not in intok or not intok[base]:
            continue
        name, size = MODEL_NAMES.get(model_dir.name, (model_dir.name, None))
        points.append((outtok.get(base, 0) / turns, intok[sk] / intok[base],
                       name, size))
    if not points:
        print(f"no ladder cells at {turns} turns", file=sys.stderr)
        return

    fig, ax = setup((3.4, 2.6))
    # Place each label on whichever side has room, so nearby points do not
    # collide.
    xs = [p[0] for p in points]
    midpoint = (min(xs) + max(xs)) / 2
    for verb, ratio, name, size in points:
        ax.scatter(verb, ratio, s=42, color="#1f77b4", zorder=3)
        # Label above the marker rather than beside it. Side placement put
        # the 12B label across the neighbouring point when the two sat close
        # together on the x axis, which they do in the observed data.
        right = verb < midpoint
        ax.annotate(name, (verb, ratio), textcoords="offset points",
                    xytext=(2, 10) if right else (-2, 10),
                    ha="left" if right else "right", fontsize=7.5)

    ax.set_xlabel("Baseline output tokens per turn")
    # Two short lines rather than one long one: the single-line form was wider
    # than the axes and got clipped at this column width.
    ax.set_ylabel(f"Input-token ratio\n({sk} / {base})")
    ax.set_ylim(0, max(p[1] for p in points) * 1.25)
    # Headroom on the right so the rightmost label stays inside the figure.
    xs = [p[0] for p in points]
    ax.set_xlim(min(xs) - 0.1 * (max(xs) - min(xs)),
                max(xs) + 0.28 * (max(xs) - min(xs)))
    ax.grid(alpha=0.25, linewidth=0.5)

    # No fitted line. Four points cannot support a functional form, and
    # drawing one would invite a reviewer to ask for a confidence band that
    # the data cannot produce.
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


def fig_scale(ladder_root: Path, sk: str, base: str, out: Path,
              confirm: tuple[float, float] | None = None) -> None:
    """Accuracy delta against model size, averaged over turn counts.

    Taller than the other figures because the legend sits below the axes:
    placing it inside put its star marker among the data points, where it
    read as a fifth model rather than as a key.
    """
    points = []
    for model_dir in sorted(ladder_root.iterdir()):
        if not model_dir.is_dir():
            continue
        name, size = MODEL_NAMES.get(model_dir.name, (model_dir.name, None))
        if size is None:
            continue
        deltas = []
        for cell in sorted(model_dir.iterdir()):
            if not TURN_RE.search(cell.name):
                continue
            rows = load_jsonl(cell / "exp3.jsonl")
            if not rows:
                continue
            em = mean_by_condition(rows, "final_em")
            if sk in em and base in em:
                deltas.append(em[sk] - em[base])
        if deltas:
            points.append((size, sum(deltas) / len(deltas), name))
    if not points:
        print("no ladder cells", file=sys.stderr)
        return
    points.sort()

    fig, ax = setup((3.4, 3.0))
    sizes = [p[0] for p in points]
    deltas = [p[1] for p in points]
    ax.plot(sizes, deltas, color="#1f77b4", marker="o", markersize=5,
            linewidth=1.3, label="N = 40 (per turn count)")
    confirm_size = confirm[0] if confirm is not None else None
    for size, delta, name in points:
        # Keep the label clear of the line: above when the point is above
        # zero, below when it is under. Where the confirmation marker shares
        # a model, shift the label sideways so the two do not overlap.
        if confirm_size is not None and abs(size - confirm_size) < 1e-9:
            offset, ha = (13, 2), "left"
        else:
            offset, ha = (0, 9 if delta >= 0 else -15), "center"
        ax.annotate(name, (size, delta), textcoords="offset points",
                    xytext=offset, ha=ha, fontsize=7.5)

    if confirm is not None:
        # The confirmation run is a different N and a single turn count, so it
        # is marked separately rather than folded into the line — conflating
        # them would misrepresent the evidence behind each point.
        cs, cd = confirm
        ax.scatter([cs], [cd], s=70, color="#d62728", marker="*", zorder=4,
                   label="N = 400, 24 turns")

    ax.axhline(0, color="#666666", linewidth=0.8, linestyle=":")
    # Vertical headroom for the annotations, which otherwise clip against the
    # top spine and collide with the x tick labels at the bottom.
    span = max(deltas) - min(deltas)
    ax.set_ylim(min(deltas) - 0.45 * span, max(deltas) + 0.30 * span)
    # Linear, not log. A log axis put 5B and 12B almost on top of each other
    # while the minor ticks collided with the labels; with four points the
    # spacing costs nothing and the axis is easier to read.
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{s:g}B" for s in sizes])
    ax.set_xlim(min(sizes) - 3, max(sizes) + 5)
    ax.set_xlabel("Model parameters")
    ax.set_ylabel(f"EM({sk})\n$-$ EM({base})")
    ax.grid(alpha=0.25, linewidth=0.5)
    # Below the axes: inside the plot the legend's star sat among the data and
    # read as a fifth model.
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28),
              ncol=2, handletextpad=0.4, columnspacing=1.2)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--figure", required=True,
                   choices=["cost", "verbosity", "scale"])
    p.add_argument("--sweep", default=None)
    p.add_argument("--ladder", default="../results/ladder")
    p.add_argument("--window", default="16k")
    p.add_argument("--turns", type=int, default=96)
    p.add_argument("--sk-condition", default="SK-index")
    p.add_argument("--baseline", default="NL-anchor")
    p.add_argument("--confirm-size", type=float, default=None,
                   help="Parameter count of the high-powered confirmation run")
    p.add_argument("--confirm-delta", type=float, default=None,
                   help="Its accuracy delta, e.g. 0.095")
    p.add_argument("--out", default=None)
    p.add_argument("--format", default="pdf", choices=["pdf", "png"])
    args = p.parse_args()

    out = Path(args.out) if args.out else Path(f"fig_{args.figure}.{args.format}")
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.figure == "cost":
        if not args.sweep:
            p.error("--sweep is required for the cost figure")
        fig_cost(Path(args.sweep), args.window, out)
    elif args.figure == "verbosity":
        fig_verbosity(Path(args.ladder), args.turns,
                      args.sk_condition, args.baseline, out)
    else:
        confirm = None
        if args.confirm_size is not None and args.confirm_delta is not None:
            confirm = (args.confirm_size, args.confirm_delta)
        fig_scale(Path(args.ladder), args.sk_condition, args.baseline,
                  out, confirm)


if __name__ == "__main__":
    main()