"""
tables.py — Generate the paper's tables directly from the result files.

Numbers get transcribed into a manuscript once and then edited, re-run,
corrected, and edited again. Every hand-copied figure is a chance for the
paper to disagree with the data. This produces the tables from the JSONL
files themselves, in LaTeX or Markdown, so a re-run regenerates the
manuscript numbers rather than inviting a second round of copying.

Tables produced:

  cost       Input tokens at a given horizon, per model, with the baseline
             verbosity that explains the ordering. The paper's central table.
  accuracy   EM by turn count and condition. Mostly nulls, presented as such.
  removal    Question-removal arms side by side, with parse-failure rates.
  fields     Six fields versus three: tokens, F1, parse failures.

Usage:
    python tables.py --table cost     --ladder ../results/ladder
    python tables.py --table accuracy --sweep ../results/sweep --format latex
    python tables.py --table removal  --ablation ../results/question_ablation
    python tables.py --table fields   --sweep ../results/sweep
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

# Display names. Directory tags are derived from repo basenames and are not
# presentable; a paper table needs the model as a reader would cite it.
MODEL_NAMES = {
    "gemma-4-e4b-it_": ("Gemma 4 E4B", 5),
    "gemma-4-12b-it-qat-w4a16-ct_": ("Gemma 4 12B", 12),
    "qwen3.5-27b_": ("Qwen 3.5 27B", 27),
    "gemma-4-31b-it-qat-w4a16-ct_": ("Gemma 4 31B", 31),
}

CONDITION_NAMES = {
    "NL-append": r"NL-append",
    "NL-anchor": r"NL-anchor",
    "SK": r"SK (6-key)",
    "SK-cite": r"SK-cite",
    "SK-index": r"SK-index",
    "SK-minimal": r"SK-minimal",
}

TURN_RE = re.compile(r"turns(\d+)$")
CELL_RE = re.compile(r"win(?P<window>[0-9]+k)_turns(?P<turns>[0-9]+)$")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def display_model(tag: str) -> tuple[str, float]:
    if tag in MODEL_NAMES:
        return MODEL_NAMES[tag]
    return (tag, float("inf"))


def mean_by_condition(rows: list[dict], field: str) -> dict[str, float]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        v = r.get(field)
        if v is not None:
            buckets[r["condition"]].append(float(v))
    return {c: sum(v) / len(v) for c, v in buckets.items() if v}


def parse_failure_rate(rows: list[dict], condition: str) -> float | None:
    g = [r for r in rows if r["condition"] == condition]
    if not g:
        return None
    total = bad = 0
    for r in g:
        for t in r.get("turn_responses", []):
            total += 1
            text = t.strip()
            if text.startswith("```"):
                text = re.sub(r"^```[a-zA-Z]*\n", "", text)
                text = re.sub(r"\n```\s*$", "", text).strip()
            try:
                json.loads(text)
            except Exception:
                bad += 1
    return bad / total if total else None


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render(rows: list[list[str]], header: list[str], fmt: str,
           caption: str = "", label: str = "", align: str = "") -> str:
    if fmt == "markdown":
        out = ["| " + " | ".join(header) + " |"]
        out.append("|" + "|".join("---" for _ in header) + "|")
        for r in rows:
            out.append("| " + " | ".join(r) + " |")
        return "\n".join(out)

    # LaTeX, booktabs. Elsevier accepts booktabs; if the template objects,
    # swap the rules for \hline.
    align = align or ("l" + "r" * (len(header) - 1))
    out = [r"\begin{table}[t]", r"\centering"]
    if caption:
        out.append(rf"\caption{{{caption}}}")
    if label:
        out.append(rf"\label{{{label}}}")
    out.append(rf"\begin{{tabular}}{{{align}}}")
    out.append(r"\toprule")
    out.append(" & ".join(header) + r" \\")
    out.append(r"\midrule")
    for r in rows:
        out.append(" & ".join(r) + r" \\")
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    out.append(r"\end{table}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def table_cost(ladder_root: Path, turns: int, sk: str, base: str,
               fmt: str) -> str:
    """Input tokens and the baseline verbosity that explains the ordering."""
    entries = []
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
        name, size = display_model(model_dir.name)
        entries.append({
            "name": name, "size": size,
            "sk_in": intok[sk], "base_in": intok[base],
            "ratio": intok[sk] / intok[base],
            "verbosity": outtok.get(base, 0) / turns,
        })

    # Ordered by verbosity, because that is the relationship being shown.
    entries.sort(key=lambda e: e["verbosity"])

    header = ["Model", "Params", "Baseline out/turn",
              "SK-index in", "Baseline in", "Ratio"]
    body = [[
        e["name"],
        "—" if e["size"] == float("inf") else f"{e['size']:.0f}B",
        f"{e['verbosity']:.1f}",
        f"{e['sk_in']:,.0f}",
        f"{e['base_in']:,.0f}",
        f"{e['ratio']:.2f}$\\times$" if fmt == "latex" else f"{e['ratio']:.2f}×",
    ] for e in entries]

    caption = (
        f"Input-token cost at {turns} turns, 16K window. Rows are ordered by "
        f"how much the natural-language baseline writes per turn. The saving "
        f"follows that ordering exactly and does not follow parameter count."
    )
    return render(body, header, fmt, caption, "tab:cost")


def table_accuracy(sweep_root: Path, window: str, fmt: str) -> str:
    """EM by turn count and condition, for one model."""
    cells: dict[int, list[dict]] = {}
    for cell in sorted(sweep_root.iterdir()):
        m = CELL_RE.search(cell.name)
        if not m or m.group("window") != window:
            continue
        rows = load_jsonl(cell / "exp3.jsonl")
        if rows:
            cells[int(m.group("turns"))] = rows
    if not cells:
        return f"% no cells for window {window} under {sweep_root}"

    conditions: list[str] = []
    for rows in cells.values():
        for c in {r["condition"] for r in rows}:
            if c not in conditions:
                conditions.append(c)
    conditions.sort(key=lambda c: (c.startswith("SK"), c))

    header = ["Turns"] + [CONDITION_NAMES.get(c, c) for c in conditions]
    body = []
    for turns in sorted(cells):
        means = mean_by_condition(cells[turns], "final_em")
        best = max((means[c] for c in conditions if c in means), default=None)
        row = [str(turns)]
        for c in conditions:
            if c not in means:
                row.append("—")
            elif fmt == "latex" and best is not None and means[c] == best:
                row.append(rf"\textbf{{{means[c]:.3f}}}")
            else:
                row.append(f"{means[c]:.3f}")
        body.append(row)

    caption = (
        f"Exact match by turn count, {window} window. No pairwise difference "
        f"reaches significance in any cell (McNemar, $\\alpha=0.05$)."
        if fmt == "latex" else
        f"Exact match by turn count, {window} window. No pairwise difference "
        f"reaches significance in any cell."
    )
    return render(body, header, fmt, caption, f"tab:accuracy-{window}")


def table_removal(ablation_root: Path, fmt: str) -> str:
    """Question-removal arms with parse-failure rates."""
    body = []
    for model_dir in sorted(ablation_root.iterdir()):
        if not model_dir.is_dir():
            continue
        kept = load_jsonl(model_dir / "question_kept" / "exp3.jsonl")
        removed = load_jsonl(model_dir / "question_removed" / "exp3.jsonl")
        if not (kept and removed):
            continue
        name, _ = display_model(model_dir.name)
        k = mean_by_condition(kept, "final_em")
        r = mean_by_condition(removed, "final_em")
        for cond in sorted(set(k) & set(r), key=lambda c: (c.startswith("SK"), c)):
            mal = parse_failure_rate(removed, cond)
            body.append([
                name,
                CONDITION_NAMES.get(cond, cond),
                f"{k[cond]:.3f}",
                f"{r[cond]:.3f}",
                f"{r[cond] - k[cond]:+.3f}",
                f"{mal:.1%}".replace("%", r"\%" if fmt == "latex" else "%")
                if mal is not None and cond.startswith("SK") else "—",
            ])
            name = ""   # only label the first row of each model block

    header = ["Model", "Condition", "EM (kept)", "EM (removed)",
              r"$\Delta$" if fmt == "latex" else "Δ", "Malformed"]
    caption = (
        "Removing the question after the first turn. Accuracy is essentially "
        "unchanged even where the protocol's own state fails to parse on most "
        "turns, indicating the objective survives in residual context rather "
        "than in propagated state."
    )
    return render(body, header, fmt, caption, "tab:removal")


def table_fields(sweep_root: Path, turns: int, window: str, fmt: str) -> str:
    """Six fields versus three: cost, accuracy, and parse reliability."""
    target = None
    for cell in sorted(sweep_root.iterdir()):
        m = CELL_RE.search(cell.name)
        if m and m.group("window") == window and int(m.group("turns")) == turns:
            target = cell
            break
    if target is None:
        return f"% no {window}/{turns}-turn cell under {sweep_root}"

    rows = load_jsonl(target / "exp3.jsonl")
    if not rows:
        return "% no rows"

    intok = mean_by_condition(rows, "cumulative_in_tokens")
    outtok = mean_by_condition(rows, "cumulative_out_tokens")
    em = mean_by_condition(rows, "final_em")
    f1 = mean_by_condition(rows, "final_f1")

    body = []
    for cond in ("SK", "SK-cite", "SK-index"):
        if cond not in em:
            continue
        mal = parse_failure_rate(rows, cond)
        body.append([
            CONDITION_NAMES.get(cond, cond),
            "6" if cond == "SK" else "3",
            f"{outtok.get(cond, 0):,.0f}",
            f"{em[cond]:.3f}",
            f"{f1.get(cond, 0):.3f}",
            f"{mal:.1%}".replace("%", r"\%" if fmt == "latex" else "%")
            if mal is not None else "—",
        ])

    header = ["Condition", "Fields", "Output tokens", "EM", "F1", "Malformed"]
    caption = (
        f"Field count at {turns} turns. The three additional fields cost "
        f"tokens and parse reliability without an accuracy return."
    )
    return render(body, header, fmt, caption, "tab:fields")


# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--table", required=True,
                   choices=["cost", "accuracy", "removal", "fields"])
    p.add_argument("--format", default="latex", choices=["latex", "markdown"])
    p.add_argument("--ladder", default="../results/ladder")
    p.add_argument("--sweep", default=None,
                   help="One model's sweep directory, e.g. ../results/sweep/qwen3.5-27b_")
    p.add_argument("--ablation", default="../results/question_ablation")
    p.add_argument("--turns", type=int, default=96)
    p.add_argument("--window", default="16k")
    p.add_argument("--sk-condition", default="SK-index")
    p.add_argument("--baseline", default="NL-anchor")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    if args.table == "cost":
        out = table_cost(Path(args.ladder), args.turns,
                         args.sk_condition, args.baseline, args.format)
    elif args.table == "accuracy":
        if not args.sweep:
            p.error("--sweep is required for the accuracy table")
        out = table_accuracy(Path(args.sweep), args.window, args.format)
    elif args.table == "removal":
        out = table_removal(Path(args.ablation), args.format)
    else:
        if not args.sweep:
            p.error("--sweep is required for the fields table")
        out = table_fields(Path(args.sweep), args.turns, args.window, args.format)

    print(out)
    if args.output:
        Path(args.output).write_text(out + "\n", encoding="utf-8")
        print(f"\n% written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()