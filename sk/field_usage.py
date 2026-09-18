"""
field_usage.py — How many fields did the SK conditions actually emit?

The SK prompt lists fifteen allowed fields and instructs the model to include
only those relevant to the turn, omitting the rest. The manuscript has been
describing this condition as a six-field specification, which is what the
protocol design specifies but not what the prompt offers. Before either
number goes into a paper, the right question is what the models emitted.

Reports the distribution of field counts per condition and how often each
field name appears, so §3 can describe the condition as implemented and §6.5
can state the ablation contrast accurately.

Usage:
    python field_usage.py --results ../results/sweep/qwen3.5-27b_
    python field_usage.py --results ../results/ladder --recursive
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path


def parse_message(text: str) -> dict | None:
    """Return the parsed object, or None when the message is not valid JSON."""
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n", "", s)
        s = re.sub(r"\n```\s*$", "", s).strip()
    if not s.startswith("{"):
        return None
    try:
        obj = json.loads(s)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def collect(paths: list[Path]) -> dict[str, dict]:
    per_condition: dict[str, dict] = defaultdict(
        lambda: {"counts": [], "fields": Counter(), "messages": 0, "unparsed": 0}
    )
    for path in paths:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            cond = row.get("condition", "")
            if not cond.startswith("SK"):
                continue
            bucket = per_condition[cond]
            for msg in row.get("turn_responses", []):
                bucket["messages"] += 1
                obj = parse_message(msg)
                if obj is None:
                    bucket["unparsed"] += 1
                    continue
                bucket["counts"].append(len(obj))
                bucket["fields"].update(obj.keys())
    return per_condition


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--recursive", action="store_true",
                   help="Search subdirectories for exp3.jsonl")
    args = p.parse_args()

    root = Path(args.results)
    paths = sorted(root.rglob("exp3.jsonl")) if args.recursive \
        else sorted(root.glob("*/exp3.jsonl")) or sorted(root.glob("exp3.jsonl"))
    if not paths:
        raise SystemExit(f"no exp3.jsonl found under {root}")
    print(f"Reading {len(paths)} result files under {root}\n")

    data = collect(paths)
    if not data:
        raise SystemExit("no SK conditions found")

    print(f"{'condition':<12} {'parsed':>8} {'unparsed':>9} "
          f"{'mean':>6} {'median':>7} {'min':>4} {'max':>4}")
    print("-" * 60)
    for cond in sorted(data):
        b = data[cond]
        c = b["counts"]
        if not c:
            print(f"{cond:<12} {'0':>8} {b['unparsed']:>9}   (nothing parsed)")
            continue
        print(f"{cond:<12} {len(c):>8,} {b['unparsed']:>9,} "
              f"{st.mean(c):>6.2f} {st.median(c):>7.1f} {min(c):>4} {max(c):>4}")

    for cond in sorted(data):
        b = data[cond]
        if not b["counts"]:
            continue
        total = len(b["counts"])
        print(f"\n{cond} — field usage across {total:,} parsed messages")
        for name, n in b["fields"].most_common():
            bar = "#" * int(40 * n / total)
            print(f"  {name:<14} {n / total:>6.1%}  {bar}")

    print("\nReading this:")
    print("  The mean field count is what the condition actually is, whatever")
    print("  the prompt allows. Fields appearing in under a few percent of")
    print("  messages are offered but effectively unused, and describing the")
    print("  condition by the size of the menu rather than by what was ordered")
    print("  would overstate the contrast with the three-field variants.")


if __name__ == "__main__":
    main()