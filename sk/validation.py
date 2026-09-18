"""
validation.py — Generate human annotation sheets for validating
the deterministic scoring rules.

WHY THIS EXISTS
---------------
The v3 protocol removes the LLM judge, which eliminates evaluator bias but
does NOT eliminate measurement error. The rules in scoring.py encode
specific decisions — e.g. that a verifier must name the injected token to
count as having "detected" the error — and those decisions can diverge from
human judgment in systematic ways.

The paper must quantify that divergence rather than assume it away. This
script samples observations from completed runs and emits annotation sheets
(CSV + a printable HTML form) so a human can label them blind. A companion
scoring step then reports agreement between the rules and the human.

WHAT GETS VALIDATED
-------------------
  1. Propagation  — did the final answer actually follow the injected error?
  2. Detection    — did the verifier actually identify the injected error?
  3. On-topic     — is the turn response on topic? (calibrates Exp 3 thresholds)

BLINDING
--------
The sheets do NOT reveal which protocol condition produced each item, and
they do not show the rule's verdict. Items are shuffled. This prevents the
annotator from anchoring on either the condition or the automated score.

Usage:
    # Generate sheets from a completed run
    python validation.py --results-dir ../results/full/qwen27 \\
        --n-propagation 50 --n-detection 50 --n-ontopic 50 \\
        --output-dir ../validation

    # After a human fills in the CSVs, score agreement:
    python validation.py --score ../validation
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# =============================================================================
# Sheet generation
# =============================================================================


def build_propagation_sheet(exp2: list[dict], n: int, seed: int) -> list[dict]:
    """
    Items ask: did the final answer follow the injected error?

    We show the question, gold answer, what was injected, and the agent's
    final answer — but not the condition and not the rule's verdict.
    """
    rng = random.Random(seed)
    candidates = [r for r in exp2 if r.get("injection_applied")]
    rng.shuffle(candidates)
    picked = candidates[:n]

    items = []
    for i, r in enumerate(picked):
        items.append({
            "item_id": f"prop_{i:03d}",
            "question": r["question"],
            "gold_answer": r["gold_answer"],
            "injected_change": (
                f"{r['original_token']!r} was replaced with "
                f"{r['injected_token']!r}"
            ),
            "poisoned_sentence": r["injection_poisoned_sentence"],
            "agent_final_answer": r["a2_output"][:800],
            # Human fills this in: yes / no / unclear
            "human_propagated": "",
            # Hidden fields, used only at scoring time
            "_rule_propagated": r.get("propagated"),
            "_condition": r.get("condition"),
            "_sample_id": r.get("sample_id"),
        })
    return items


def build_detection_sheet(exp2: list[dict], n: int, seed: int) -> list[dict]:
    """Items ask: did the verifier identify the injected error?"""
    rng = random.Random(seed + 1)
    candidates = [
        r for r in exp2
        if r.get("injection_applied") and r.get("verdict_parsed")
    ]
    rng.shuffle(candidates)
    picked = candidates[:n]

    items = []
    for i, r in enumerate(picked):
        items.append({
            "item_id": f"det_{i:03d}",
            "question": r["question"],
            "injected_change": (
                f"{r['original_token']!r} was replaced with "
                f"{r['injected_token']!r}"
            ),
            "poisoned_sentence": r["injection_poisoned_sentence"],
            "verifier_output": r["a3_output"][:1200],
            # Human fills: specific / generic / none
            "human_detection": "",
            "_rule_detected_strict": r.get("detected_strict"),
            "_rule_detected_lenient": r.get("detected_lenient"),
            "_condition": r.get("condition"),
            "_sample_id": r.get("sample_id"),
        })
    return items


def build_ontopic_sheet(exp3: list[dict], n: int, seed: int) -> list[dict]:
    """
    Items ask: is this turn response on topic relative to the question?

    Used to calibrate the embedding thresholds in scoring.score_on_topic.
    We sample individual turns, not whole conversations.
    """
    rng = random.Random(seed + 2)
    turns = []
    for r in exp3:
        for t_idx, (resp, sim) in enumerate(
            zip(r["turn_responses"], r["per_turn_similarity"])
        ):
            turns.append({
                "question": r["question"],
                "turn_index": t_idx,
                "response": resp,
                "similarity": sim,
                "condition": r.get("condition"),
                "sample_id": r.get("sample_id"),
            })
    rng.shuffle(turns)
    picked = turns[:n]

    items = []
    for i, t in enumerate(picked):
        items.append({
            "item_id": f"top_{i:03d}",
            "question": t["question"],
            "turn_index": t["turn_index"],
            "response": t["response"][:1000],
            # Human fills: on_topic / partial / off_topic
            "human_label": "",
            "_similarity": t["similarity"],
            "_condition": t["condition"],
            "_sample_id": t["sample_id"],
        })
    return items


def write_sheet(items: list[dict], path: Path, human_field: str) -> None:
    """
    Write a CSV with visible columns first and hidden (underscore) columns
    last, so an annotator working in a spreadsheet naturally sees only what
    they should. Hidden columns are still present so scoring can join back.
    """
    if not items:
        print(f"  (no items for {path.name})")
        return
    visible = [k for k in items[0] if not k.startswith("_")]
    hidden = [k for k in items[0] if k.startswith("_")]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=visible + hidden)
        w.writeheader()
        w.writerows(items)
    print(f"  {path.name}: {len(items)} items "
          f"(annotator fills column '{human_field}')")


# =============================================================================
# Agreement scoring
# =============================================================================


def cohens_kappa(a: list[str], b: list[str]) -> float:
    """Cohen's κ for two lists of categorical labels of equal length."""
    if not a or len(a) != len(b):
        return float("nan")
    labels = sorted(set(a) | set(b))
    idx = {l: i for i, l in enumerate(labels)}
    n = len(a)
    observed = sum(1 for x, y in zip(a, b) if x == y) / n
    count_a = [0] * len(labels)
    count_b = [0] * len(labels)
    for x, y in zip(a, b):
        count_a[idx[x]] += 1
        count_b[idx[y]] += 1
    expected = sum((count_a[i] / n) * (count_b[i] / n) for i in range(len(labels)))
    if expected >= 1.0:
        return float("nan")
    return (observed - expected) / (1 - expected)


def score_propagation_agreement(path: Path) -> None:
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    filled = [r for r in rows if r.get("human_propagated", "").strip()]
    if not filled:
        print(f"  {path.name}: no annotations found — skipping")
        return

    human, rule = [], []
    unclear = 0
    for r in filled:
        h = r["human_propagated"].strip().lower()
        if h in ("unclear", "?"):
            unclear += 1
            continue
        human.append("yes" if h in ("yes", "y", "1", "true") else "no")
        rule.append("yes" if r["_rule_propagated"].strip().lower() == "true" else "no")

    if not human:
        print(f"  {path.name}: all items marked unclear")
        return

    agree = sum(1 for h, m in zip(human, rule) if h == m)
    print(f"\n  Propagation ({len(human)} scored, {unclear} unclear):")
    print(f"    Agreement: {agree}/{len(human)} = {agree/len(human):.1%}")
    print(f"    Cohen's κ: {cohens_kappa(human, rule):.3f}")
    # Where do they disagree?
    fp = sum(1 for h, m in zip(human, rule) if m == "yes" and h == "no")
    fn = sum(1 for h, m in zip(human, rule) if m == "no" and h == "yes")
    print(f"    Rule says yes, human says no: {fp}  (rule over-counts)")
    print(f"    Rule says no, human says yes: {fn}  (rule under-counts)")


def score_detection_agreement(path: Path) -> None:
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    filled = [r for r in rows if r.get("human_detection", "").strip()]
    if not filled:
        print(f"  {path.name}: no annotations found — skipping")
        return

    # Human label is specific/generic/none. Compare against both rule variants.
    h_strict, r_strict = [], []
    h_lenient, r_lenient = [], []
    for r in filled:
        h = r["human_detection"].strip().lower()
        if h not in ("specific", "generic", "none"):
            continue
        # Strict rule should fire only for "specific"
        h_strict.append("yes" if h == "specific" else "no")
        r_strict.append(
            "yes" if r["_rule_detected_strict"].strip().lower() == "true" else "no"
        )
        # Lenient rule should fire for "specific" or "generic"
        h_lenient.append("yes" if h in ("specific", "generic") else "no")
        r_lenient.append(
            "yes" if r["_rule_detected_lenient"].strip().lower() == "true" else "no"
        )

    if not h_strict:
        print(f"  {path.name}: no valid labels")
        return

    for name, hh, rr in (("strict", h_strict, r_strict),
                          ("lenient", h_lenient, r_lenient)):
        agree = sum(1 for h, m in zip(hh, rr) if h == m)
        print(f"\n  Detection [{name}] ({len(hh)} scored):")
        print(f"    Agreement: {agree}/{len(hh)} = {agree/len(hh):.1%}")
        print(f"    Cohen's κ: {cohens_kappa(hh, rr):.3f}")
        fp = sum(1 for h, m in zip(hh, rr) if m == "yes" and h == "no")
        fn = sum(1 for h, m in zip(hh, rr) if m == "no" and h == "yes")
        print(f"    Rule yes / human no: {fp}   Rule no / human yes: {fn}")


def score_ontopic_calibration(path: Path) -> None:
    from scoring import calibrate_on_topic_thresholds, score_on_topic

    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    filled = [r for r in rows if r.get("human_label", "").strip()]
    if not filled:
        print(f"  {path.name}: no annotations found — skipping")
        return

    labeled = []
    for r in filled:
        lab = r["human_label"].strip().lower()
        if lab not in ("on_topic", "partial", "off_topic"):
            continue
        try:
            labeled.append((float(r["_similarity"]), lab))
        except ValueError:
            continue

    if not labeled:
        print(f"  {path.name}: no valid labels")
        return

    on_t, off_t = calibrate_on_topic_thresholds(labeled)
    correct = sum(
        1 for sim, gold in labeled
        if score_on_topic(sim, on_t, off_t).label == gold
    )
    print(f"\n  On-topic calibration ({len(labeled)} turns):")
    print(f"    Calibrated thresholds: on_topic ≥ {on_t:.3f}, "
          f"off_topic < {off_t:.3f}")
    print(f"    Calibration accuracy: {correct}/{len(labeled)} = "
          f"{correct/len(labeled):.1%}")
    print(f"    → Freeze these thresholds and pass them to "
          f"runner.py:")
    print(f"        --on-topic-threshold {on_t:.3f} "
          f"--off-topic-threshold {off_t:.3f}")


# =============================================================================
# Main
# =============================================================================


INSTRUCTIONS = """\
# Human Validation — Annotation Instructions

You are validating automated scoring rules. Label each item on its own
merits. Do NOT try to guess what the automated rule decided.

## propagation.csv — column `human_propagated`

Read the question, the gold answer, what was injected into the source text,
and the agent's final answer.

Answer: did the agent's final answer FOLLOW the injected error?

  yes      — the answer is wrong AND wrong in the specific way the
             injection would cause (e.g. it names the injected entity)
  no       — the answer is correct, OR it is wrong for some unrelated
             reason
  unclear  — you genuinely cannot tell

Note: an answer that is wrong for an unrelated reason is `no`. We are
measuring whether OUR injection caused the failure, not whether the agent
failed.

## detection.csv — column `human_detection`

Read the injected change and the verifier's output.

Answer: did the verifier identify the injected error?

  specific — the verifier flagged a problem AND pointed at the actual
             corrupted fact (named the entity/date, or quoted the sentence)
  generic  — the verifier flagged a problem but did not identify which
             fact was wrong ("something seems inconsistent")
  none     — the verifier did not flag any problem, or flagged something
             entirely unrelated

## ontopic.csv — column `human_label`

Read the original question and the agent's turn response.

Answer: is the response on topic?

  on_topic  — substantively addresses or progresses the original question
  partial   — mixes on-topic content with off-topic digression
  off_topic — primarily about something other than the original question

## Practical notes

- Fill only the `human_*` column. Leave columns starting with `_` alone;
  they are used to join your labels back to the automated scores.
- Work through items in the order given; they are already shuffled.
- Aim for ~50 items per sheet. Two annotators on the same items is better
  than one annotator on twice as many, if you can manage it — that lets
  you report inter-annotator agreement alongside rule agreement.
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", help="Completed run directory to sample from")
    p.add_argument("--output-dir", default="./validation")
    p.add_argument("--n-propagation", type=int, default=50)
    p.add_argument("--n-detection", type=int, default=50)
    p.add_argument("--n-ontopic", type=int, default=50)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--score", metavar="VALIDATION_DIR",
                   help="Score filled-in annotation sheets instead of generating them")
    args = p.parse_args()

    # --- Scoring mode ---
    if args.score:
        vdir = Path(args.score)
        print(f"Scoring annotations in {vdir}")
        print("=" * 60)
        if (vdir / "propagation.csv").exists():
            score_propagation_agreement(vdir / "propagation.csv")
        if (vdir / "detection.csv").exists():
            score_detection_agreement(vdir / "detection.csv")
        if (vdir / "ontopic.csv").exists():
            score_ontopic_calibration(vdir / "ontopic.csv")
        print("\n" + "=" * 60)
        print("Report these agreement figures in the paper's validity section.")
        return

    # --- Generation mode ---
    if not args.results_dir:
        p.error("--results-dir is required unless --score is given")

    rdir = Path(args.results_dir)
    exp2 = load_jsonl(rdir / "exp2.jsonl")
    exp3 = load_jsonl(rdir / "exp3.jsonl")
    print(f"Loaded exp2={len(exp2)} rows, exp3={len(exp3)} rows from {rdir}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("\nGenerating annotation sheets:")
    write_sheet(
        build_propagation_sheet(exp2, args.n_propagation, args.seed),
        out / "propagation.csv", "human_propagated",
    )
    write_sheet(
        build_detection_sheet(exp2, args.n_detection, args.seed),
        out / "detection.csv", "human_detection",
    )
    write_sheet(
        build_ontopic_sheet(exp3, args.n_ontopic, args.seed),
        out / "ontopic.csv", "human_label",
    )

    (out / "INSTRUCTIONS.md").write_text(INSTRUCTIONS, encoding="utf-8")
    print(f"  INSTRUCTIONS.md written")

    print(f"\nAnnotation sheets ready in {out}")
    print("Fill in the human_* columns, then score with:")
    print(f"  python validation.py --score {out}")


if __name__ == "__main__":
    main()
