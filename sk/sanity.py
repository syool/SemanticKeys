"""
sanity.py — Post-pilot sanity checks.

Given a results directory produced by the earlier judge-based runner, this script
surfaces common failure modes before you trust the numbers:

  1. Judge parse-error rate (if > 15% on any condition, the judge prompt
     needs tightening before the full run).
  2. Injection failure rate (if > 25%, the injection pool is too narrow
     for the sample distribution — consider spaCy NER upgrade).
  3. Floor/ceiling effects (if any condition's metric is near 0 or 1 for
     every sample, the measurement is not discriminating between
     conditions; reviewer will call this out).
  4. Suspicious identical-response patterns (API caching or prompt bug).
  5. Per-sample heterogeneity — if variance within a condition is tiny
     but across conditions is large, suspicious; if the reverse,
     underpowered.

Output is a human-readable checklist. Exit code 0 if no red flags,
non-zero otherwise (useful for CI).

Usage:
    python sanity.py path/to/results_dir
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


RED = "\033[91m"
YEL = "\033[93m"
GRN = "\033[92m"
RST = "\033[0m"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def check_judge_parse_errors(rows: list[dict], experiment_name: str) -> list[str]:
    """Judge-dependent fields being None/empty often means parse failure."""
    flags: list[str] = []
    if not rows:
        return flags

    if experiment_name == "exp1":
        # Experiment 1 has no judge in v3: reconstruction is scored by rule
        # (HotpotQA EM/F1). The v2 check looked for a `recoverable` field that
        # v3 never writes, so it reported 100% failure on healthy runs and
        # buried the real flags. Accuracy problems here surface through the
        # floor/ceiling and truncation checks instead.
        return flags
    elif experiment_name == "exp2":
        by_cond: dict[str, list] = defaultdict(list)
        for r in rows:
            # Judge ran if injection applied; None then means parse error.
            if r.get("injection_applied"):
                by_cond[r["condition"]].append(r.get("propagated") is None)
        for cond, errs in by_cond.items():
            if not errs:
                continue
            rate = sum(errs) / len(errs)
            severity = RED if rate > 0.15 else (YEL if rate > 0.05 else GRN)
            flags.append(
                f"  {severity}exp2 propagation scoring incomplete [{cond}]: "
                f"{rate:.1%} ({sum(errs)}/{len(errs)}){RST}"
            )
    elif experiment_name == "exp3":
        # Count "parse_error" labels in on_topic_labels
        by_cond: dict[str, tuple[int, int]] = defaultdict(lambda: (0, 0))
        for r in rows:
            errs = sum(1 for l in r.get("on_topic_labels", []) if l == "parse_error")
            total = len(r.get("on_topic_labels", []))
            prev = by_cond[r["condition"]]
            by_cond[r["condition"]] = (prev[0] + errs, prev[1] + total)
        for cond, (errs, total) in by_cond.items():
            if total == 0:
                continue
            rate = errs / total
            severity = RED if rate > 0.15 else (YEL if rate > 0.05 else GRN)
            flags.append(
                f"  {severity}exp3 on-topic labels missing [{cond}]: "
                f"{rate:.1%} ({errs}/{total} turns){RST}"
            )
    return flags


def check_injection_coverage(exp2: list[dict]) -> list[str]:
    if not exp2:
        return []
    # Each sample × condition produces a row, but injection is per-sample.
    unique_samples = set()
    failed_samples = set()
    strat_counts: Counter = Counter()
    for r in exp2:
        sid = r["sample_id"]
        unique_samples.add(sid)
        if not r["injection_applied"]:
            failed_samples.add(sid)
        else:
            strat_counts[r["injection_strategy"]] += 1
    fail_rate = len(failed_samples) / len(unique_samples) if unique_samples else 0.0
    severity = RED if fail_rate > 0.25 else (YEL if fail_rate > 0.10 else GRN)
    out = [
        f"  {severity}Injection failure rate: {fail_rate:.1%} "
        f"({len(failed_samples)}/{len(unique_samples)} samples){RST}"
    ]
    total_applied = sum(strat_counts.values())
    if total_applied > 0:
        balance = strat_counts.most_common(1)[0][1] / total_applied
        sev = YEL if balance > 0.75 else GRN
        out.append(
            f"  {sev}Injection strategy balance: "
            f"{dict(strat_counts)} (max share {balance:.0%}, ideal ≈50%){RST}"
        )
    return out


def check_evidence_completeness(exp1: list[dict]) -> list[str]:
    """
    Flag samples whose supporting passages never reached the model.

    This is the quietest failure mode in the whole pipeline. Truncating the
    context drops evidence, every condition then correctly reports that the
    passages do not contain the answer, and the accuracy columns fill with
    zeros that look like a uniformly weak model. Nothing errors. A pilot ran
    to completion this way and the numbers were meaningless.

    Two checks: the recorded flag, and a behavioural tell for runs predating
    it — a high rate of "not in the passages" responses across every
    condition at once, which a working setup does not produce.
    """
    flags: list[str] = []
    if not exp1:
        return flags

    flagged = [r for r in exp1 if r.get("evidence_complete") is False]
    if flagged:
        n_samples = len({r["sample_id"] for r in flagged})
        total = len({r["sample_id"] for r in exp1})
        flags.append(
            f"  {RED}exp1: {n_samples}/{total} samples lost supporting passages "
            f"before reaching the model — those rows measure prompt "
            f"construction, not protocol{RST}"
        )

    # Behavioural fallback for older result files without the flag.
    refusal_markers = (
        "do not contain", "does not contain", "no information",
        "cannot be answered", "not possible to answer", "not mention",
        "insufficient",
    )
    by_sample: dict[str, list[dict]] = defaultdict(list)
    for r in exp1:
        by_sample[r["sample_id"]].append(r)

    unanimous = 0
    for sid, rows in by_sample.items():
        if len(rows) < 2:
            continue
        if all(
            any(m in (r.get("response", "") or "").lower() for m in refusal_markers)
            for r in rows
        ):
            unanimous += 1

    if by_sample:
        rate = unanimous / len(by_sample)
        if rate > 0.4:
            flags.append(
                f"  {RED}exp1: on {rate:.0%} of samples every condition reported "
                f"the answer absent from the passages — suspect context "
                f"truncation dropping supporting facts{RST}"
            )
        elif rate > 0.2:
            flags.append(
                f"  {YEL}exp1: {rate:.0%} of samples had all conditions report "
                f"missing evidence — check context construction{RST}"
            )
    return flags


def check_structured_output(exp1: list[dict]) -> list[str]:
    """
    Flag conditions whose JSON output does not parse.

    A protocol message that a receiving agent cannot parse has failed at its
    only job, but nothing downstream notices: token counts still record, and
    the reconstruction stage still extracts an answer from the raw text. One
    pilot condition emitted comma-separated strings instead of a JSON array
    on 60% of samples and reported perfectly ordinary-looking figures.

    Only conditions that are supposed to emit JSON are checked; prose
    baselines are skipped.
    """
    flags: list[str] = []
    if not exp1:
        return flags

    structured_prefixes = ("SK", "JSON-")
    by_cond: dict[str, list[dict]] = defaultdict(list)
    for r in exp1:
        cond = r.get("condition", "")
        if cond.startswith(structured_prefixes):
            by_cond[cond].append(r)

    for cond, rows in sorted(by_cond.items()):
        bad = 0
        for r in rows:
            text = (r.get("response") or "").strip()
            if text.startswith("```"):
                text = re.sub(r"^```[a-zA-Z]*\n", "", text)
                text = re.sub(r"\n```\s*$", "", text).strip()
            try:
                json.loads(text)
            except (json.JSONDecodeError, ValueError):
                bad += 1
        if not rows:
            continue
        rate = bad / len(rows)
        if rate > 0.20:
            flags.append(
                f"  {RED}exp1 [{cond}]: {rate:.0%} of responses are not valid "
                f"JSON ({bad}/{len(rows)}) — the protocol is not being "
                f"produced, so its token and accuracy figures are not "
                f"comparable{RST}"
            )
        elif rate > 0.05:
            flags.append(
                f"  {YEL}exp1 [{cond}]: {rate:.0%} of responses are malformed "
                f"JSON ({bad}/{len(rows)}){RST}"
            )
    return flags


def check_truncation(exp1: list[dict]) -> list[str]:
    """
    Flag runs where output hit the token cap.

    This matters more than it looks. Experiment 1 measures output tokens as a
    proxy for protocol verbosity; if every condition is truncated at the same
    limit, the compression ratio is exactly 1.0 by construction and tells us
    nothing. A previous pilot reported 512.0 tokens with a [512, 512] interval
    across all three conditions — zero variance, which no real generation
    process produces — and every downstream accuracy metric was zero because
    the answers never got written before the cap.

    Usual cause: the model is emitting chain-of-thought as ordinary output.
    """
    flags: list[str] = []
    if not exp1:
        return flags

    by_cond: dict[str, list[dict]] = defaultdict(list)
    for r in exp1:
        by_cond[r["condition"]].append(r)

    for cond, rows in sorted(by_cond.items()):
        toks = [r.get("out_tokens", 0) for r in rows]
        if not toks:
            continue
        n_len_finish = sum(1 for r in rows if r.get("finish_reason") == "length")
        rate = n_len_finish / len(rows)

        # Zero variance in output length is the strongest tell: real
        # generations vary, capped ones do not.
        if len(set(toks)) == 1 and toks[0] > 0:
            flags.append(
                f"  {RED}exp1 [{cond}]: every response is exactly {toks[0]} "
                f"tokens — output is capped, not measured{RST}"
            )
        elif rate > 0.25:
            flags.append(
                f"  {RED}exp1 [{cond}]: {rate:.0%} of responses hit the token "
                f"cap (finish_reason=length){RST}"
            )
        elif rate > 0.05:
            flags.append(
                f"  {YEL}exp1 [{cond}]: {rate:.0%} of responses truncated{RST}"
            )

    # Leaked reasoning, when the backend reports it separately.
    n_reasoning = sum(1 for r in exp1 if (r.get("reasoning") or "").strip())
    if n_reasoning:
        flags.append(
            f"  {RED}exp1: {n_reasoning}/{len(exp1)} responses carried separate "
            f"reasoning output — those tokens are billed as protocol cost{RST}"
        )

    return flags


def check_verdict_parsing(exp2: list[dict]) -> list[str]:
    """
    Flag when verifier verdicts could not be read.

    An unparsed verdict is not a negative detection — it is a missing
    observation. If the rate is high, the detection metric is computed over
    whatever fraction happened to parse, which is not a random subset.
    """
    flags: list[str] = []
    applied = [r for r in exp2 if r.get("injection_applied")]
    if not applied:
        return flags

    by_cond: dict[str, list[dict]] = defaultdict(list)
    for r in applied:
        by_cond[r["condition"]].append(r)

    for cond, rows in sorted(by_cond.items()):
        n_parsed = sum(1 for r in rows if r.get("verdict_parsed"))
        rate = 1 - (n_parsed / len(rows))
        if n_parsed == 0:
            flags.append(
                f"  {RED}exp2 [{cond}]: 0/{len(rows)} verdicts parsed — the "
                f"detection metric has no data at all{RST}"
            )
        elif rate > 0.20:
            flags.append(
                f"  {RED}exp2 [{cond}]: {rate:.0%} of verdicts unparsed "
                f"({n_parsed}/{len(rows)} usable){RST}"
            )
        elif rate > 0.05:
            flags.append(
                f"  {YEL}exp2 [{cond}]: {rate:.0%} of verdicts unparsed{RST}"
            )
    return flags


def check_floor_ceiling(rows: list[dict], metric: str, experiment_name: str) -> list[str]:
    """Flag if every sample in a condition scores 0 or 1 on a bounded metric."""
    flags: list[str] = []
    by_cond: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        v = r.get(metric)
        if v is None or isinstance(v, (list, dict)):
            continue
        try:
            by_cond[r["condition"]].append(float(v))
        except (TypeError, ValueError):
            continue
    for cond, vals in by_cond.items():
        if not vals:
            continue
        mn, mx = min(vals), max(vals)
        if mn == mx == 0:
            flags.append(
                f"  {RED}{experiment_name}.{metric} [{cond}]: FLOOR "
                f"(all {len(vals)} samples = 0){RST}"
            )
        elif mn == mx == 1:
            flags.append(
                f"  {RED}{experiment_name}.{metric} [{cond}]: CEILING "
                f"(all {len(vals)} samples = 1){RST}"
            )
        elif abs(mx - mn) < 1e-6:
            flags.append(
                f"  {YEL}{experiment_name}.{metric} [{cond}]: zero variance "
                f"(all = {mn:.3f}){RST}"
            )
    return flags


def check_identical_responses(exp1: list[dict]) -> list[str]:
    """If many samples produce identical responses within a condition, smells like a bug."""
    flags: list[str] = []
    by_cond: dict[str, list[str]] = defaultdict(list)
    for r in exp1:
        by_cond[r["condition"]].append(r.get("response", "")[:200])
    for cond, resps in by_cond.items():
        if len(resps) < 3:
            continue
        counter = Counter(resps)
        most_common, count = counter.most_common(1)[0]
        if count > len(resps) * 0.5:
            flags.append(
                f"  {RED}exp1 [{cond}]: {count}/{len(resps)} samples produced "
                f"the same 200-char response prefix (possible bug or cache issue){RST}"
            )
    return flags


def summarize_numbers(rows: list[dict], experiment_name: str, metrics: list[str]) -> list[str]:
    out: list[str] = []
    if not rows:
        out.append(f"  (no {experiment_name} data)")
        return out
    by_cond: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cond[r["condition"]].append(r)
    for cond in sorted(by_cond):
        line_parts = [f"  [{cond}] n={len(by_cond[cond])}"]
        for m in metrics:
            vals = []
            for r in by_cond[cond]:
                v = r.get(m)
                if v is None:
                    continue
                try:
                    vals.append(float(v))
                except (TypeError, ValueError):
                    continue
            if vals:
                mean = sum(vals) / len(vals)
                line_parts.append(f"{m}={mean:.3f}")
        out.append("  ".join(line_parts))
    return out


def main():
    if len(sys.argv) < 2:
        print("Usage: python sanity.py <results_dir>", file=sys.stderr)
        sys.exit(2)

    results_dir = Path(sys.argv[1])
    exp1 = load_jsonl(results_dir / "exp1.jsonl")
    exp2 = load_jsonl(results_dir / "exp2.jsonl")
    exp3 = load_jsonl(results_dir / "exp3.jsonl")

    print(f"Inspecting {results_dir}")
    print(f"  exp1.jsonl: {len(exp1)} rows")
    print(f"  exp2.jsonl: {len(exp2)} rows")
    print(f"  exp3.jsonl: {len(exp3)} rows")
    print()

    all_flags: list[str] = []

    print("=== 1. Scoring completeness ===")
    f = check_judge_parse_errors(exp1, "exp1")
    all_flags.extend(f); [print(x) for x in f]
    f = check_judge_parse_errors(exp2, "exp2")
    all_flags.extend(f); [print(x) for x in f]
    f = check_judge_parse_errors(exp3, "exp3")
    all_flags.extend(f); [print(x) for x in f]
    print()

    if exp2:
        print("=== 2. Error injection coverage ===")
        f = check_injection_coverage(exp2)
        all_flags.extend(f); [print(x) for x in f]
        print()

    print("=== 3. Evidence completeness ===")
    # Runs first among the data-quality checks: if the evidence never arrived,
    # every accuracy figure below is measuring the wrong thing.
    f = check_evidence_completeness(exp1)
    all_flags.extend(f); [print(x) for x in f]
    if not f:
        print(f"  {GRN}supporting passages reached the model{RST}")
    print()

    print("=== 4. Structured output validity ===")
    f = check_structured_output(exp1)
    all_flags.extend(f); [print(x) for x in f]
    if not f:
        print(f"  {GRN}structured conditions emit valid JSON{RST}")
    print()

    print("=== 5. Output truncation and verdict parsing ===")
    # These run before floor/ceiling because they usually explain it: an
    # all-zero accuracy column is a symptom when the answers were cut off.
    f = check_truncation(exp1)
    all_flags.extend(f); [print(x) for x in f]
    f = check_verdict_parsing(exp2)
    all_flags.extend(f); [print(x) for x in f]
    if not all_flags:
        print(f"  {GRN}no truncation or parsing problems{RST}")
    print()

    print("=== 6. Floor/ceiling effects ===")
    for metric in ("em", "f1"):
        f = check_floor_ceiling(exp1, metric, "exp1")
        all_flags.extend(f); [print(x) for x in f]
    for metric in ("final_em", "final_f1", "on_topic_proportion"):
        f = check_floor_ceiling(exp3, metric, "exp3")
        all_flags.extend(f); [print(x) for x in f]
    print()

    print("=== 7. Suspicious identical responses ===")
    f = check_identical_responses(exp1)
    all_flags.extend(f); [print(x) for x in f]
    if not f:
        print(f"  {GRN}no identical-response clusters{RST}")
    print()

    print("=== 8. Headline numbers ===")
    print("Experiment 1 (compression + SRE):")
    for line in summarize_numbers(exp1, "exp1", ["out_tokens", "f1", "em"]):
        print(line)
    print()
    print("Experiment 2 (error propagation/detection):")
    for line in summarize_numbers(
        exp2, "exp2",
        ["a1_tokens", "a2_tokens", "a3_tokens"],
    ):
        print(line)
    # Propagation / detection need their own loop (they're bool/None)
    if exp2:
        by_cond: dict[str, list[dict]] = defaultdict(list)
        for r in exp2:
            by_cond[r["condition"]].append(r)
        for cond in sorted(by_cond):
            props = [r["propagated"] for r in by_cond[cond] if r.get("propagated") is not None]
            dets = [r["detected"] for r in by_cond[cond] if r.get("detected") is not None]
            p_rate = sum(props) / len(props) if props else None
            d_rate = sum(dets) / len(dets) if dets else None
            print(
                f"  [{cond}] P-rate={p_rate:.3f} ({len(props)})  "
                f"D-rate={d_rate:.3f} ({len(dets)})"
                if (p_rate is not None and d_rate is not None) else
                f"  [{cond}] (insufficient judge data)"
            )
    print()
    print("Experiment 3 (drift):")
    for line in summarize_numbers(
        exp3, "exp3", ["mean_drift", "on_topic_proportion", "final_em", "final_f1"]
    ):
        print(line)
    print()

    red_count = sum(1 for f in all_flags if RED in f)
    yel_count = sum(1 for f in all_flags if YEL in f)
    print("=== Summary ===")
    print(f"Red flags:    {red_count}")
    print(f"Yellow flags: {yel_count}")
    if red_count > 0:
        print(f"{RED}Pilot has serious issues — do NOT proceed to full run without fixes.{RST}")
        sys.exit(1)
    elif yel_count > 0:
        print(f"{YEL}Pilot has minor issues — review flags before full run.{RST}")
        sys.exit(0)
    else:
        print(f"{GRN}All checks clean — safe to proceed to full run.{RST}")
        sys.exit(0)


if __name__ == "__main__":
    main()