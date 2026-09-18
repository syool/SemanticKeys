"""
runner.py — All four experiments.

Runs against a local vLLM server. Nothing here requires network access, and
no language model participates in scoring: every metric is computed by
deterministic rule over ground truth the design itself supplies. See
docs/protocol.md for why that is possible and what it buys.

Experiments
-----------
1. Compression      Single-turn. Each condition produces one protocol message;
                    a reconstructor then answers from that message alone, and
                    HotpotQA EM/F1 scores the attempt. Measures what the
                    protocol costs and whether it preserves the answer.

2. Error detection  Three-agent chain over a poisoned passage. Propagation and
                    detection are decided by rule against the token we injected,
                    so "wrong for an unrelated reason" is not counted as
                    propagation.

3. Drift            Six-turn conversation with distractors. Drift is embedding
                    cosine distance from the question, after stripping protocol
                    syntax so the comparison is about topic rather than
                    JSON-versus-prose.

4. Horizon sweep    Experiment 3 across turn counts and context windows, driven
                    by scripts/sweep.sh. Records per-turn token cost and how
                    many turns each condition had to drop, which is where the
                    structural difference between protocols shows up.

Design notes
------------
Constrained decoding is applied only to the verifier verdict in Experiment 2.
That turn is a measurement instrument shared by every condition, so
constraining it advantages no protocol; constraining the agents' primary
outputs would hand structured conditions a reliability the prose ones lack.

Requests run concurrently. Sequential execution measured ~20 tok/s on an
L40S against ~308 at concurrency 32, which is the difference between a sweep
finishing overnight and taking a week.

Usage
-----
    python runner.py \\
        --base-url http://localhost:8000/v1 \\
        --model Qwen/Qwen3.5-27B \\
        --data-path ../data/hotpot_dev_distractor_v1.json \\
        --n-samples 100 \\
        --conditions NL-free NL-concise NL-CoT JSON-free SK SK-index \\
        --exp all --concurrency 32 \\
        --output-dir ../results/full/qwen
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import Lock
from typing import Callable, Iterable, TypeVar

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from protocols import (
    MULTI_TURN_ONLY,
    VALID_CONDITIONS,
    build_system_prompt,
    build_turn_goal_reminder,
    build_verifier_prompt,
    pick_distractor,
)
from data import (
    extract_context_text,
    load_hotpotqa,
    load_samples,
    supporting_facts_present,
)
from injection import balanced_injection_plan
from client import OpenAICompatibleClient
from metrics import EmbeddingBackend, cosine, exact_match, f1_score
from scoring import (
    RECONSTRUCT_SYSTEM_PROMPT,
    verifier_response_format,
    build_reconstruct_prompt,
    clean_reconstructed_answer,
    score_detection,
    score_on_topic,
    score_propagation,
    score_recoverability,
)


# =============================================================================
# Helpers
# =============================================================================


def _save_jsonl(path: str | Path, records: Iterable[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


_T = TypeVar("_T")
_R = TypeVar("_R")

# Default concurrency. vLLM batches concurrent requests, so throughput rises
# sharply above one in-flight request — measured ~21 tok/s sequential on an
# L40S, which turns a long-horizon sweep into days of wall-clock. Keep this
# comfortably under the server's --max-num-seqs.
DEFAULT_CONCURRENCY = 16


def parallel_map(
    fn: Callable[[_T], _R],
    items: list[_T],
    concurrency: int = DEFAULT_CONCURRENCY,
    label: str = "",
) -> list[_R]:
    """
    Apply `fn` across `items` with bounded concurrency, preserving input order.

    Order matters for reproducibility: results are written to JSONL and later
    paired by sample_id, and a stable order keeps runs diffable. Exceptions
    propagate rather than being swallowed — a silently dropped observation
    would shrink one condition's N without anyone noticing.

    Concurrency is safe here because each unit of work is an independent
    (sample, condition) pair. The clients hold no mutable state across calls.
    """
    if concurrency <= 1:
        out: list[_R] = []
        for i, item in enumerate(items, 1):
            out.append(fn(item))
            if label:
                print(f"  [{label}] {i}/{len(items)}")
        return out

    results: list[_R | None] = [None] * len(items)
    done = 0
    lock = Lock()

    def run(idx_item: tuple[int, _T]) -> None:
        nonlocal done
        idx, item = idx_item
        value = fn(item)
        with lock:
            results[idx] = value
            done += 1
            if label and (done % 10 == 0 or done == len(items)):
                print(f"  [{label}] {done}/{len(items)}")

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        # list() forces evaluation so exceptions surface here rather than
        # being lost when the executor is garbage-collected.
        list(pool.map(run, enumerate(items)))

    missing = [i for i, r in enumerate(results) if r is None]
    if missing:
        raise RuntimeError(
            f"parallel_map lost {len(missing)} results at indices {missing[:5]}"
        )
    return results  # type: ignore[return-value]


def _extract_verifier_verdict(text: str) -> tuple[bool | None, list[str]]:
    """
    Read {"verified": bool, "reasons": [...]} from a verifier response.

    With guided decoding enabled this should essentially always succeed. We
    keep the tolerant parser anyway so the runner degrades gracefully if
    guided decoding is unavailable on the serving backend.
    """
    from metrics import _extract_first_json

    obj = _extract_first_json(text)
    if obj.get("_parse_error"):
        return None, []

    if "verified" in obj:
        reasons = obj.get("reasons", []) or []
        if not isinstance(reasons, list):
            reasons = [str(reasons)]
        return bool(obj["verified"]), [str(r) for r in reasons]

    # SK-style fallbacks (in case guided decoding is off and the model used
    # the protocol's own fields instead of the verdict schema).
    cond = obj.get("conditions")
    expl = obj.get("explanation", "")
    if cond in ("verified", "unverified"):
        return cond == "verified", [expl] if expl else []
    if isinstance(expl, str):
        upper = expl.strip().upper()
        if upper.startswith("VERIFIED:"):
            return True, [expl]
        if upper.startswith("UNVERIFIED:"):
            return False, [expl]

    return None, []


# =============================================================================
# Experiment 1 — Compression + reconstruction (judge-free)
# =============================================================================


@dataclass
class Exp1Observation:
    sample_id: str
    condition: str
    question: str
    gold_answer: str
    response: str
    in_tokens: int
    out_tokens: int
    # Reconstruction stage
    reconstructed_raw: str = ""
    reconstructed_answer: str = ""
    em: int = 0
    f1: float = 0.0
    recoverable_strict: bool = False
    recoverable_lenient: bool = False
    # Diagnostics
    latency_ms: float = 0.0
    finish_reason: str = ""
    # Non-empty when the backend returned chain-of-thought separately.
    # Those tokens are billed to completion_tokens, so any value here
    # means the compression figures for this row are not trustworthy.
    reasoning: str = ""
    # False when a supporting passage never reached the model. Such a sample
    # is unanswerable by construction, and every condition will correctly say
    # so — which looks like uniform model failure rather than a broken prompt.
    evidence_complete: bool = True


def run_experiment1(
    client,
    tasks: list[dict],
    conditions: list[str],
    max_tokens: int = 512,
    reconstruct_max_tokens: int = 48,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> list[Exp1Observation]:
    """
    Stage 1: Researcher produces a protocol message.
    Stage 2: Reconstructor sees ONLY that message + the question, and answers.
    Stage 3: Deterministic EM/F1 scoring of the reconstruction.

    The reconstructor is the same local model for every condition, so any
    capability ceiling applies uniformly and cannot manufacture a
    between-condition difference.
    """
    # Sentence numbering is applied uniformly whenever any condition needs it,
    # so the prompt is identical across the comparison. Enabling it only for
    # SK-index would change that condition's input and confound the contrast.
    number_sentences = any(c == "SK-index" for c in conditions)

    units = [(task, cond) for task in tasks for cond in conditions]

    def run_one(unit: tuple[dict, str]) -> Exp1Observation:
        task, condition = unit
        context = extract_context_text(task, number_sentences=number_sentences)
        user_prompt = (
            f"Analyze the following passages to answer the question.\n\n"
            f"Question: {task['question']}\n\nPassages:\n{context}"
        )
        evidence_ok = supporting_facts_present(task, context)

        system = build_system_prompt("researcher", condition)
        resp = client.call(system, user_prompt, max_tokens=max_tokens)

        obs = Exp1Observation(
            sample_id=task["_id"],
            condition=condition,
            question=task["question"],
            gold_answer=task["answer"],
            response=resp.content,
            in_tokens=resp.input_tokens,
            out_tokens=resp.output_tokens,
            latency_ms=resp.latency_ms,
            finish_reason=resp.finish_reason,
            reasoning=resp.reasoning,
            evidence_complete=evidence_ok,
        )

        # Stage 2: reconstruct from the protocol message alone.
        recon = client.call(
            RECONSTRUCT_SYSTEM_PROMPT,
            build_reconstruct_prompt(task["question"], resp.content),
            max_tokens=reconstruct_max_tokens,
        )
        obs.reconstructed_raw = recon.content
        obs.reconstructed_answer = clean_reconstructed_answer(recon.content)

        # Stage 3: deterministic scoring.
        scored = score_recoverability(obs.reconstructed_answer, task["answer"])
        obs.em = scored.em
        obs.f1 = scored.f1
        obs.recoverable_strict = scored.recoverable_strict
        obs.recoverable_lenient = scored.recoverable_lenient
        return obs

    return parallel_map(run_one, units, concurrency=concurrency, label="Exp1")


# =============================================================================
# Experiment 2 — Cascading error detection (judge-free)
# =============================================================================


@dataclass
class Exp2Observation:
    sample_id: str
    condition: str
    injection_strategy: str
    injection_applied: bool
    injection_original_sentence: str
    injection_poisoned_sentence: str
    injected_token: str
    original_token: str
    gold_answer: str
    question: str
    a1_output: str
    a2_output: str
    a3_output: str
    # Verifier verdict
    verified_flag: bool | None = None
    verifier_reasons: list[str] = field(default_factory=list)
    verdict_parsed: bool = False
    # Deterministic propagation scoring
    final_em: int = 0
    final_f1: float = 0.0
    wrong_vs_gold: bool = False
    mentions_injected: bool = False
    mentions_original: bool = False
    propagated: bool = False
    propagated_lenient: bool = False
    # Deterministic detection scoring
    detected_strict: bool = False
    detected_lenient: bool = False
    detection_reasons_text: str = ""
    # Tokens
    a1_tokens: int = 0
    a2_tokens: int = 0
    a3_tokens: int = 0


def run_experiment2(
    client,
    verifier_client,
    tasks: list[dict],
    conditions: list[str],
    max_tokens_per_agent: int = 400,
) -> list[Exp2Observation]:
    """
    Three-agent chain on poisoned inputs, scored entirely by rule.

    `verifier_client` is a separate client instance configured with guided
    decoding for the verdict schema. Using a distinct client (rather than a
    per-call flag) keeps the constrained-decoding surface explicit and
    auditable: only verifier turns go through it.
    """
    plan = balanced_injection_plan(tasks, seed=2024)
    results: list[Exp2Observation] = []
    total = len(plan) * len(conditions)
    step = 0

    for (poisoned_task, rec, strat) in plan:
        # Budgeted, but supporting facts are kept first — the injected error
        # lives in one of them, so dropping it would void the trial.
        poisoned_context = extract_context_text(poisoned_task, max_chars=6000)
        q = poisoned_task["question"]
        gold = poisoned_task["answer"]

        for condition in conditions:
            step += 1
            print(f"  [Exp2] {step}/{total} cond={condition} strat={strat}")

            # --- Agent 1: Researcher
            a1 = client.call(
                build_system_prompt("researcher", condition),
                f"Find information to answer: {q}\n\nSources:\n{poisoned_context}",
                max_tokens=max_tokens_per_agent,
            )
            # --- Agent 2: Reasoner
            a2 = client.call(
                build_system_prompt("reasoner", condition),
                f"Based on the researcher's findings, determine the answer.\n\n"
                f"Question: {q}\nResearcher output:\n{a1.content}",
                max_tokens=max_tokens_per_agent,
            )
            # --- Agent 3: Verifier (guided decoding on the verdict)
            a3 = verifier_client.call(
                build_verifier_prompt(condition, include_field_check=False),
                f"Verify the reasoning chain below.\n\n"
                f"Question: {q}\n"
                f"Researcher output:\n{a1.content[:600]}\n\n"
                f"Reasoner output:\n{a2.content}",
                max_tokens=max_tokens_per_agent,
            )

            verified, reasons = _extract_verifier_verdict(a3.content)

            obs = Exp2Observation(
                sample_id=poisoned_task["_id"],
                condition=condition,
                injection_strategy=strat,
                injection_applied=rec.applied,
                injection_original_sentence=rec.original_sentence,
                injection_poisoned_sentence=rec.poisoned_sentence,
                injected_token=rec.replacement_token,
                original_token=rec.original_token,
                gold_answer=gold,
                question=q,
                a1_output=a1.content,
                a2_output=a2.content,
                a3_output=a3.content,
                verified_flag=verified,
                verifier_reasons=reasons,
                a1_tokens=a1.output_tokens,
                a2_tokens=a2.output_tokens,
                a3_tokens=a3.output_tokens,
            )

            if rec.applied:
                # --- Deterministic propagation scoring
                prop = score_propagation(
                    final_answer=a2.content,
                    gold_answer=gold,
                    injected_token=rec.replacement_token,
                    original_token=rec.original_token,
                )
                obs.final_em = prop.final_em
                obs.final_f1 = prop.final_f1
                obs.wrong_vs_gold = prop.wrong_vs_gold
                obs.mentions_injected = prop.mentions_injected
                obs.mentions_original = prop.mentions_original
                obs.propagated = prop.propagated
                obs.propagated_lenient = prop.propagated_lenient

                # --- Deterministic detection scoring
                det = score_detection(
                    verified_flag=verified,
                    reasons=reasons,
                    injected_token=rec.replacement_token,
                    original_token=rec.original_token,
                    full_verifier_text=a3.content,
                )
                obs.verdict_parsed = det.verdict_parsed
                obs.detected_strict = det.detected_strict
                obs.detected_lenient = det.detected_lenient
                obs.detection_reasons_text = det.reasons_text

            results.append(obs)

    return results


# =============================================================================
# Experiment 3 — Context drift (judge-free)
# =============================================================================


def normalize_for_embedding(text: str) -> str:
    """
    Strip protocol scaffolding so that drift reflects content, not format.

    This exists because of a measurement confound found in piloting. SK emits
    JSON; the NL conditions emit prose. Embedding both directly and comparing
    each against a natural-language question penalises SK for its syntax:
    braces, quotes, and field names are lexically distant from the question no
    matter how on-topic the content is. A pilot showed SK apparently drifting
    more than an NL baseline, which is not a finding about state anchoring —
    it is an artifact of comparing JSON to prose.

    So before embedding we reduce every condition to its semantic payload:
    for structured output, the field *values* joined together, dropping keys
    and punctuation; for prose, the text unchanged. Both conditions then get
    compared on equal footing.

    Non-JSON input is returned unchanged, so this is safe to apply uniformly.
    """
    if not text:
        return ""
    stripped = text.strip()

    # Tolerate markdown fences around JSON.
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z]*\n", "", stripped)
        stripped = re.sub(r"\n```\s*$", "", stripped)
        stripped = stripped.strip()

    if not stripped.startswith("{"):
        return text  # prose, nothing to strip

    try:
        obj = json.loads(stripped)
    except json.JSONDecodeError:
        return text  # malformed JSON — treat as prose rather than guessing

    if not isinstance(obj, dict):
        return text

    def flatten(value) -> list[str]:
        if isinstance(value, str):
            return [value]
        if isinstance(value, (int, float, bool)):
            return [str(value)]
        if isinstance(value, list):
            return [s for v in value for s in flatten(v)]
        if isinstance(value, dict):
            # Values only: field names are protocol vocabulary, not content.
            return [s for v in value.values() for s in flatten(v)]
        return []

    parts = [p.strip() for p in flatten(obj) if p and p.strip()]
    # If a structured message carried no text, fall back rather than returning
    # an empty string, which would produce a meaningless embedding.
    return " ".join(parts) if parts else text


@dataclass
class Exp3Observation:
    sample_id: str
    condition: str
    question: str
    gold_answer: str
    num_turns: int
    turn_responses: list[str]
    per_turn_similarity: list[float]
    per_turn_drift: list[float]
    mean_drift: float
    on_topic_labels: list[str]
    on_topic_proportion: float
    final_answer: str
    final_em: int
    final_f1: float
    # Cost axis for the horizon sweep. Input tokens matter as much as output:
    # NL conditions re-send a growing transcript each turn while SK propagates
    # a fixed-size state tuple, so the gap should widen with turn count.
    per_turn_out_tokens: list[int] = field(default_factory=list)
    per_turn_in_tokens: list[int] = field(default_factory=list)
    cumulative_out_tokens: int = 0
    cumulative_in_tokens: int = 0
    # True when the transcript had to be trimmed to fit the window.
    context_overflowed: bool = False
    # How many oldest turns were dropped at the worst point. SK variants
    # report 0 by construction; the gap is a finding, not missing data.
    turns_dropped: int = 0
    # True when the question text was withheld after the first turn, so
    # the objective had to survive in whatever the protocol carried.
    question_removed: bool = False


FINAL_ANSWER_SYSTEM = """You extract a short answer from a record of work already done.

Reply with the answer span alone — the shortest phrase that answers the
question. A name, a date, a place, a number, or yes/no.

Do not include:
  - leading words such as "a", "the", "part of", "he was", "it is"
  - explanation, punctuation, or units unless they are part of the answer
  - any text other than the span itself

If the record does not contain the answer, reply exactly: INSUFFICIENT

Examples
  Question: What was he on during the attack?
  Record:   "...he was part of the flotilla that attacked..."
  Reply:    flotilla

  Question: When did the park open?
  Record:   "The park opened on 15 August 1843."
  Reply:    15 August 1843
"""


def _trim_transcript(
    header: str,
    turns: list[tuple[str, str]],
    budget_chars: int,
) -> tuple[str, int]:
    """
    Assemble a transcript that fits `budget_chars`, dropping oldest turns first.

    This is the crux of the horizon sweep. NL conditions re-send the entire
    conversation each turn, so their prompt grows without bound: at 250 output
    tokens per turn an NL transcript fills a 4K window around turn 11 and a 16K
    window around turn 53. SK propagates a state tuple of roughly constant size
    and never approaches either limit.

    Something has to give at that point, and the choice shapes what the
    experiment measures. Erroring out yields no data exactly where the
    comparison gets interesting. Keeping the oldest turns and dropping recent
    ones is not what any deployment does. Dropping the oldest — a sliding
    window — is the standard mitigation and the fair comparison: it is what an
    engineer would actually do, and it costs the NL condition its early
    context, which is precisely the failure mode SK claims to avoid.

    The header (task context and instructions) is always retained; only
    conversation turns are dropped. Returns the assembled text and the number
    of turns dropped, so the caller can record that degradation began rather
    than silently reporting a shorter transcript.
    """
    parts = [f"\nUser: {u}\nAssistant: {a}\n" for u, a in turns]
    dropped = 0
    while parts and len(header) + sum(len(p) for p in parts) > budget_chars:
        parts.pop(0)
        dropped += 1
    return header + "".join(parts), dropped


def _run_turns_nl(
    client, task: dict, condition: str, with_goal_anchor: bool,
    num_turns: int, distractor_tier: str, max_tokens: int = 250,
    prompt_budget_chars: int = 48_000,
    remove_question: bool = False,
) -> tuple[list[str], str, list[int], list[int], int]:
    # Under question removal the reminder keeps its function — a standing
    # instruction not to drift — but stops restating the question, so the
    # model must recover the objective from the transcript it retained.
    goal_reminder = build_turn_goal_reminder(
        task["question"], include_question=not remove_question
    )
    context_snippet = extract_context_text(task, max_chars=3000)
    header = f"Context:\n{context_snippet}\n\n"
    turn_log: list[tuple[str, str]] = []
    responses: list[str] = []
    out_toks: list[int] = []
    in_toks: list[int] = []
    dropped_total = 0

    for turn in range(num_turns):
        distractor = pick_distractor(turn, tier=distractor_tier)
        if turn == 0:
            # The question is stated once, on the first turn, for every
            # condition. Previously only the anchored conditions ever saw it —
            # SK receives it in the state tuple's goal field and NL-anchor in
            # its per-turn reminder — so NL-append spent entire runs replying
            # "please provide the question". That measured a prompt omission,
            # not a protocol. What still differs between conditions is how the
            # question is *retained* after this turn, which is the thing under
            # test.
            user_msg = (
                f"Question: {task['question']}\n\n"
                f"Begin working toward the answer (step 1)."
            )
        elif turn == 1:
            user_msg = f"Continue toward the answer (step {turn + 1})."
        elif turn < 5:
            user_msg = f"{distractor}\nAlso, continue the original task (step {turn + 1})."
        else:
            user_msg = f"{distractor}\nAnyway, what were we doing? Continue step {turn + 1}."

        sys_prompt = build_system_prompt("reasoner", condition)
        if with_goal_anchor:
            sys_prompt = goal_reminder + "\n\n" + sys_prompt

        # Reserve room for the system prompt, this turn's message, and the
        # response. Chars-per-token is approximate; the margin is deliberate
        # so a mis-estimate costs one dropped turn rather than a failed call.
        budget = max(
            500,
            prompt_budget_chars - len(sys_prompt) - len(user_msg) - max_tokens * 4,
        )
        history, dropped = _trim_transcript(header, turn_log, budget)
        dropped_total = max(dropped_total, dropped)

        resp = client.call(sys_prompt, history + user_msg, max_tokens=max_tokens)
        responses.append(resp.content)
        out_toks.append(resp.output_tokens)
        in_toks.append(resp.input_tokens)

        # What gets written into the transcript for later turns. Under question
        # removal the first turn's message must be recorded in redacted form:
        # NL conditions re-send their history every turn, so leaving the
        # question there would carry it forward indefinitely and the condition
        # would test nothing. The model's own response is left untouched —
        # whatever it chose to restate is part of what the protocol retained,
        # and editing that would be editing the result.
        if remove_question and turn == 0:
            logged_msg = "Begin working toward the answer (step 1)."
        else:
            logged_msg = user_msg
        turn_log.append((logged_msg, resp.content))

    final_history, _ = _trim_transcript(header, turn_log, prompt_budget_chars // 2)
    final = client.call(
        FINAL_ANSWER_SYSTEM,
        f"Original question: {task['question']}\n\nConversation:\n{final_history[-2000:]}",
        max_tokens=48,
    )
    out_toks.append(final.output_tokens)
    in_toks.append(final.input_tokens)
    return responses, final.content, out_toks, in_toks, dropped_total


def _run_turns_sk(
    client, task: dict, num_turns: int, distractor_tier: str,
    max_tokens: int = 250, sk_condition: str = "SK",
    prompt_budget_chars: int = 48_000,
    remove_question: bool = False,
) -> tuple[list[str], str, list[int], list[int], int]:
    context_snippet = extract_context_text(task, max_chars=3000)
    sk_state = {
        # topic normally seeds from the question text. Under question removal
        # it must not, or the objective would survive here regardless of what
        # the protocol itself preserved — which is precisely what the
        # condition exists to test.
        "topic": (
            "the task established at the start of this conversation"
            if remove_question
            else task["question"][:100]
        ),
        # The goal field normally carries the question. Under question
        # removal it holds only the instruction, so the state tuple must
        # preserve the objective through its own fields or lose it.
        "goal": (
            "Answer the question posed at the start of this conversation."
            if remove_question
            else f"Answer: {task['question']}"
        ),
        # Same budget the NL conditions receive. Clipping this to a few hundred
        # characters while NL sees the full snippet would hand SK a smaller
        # evidence base and then attribute the resulting errors to the protocol.
        "background": context_snippet,
        "context": "Turn 0: initial state",
    }
    responses: list[str] = []
    out_toks: list[int] = []
    in_toks: list[int] = []

    for turn in range(num_turns):
        distractor = pick_distractor(turn, tier=distractor_tier)
        if turn == 0:
            # Identical wording to the NL conditions. SK also carries the
            # question in its state tuple's goal field, but stating it here
            # too keeps the user-visible prompt the same across conditions:
            # the comparison should turn on how state is retained, not on
            # which condition was phrased more helpfully.
            user_msg = (
                f"Question: {task['question']}\n\n"
                f"Begin working toward the answer (step 1)."
            )
        elif turn == 1:
            user_msg = f"Continue toward the answer (step {turn + 1})."
        elif turn < 5:
            user_msg = f"{distractor}\nAlso, continue the original task (step {turn + 1})."
        else:
            user_msg = f"{distractor}\nAnyway, what were we doing? Continue step {turn + 1}."

        sk_system = (
            build_system_prompt("reasoner", sk_condition)
            + "\n\nPERSISTENT STATE ANCHOR — do not overwrite:\n"
            + f"  topic: {sk_state['topic']}\n"
            + f"  goal:  {sk_state['goal']}"
        )
        resp = client.call(
            sk_system,
            f"Current state:\n{json.dumps(sk_state)}\n\nNew input: {user_msg}",
            max_tokens=max_tokens,
        )
        responses.append(resp.content)
        out_toks.append(resp.output_tokens)
        in_toks.append(resp.input_tokens)

        try:
            new_state = json.loads(resp.content)
            if isinstance(new_state, dict):
                new_state["goal"] = sk_state["goal"]
                new_state["topic"] = sk_state["topic"]
                new_state["context"] = f"Turn {turn + 1}"
                sk_state.update({k: v for k, v in new_state.items() if v})
        except json.JSONDecodeError:
            sk_state["context"] = f"Turn {turn + 1}: {resp.content[:120]}"

    final = client.call(
        FINAL_ANSWER_SYSTEM,
        f"Original question: {task['question']}\n\nFinal state:\n{json.dumps(sk_state)}",
        max_tokens=48,
    )
    out_toks.append(final.output_tokens)
    in_toks.append(final.input_tokens)
    # SK's propagated state is bounded, so no turns are ever dropped.
    # Returning 0 rather than omitting the field keeps the comparison
    # symmetric: the difference from the NL conditions is a finding, not
    # a gap in the measurement.
    return responses, final.content, out_toks, in_toks, 0


def run_experiment3(
    client,
    embedder: EmbeddingBackend,
    tasks: list[dict],
    conditions: list[str],
    num_turns: int = 8,
    distractor_tier: str = "mixed",
    on_topic_threshold: float = 0.55,
    off_topic_threshold: float = 0.35,
    warmup_turns: int = 2,
    concurrency: int = DEFAULT_CONCURRENCY,
    prompt_budget_chars: int = 48_000,
    remove_question: bool = False,
) -> list[Exp3Observation]:
    """
    Multi-turn drift measurement. Drift and on-topic labels both derive from
    embedding cosine similarity — no LLM judgment.

    NOTE for the paper: because the on-topic label is a thresholded version
    of the same similarity that produces the drift score, it is not
    independent corroboration. State this explicitly rather than presenting
    two apparently independent drift measures.
    """
    units = [(task, cond) for task in tasks for cond in conditions]

    def run_one(unit: tuple[dict, str]) -> Exp3Observation:
        task, condition = unit

        # Any SK variant propagates a state tuple; the NL variants accumulate
        # a transcript. That difference is what the sweep measures, so route on
        # it rather than on the exact condition name.
        if condition.startswith("SK"):
            turn_responses, final, out_toks, in_toks, dropped = _run_turns_sk(
                client, task, num_turns, distractor_tier,
                sk_condition=condition, prompt_budget_chars=prompt_budget_chars,
                remove_question=remove_question,
            )
        elif condition == "NL-anchor":
            turn_responses, final, out_toks, in_toks, dropped = _run_turns_nl(
                client, task, "NL-free", True, num_turns, distractor_tier,
                prompt_budget_chars=prompt_budget_chars,
                remove_question=remove_question,
            )
        elif condition == "NL-append":
            turn_responses, final, out_toks, in_toks, dropped = _run_turns_nl(
                client, task, "NL-free", False, num_turns, distractor_tier,
                prompt_budget_chars=prompt_budget_chars,
                remove_question=remove_question,
            )
        else:
            turn_responses, final, out_toks, in_toks, dropped = _run_turns_nl(
                client, task, condition, False, num_turns, distractor_tier,
                prompt_budget_chars=prompt_budget_chars,
                remove_question=remove_question,
            )

        # Drift, after stripping protocol syntax so the comparison is about
        # topic rather than JSON-versus-prose surface form.
        normalized = [normalize_for_embedding(r) for r in turn_responses]
        embs = embedder.embed_batch([task["question"]] + normalized)
        q_emb = embs[0]
        sims = [cosine(q_emb, embs[i + 1]) for i in range(len(turn_responses))]
        drifts = [1.0 - s for s in sims]
        tail = drifts[warmup_turns:]
        mean_drift = sum(tail) / len(tail) if tail else 0.0

        labels = [
            score_on_topic(s, on_topic_threshold, off_topic_threshold).label
            for s in sims
        ]
        on_topic_prop = (
            sum(1 for l in labels if l in ("on_topic", "partial")) / len(labels)
            if labels else 0.0
        )

        final_clean = clean_reconstructed_answer(final)

        # Turns dropped to fit the window. Non-zero means this condition
        # lost early context — the degradation the sweep exists to measure,
        # recorded as a count rather than inferred from response shape.
        overflowed = dropped > 0

        return Exp3Observation(
            sample_id=task["_id"],
            condition=condition,
            question=task["question"],
            gold_answer=task["answer"],
            num_turns=num_turns,
            turn_responses=turn_responses,
            per_turn_similarity=sims,
            per_turn_drift=drifts,
            per_turn_out_tokens=out_toks,
            per_turn_in_tokens=in_toks,
            cumulative_out_tokens=sum(out_toks),
            cumulative_in_tokens=sum(in_toks),
            context_overflowed=overflowed,
            turns_dropped=dropped,
            question_removed=remove_question,
            mean_drift=mean_drift,
            on_topic_labels=labels,
            on_topic_proportion=on_topic_prop,
            final_answer=final_clean,
            final_em=exact_match(final_clean, task["answer"]),
            final_f1=f1_score(final_clean, task["answer"]),
        )

    return parallel_map(run_one, units, concurrency=concurrency, label="Exp3")


# =============================================================================
# Reporting
# =============================================================================


def summarize(rows: list, accessor) -> dict[str, tuple[float, float, float]]:
    from collections import defaultdict
    from metrics import bootstrap_ci

    buckets: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        v = accessor(r)
        if v is None:
            continue
        buckets[r.condition].append(float(v))
    return {c: bootstrap_ci(v) for c, v in buckets.items()}


def generate_report(exp1, exp2, exp3, output_path: str | Path) -> str:
    from collections import Counter

    def counts(rows):
        return dict(Counter(r.condition for r in rows))

    L: list[str] = []
    L.append("=" * 78)
    L.append("Semantic Keys — Offline Experiment Report (v3, judge-free)")
    L.append("=" * 78)

    if exp1:
        L.append("")
        L.append("[Exp 1] Compression + reconstruction (deterministic EM/F1)")
        L.append("-" * 78)
        out_tok = summarize(exp1, lambda r: r.out_tokens)
        strict = summarize(exp1, lambda r: 1.0 if r.recoverable_strict else 0.0)
        lenient = summarize(exp1, lambda r: 1.0 if r.recoverable_lenient else 0.0)
        f1s = summarize(exp1, lambda r: r.f1)
        L.append(f"  N per condition: {counts(exp1)}")
        L.append("")
        L.append(f"  {'condition':<13} {'out_tok':<24} {'A-SRE strict':<20} "
                 f"{'A-SRE lenient':<20} {'F1':<20}")
        for c in sorted(out_tok):
            o, s, le, f = out_tok[c], strict.get(c, (0,0,0)), lenient.get(c, (0,0,0)), f1s.get(c, (0,0,0))
            L.append(f"  {c:<13} {f'{o[0]:.1f} [{o[1]:.0f},{o[2]:.0f}]':<24} "
                     f"{f'{s[0]:.3f} [{s[1]:.2f},{s[2]:.2f}]':<20} "
                     f"{f'{le[0]:.3f} [{le[1]:.2f},{le[2]:.2f}]':<20} "
                     f"{f'{f[0]:.3f} [{f[1]:.2f},{f[2]:.2f}]':<20}")

    if exp2:
        L.append("")
        L.append("[Exp 2] Error propagation + detection (rule-based)")
        L.append("-" * 78)
        applied = [r for r in exp2 if r.injection_applied]
        parsed = [r for r in applied if r.verdict_parsed]
        L.append(f"  Injections applied: {len(applied)}/{len(exp2)}")
        L.append(f"  Verdicts parsed:    {len(parsed)}/{len(applied)} "
                 f"(unparsed excluded from D-rate)")
        L.append("")
        prop = summarize(applied, lambda r: 1.0 if r.propagated else 0.0)
        det_s = summarize(parsed, lambda r: 1.0 if r.detected_strict else 0.0)
        det_l = summarize(parsed, lambda r: 1.0 if r.detected_lenient else 0.0)
        L.append(f"  {'condition':<13} {'P-rate (↓)':<22} {'D-strict (↑)':<22} "
                 f"{'D-lenient (↑)':<22}")
        for c in sorted(prop):
            p, ds, dl = prop[c], det_s.get(c, (0,0,0)), det_l.get(c, (0,0,0))
            L.append(f"  {c:<13} {f'{p[0]:.3f} [{p[1]:.2f},{p[2]:.2f}]':<22} "
                     f"{f'{ds[0]:.3f} [{ds[1]:.2f},{ds[2]:.2f}]':<22} "
                     f"{f'{dl[0]:.3f} [{dl[1]:.2f},{dl[2]:.2f}]':<22}")

    if exp3:
        L.append("")
        L.append("[Exp 3] Context drift (embedding-based)")
        L.append("-" * 78)
        drift = summarize(exp3, lambda r: r.mean_drift)
        topic = summarize(exp3, lambda r: r.on_topic_proportion)
        gem = summarize(exp3, lambda r: r.final_em)
        gf1 = summarize(exp3, lambda r: r.final_f1)
        L.append(f"  Turns per sample: {exp3[0].num_turns}  N per cond: {counts(exp3)}")
        L.append("")
        L.append(f"  {'condition':<13} {'drift (↓)':<22} {'on-topic':<20} "
                 f"{'goal EM':<18} {'goal F1':<18}")
        for c in sorted(drift):
            d, t, e, f = drift[c], topic.get(c, (0,0,0)), gem.get(c, (0,0,0)), gf1.get(c, (0,0,0))
            L.append(f"  {c:<13} {f'{d[0]:.3f} [{d[1]:.2f},{d[2]:.2f}]':<22} "
                     f"{f'{t[0]:.2f} [{t[1]:.2f},{t[2]:.2f}]':<20} "
                     f"{f'{e[0]:.2f} [{e[1]:.2f},{e[2]:.2f}]':<18} "
                     f"{f'{f[0]:.2f} [{f[1]:.2f},{f[2]:.2f}]':<18}")

    L.append("")
    L.append("=" * 78)
    L.append("All metrics computed by deterministic rules. No LLM judge involved.")
    L.append("Intervals are 95% bootstrap (percentile, 10k resamples).")
    L.append("Hypothesis tests: run analyze.py on the saved JSONL outputs.")

    text = "\n".join(L)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(text, encoding="utf-8")
    return text


# =============================================================================
# Main
# =============================================================================


def main():
    p = argparse.ArgumentParser(
        description="Semantic Keys offline experiment runner v3 (judge-free)"
    )
    p.add_argument("--base-url", default="http://localhost:8000/v1",
                   help="vLLM OpenAI-compatible endpoint")
    p.add_argument("--model", required=True, help="Model ID as loaded in vLLM")
    p.add_argument("--data-path", required=True,
                   help="Local HotpotQA JSON (from the offline bundle)")
    p.add_argument("--embedder-path", default=None,
                   help="Local path to all-mpnet-base-v2. If omitted, "
                        "SK_EMBED_MODEL env var or the HF default is used.")
    p.add_argument("--n-samples", type=int, default=100)
    p.add_argument("--conditions", nargs="+",
                   default=["NL-free", "NL-concise", "NL-CoT", "JSON-free", "SK"],
                   choices=list(VALID_CONDITIONS))
    p.add_argument("--exp", choices=["1", "2", "3", "all"], default="all")
    p.add_argument("--num-turns", type=int, default=8)
    p.add_argument("--distractor-tier", choices=["hard", "soft", "mixed"],
                   default="mixed")
    p.add_argument("--output-dir", default="./results_v3")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prompt-budget-chars", type=int, default=48_000,
                   help="Character budget for the assembled prompt. NL "
                        "conditions drop their oldest turns to fit; SK "
                        "variants never approach it. Roughly 4 chars per "
                        "token: use 48000 for a 16K window, 12000 for 4K.")
    p.add_argument("--remove-question", action="store_true",
                   help="Withhold the question text after turn 1. It is "
                        "stated once at the start and then removed from "
                        "goal reminders and SK state fields, so the "
                        "objective must survive in what the protocol "
                        "itself carries. Tests whether accuracy holds "
                        "because of the protocol or because the question "
                        "stays visible every turn.")
    p.add_argument("--embedder-device", default=None,
                   help="Device for the drift embedder: cpu (default) or "
                        "cuda. vLLM reserves most of the GPU, so loading "
                        "the embedder there can fail with a CUBLAS "
                        "allocation error on long-turn cells. CPU costs "
                        "about 2%% of total runtime.")
    p.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                   help="Concurrent in-flight requests. Measured on an L40S: "
                        "1 gives ~20 tok/s, 16 gives ~205, 32 gives ~308. "
                        "Keep below the server's --max-num-seqs.")
    p.add_argument("--no-guided-decoding", action="store_true",
                   help="Disable guided JSON for verifier verdicts. Use only "
                        "if the serving backend does not support it.")
    p.add_argument("--on-topic-threshold", type=float, default=0.55)
    p.add_argument("--off-topic-threshold", type=float, default=0.35)
    args = p.parse_args()

    random.seed(args.seed)

    # NL-anchor and NL-append are defined by how they carry conversation state
    # across turns, so they mean nothing in a single-turn experiment. Catching
    # this here avoids producing a cell of data that silently measures
    # something other than what was asked for.
    misplaced = [c for c in args.conditions if c in MULTI_TURN_ONLY]
    if misplaced and args.exp in ("1", "2"):
        p.error(
            f"conditions {misplaced} are multi-turn only and cannot be used "
            f"with --exp {args.exp}; they apply to experiment 3"
        )
    if args.exp == "3" and not any(
        c in MULTI_TURN_ONLY or c.startswith("SK") for c in args.conditions
    ):
        p.error(
            "experiment 3 needs at least one of NL-anchor, NL-append, or an "
            "SK variant; the other conditions have no multi-turn behaviour"
        )

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Clients -----------------------------------------------------------
    client = OpenAICompatibleClient(base_url=args.base_url, model=args.model)

    if args.no_guided_decoding:
        verifier_client = client
        print("NOTE: guided decoding disabled — expect some unparsed verdicts.")
    else:
        # Guided decoding applied ONLY here: the verdict is a measurement
        # instrument shared identically by every condition, so constraining
        # it does not advantage any protocol.
        verifier_client = OpenAICompatibleClient(
            base_url=args.base_url,
            model=args.model,
            extra_body=verifier_response_format(),
        )

    # ---- Embedder ----------------------------------------------------------
    if args.embedder_device:
        os.environ["SK_EMBED_DEVICE"] = args.embedder_device
    embedder = EmbeddingBackend(model_name=args.embedder_path) \
        if args.embedder_path else EmbeddingBackend()

    # ---- Data --------------------------------------------------------------
    all_samples = load_samples(args.data_path)
    rng = random.Random(args.seed)
    filtered = [s for s in all_samples if s.get("level") in ("medium", "hard")]
    if not filtered:
        filtered = all_samples
    rng.shuffle(filtered)
    samples = filtered[: args.n_samples]
    print(f"Loaded {len(samples)} samples from {args.data_path}")
    print(f"Model: {args.model}  |  Guided decoding: {not args.no_guided_decoding}")

    exp1: list[Exp1Observation] = []
    exp2: list[Exp2Observation] = []
    exp3: list[Exp3Observation] = []

    if args.exp in ("1", "all"):
        print("\n[Exp 1] Compression + reconstruction")
        exp1 = run_experiment1(client, samples, args.conditions,
                               concurrency=args.concurrency)
        _save_jsonl(f"{args.output_dir}/exp1.jsonl", [asdict(r) for r in exp1])

    if args.exp in ("2", "all"):
        print("\n[Exp 2] Error propagation + detection")
        e2_conds = [c for c in args.conditions
                    if c in ("NL-free", "NL-CoT", "JSON-free", "SK")]
        exp2 = run_experiment2(client, verifier_client, samples, e2_conds)
        _save_jsonl(f"{args.output_dir}/exp2.jsonl", [asdict(r) for r in exp2])

    if args.exp in ("3", "all"):
        print("\n[Exp 3] Context drift")
        # Use exactly what was asked for. An earlier version hardcoded the
        # baselines and admitted only "SK", which silently dropped SK-index
        # from any sweep that requested it — the run completed and the missing
        # condition only showed up as an absent column in analysis.
        e3_conds = [
            c for c in args.conditions
            if c in MULTI_TURN_ONLY or c.startswith("SK")
        ]
        # When the caller gave only an SK variant, supply the baselines it
        # needs to be compared against rather than running an unpaired cell.
        if not any(c in MULTI_TURN_ONLY for c in e3_conds):
            e3_conds = e3_conds + ["NL-anchor", "NL-append"]
        print(f"  conditions: {', '.join(e3_conds)}")
        exp3 = run_experiment3(
            client, embedder, samples, e3_conds,
            num_turns=args.num_turns,
            distractor_tier=args.distractor_tier,
            on_topic_threshold=args.on_topic_threshold,
            off_topic_threshold=args.off_topic_threshold,
            concurrency=args.concurrency,
            prompt_budget_chars=args.prompt_budget_chars,
            remove_question=args.remove_question,
        )
        _save_jsonl(f"{args.output_dir}/exp3.jsonl", [asdict(r) for r in exp3])

    # ---- Run metadata for reproducibility ---------------------------------
    meta = {
        "model": args.model,
        "base_url": args.base_url,
        "data_path": args.data_path,
        "embedder_path": args.embedder_path,
        "n_samples": len(samples),
        "conditions": args.conditions,
        "num_turns": args.num_turns,
        "remove_question": args.remove_question,
        "distractor_tier": args.distractor_tier,
        "seed": args.seed,
        "concurrency": args.concurrency,
        "embedder_device": os.environ.get("SK_EMBED_DEVICE", "cpu"),
        "prompt_budget_chars": args.prompt_budget_chars,
        "num_turns": args.num_turns,
        "remove_question": args.remove_question,
        "guided_decoding": not args.no_guided_decoding,
        "on_topic_threshold": args.on_topic_threshold,
        "off_topic_threshold": args.off_topic_threshold,
        "scoring": "deterministic (scoring v3, no LLM judge)",
    }
    Path(f"{args.output_dir}/run_metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    report = generate_report(exp1, exp2, exp3, f"{args.output_dir}/report.txt")
    print("\n" + report)


if __name__ == "__main__":
    main()