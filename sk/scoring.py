"""
scoring.py — Deterministic, judge-free scoring for fully offline runs.

MOTIVATION
----------
The v2 design used an LLM-as-judge for three measurements: answer
recoverability (Exp 1), error propagation and detection (Exp 2), and
on-topic labeling (Exp 3). In a fully offline environment there is no
frontier API judge available, and using a local open-weight model as judge
introduces two problems a reviewer will raise:

  1. Evaluator contamination — if judge and main model share a family or
     architecture, judgments inherit shared preferences.
  2. Judge quality — open-weight judges have lower human agreement than
     frontier models, weakening every judge-dependent metric.

This module sidesteps both by replacing judge calls with deterministic
scoring wherever the experimental design already gives us ground truth.

KEY INSIGHT
-----------
We control the error injection. We know exactly which token was replaced
and what it was replaced with (see injection.InjectionRecord). That
makes propagation and detection *decidable by rule*, not by judgment:

  - propagated  := final answer is wrong vs gold (EM/F1) AND contains the
                   injected token (or a normalized variant of it)
  - detected    := verifier's structured verdict is "unverified" AND its
                   stated reason mentions the injected token or the
                   original token it replaced

Similarly, answer recoverability doesn't need a judge to *decide* — it needs
a model to *attempt reconstruction*, then HotpotQA's official EM/F1 to score
the attempt. The reconstruction step uses the same local model; the scoring
step is deterministic and matches the published benchmark protocol.

WHAT THIS BUYS US
-----------------
  - Fully offline: no API dependency anywhere in the pipeline.
  - Fully deterministic scoring: same inputs → same numbers, always.
  - Stronger review posture: "our metrics do not depend on LLM judgment"
    is a stronger claim than "our judge agreed with humans 87% of the time."
  - Reproducible for years: no model deprecation risk in the scoring path.

WHAT WE GIVE UP
---------------
  - Semantic nuance. A verifier that says "the birthplace claim looks
    wrong" without naming the city is scored as not-detected by the strict
    rule, though a human might credit it. We handle this by reporting BOTH
    a strict and a lenient variant of detection (see `detect_error`), and
    by validating a sample against human annotation.
  - Answer paraphrase in propagation. Mitigated by normalized matching plus
    the F1-based wrongness check rather than exact string equality.

All functions here are pure: no network, no model calls, no global state.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from metrics import exact_match, f1_score, normalize_answer


# =============================================================================
# 1. Answer recoverability without a judge (Experiment 1)
# =============================================================================


@dataclass
class RecoverabilityResult:
    """
    Outcome of scoring a reconstruction attempt.

    `reconstructed_answer` is what the reconstructor model produced when
    given only the protocol message. Scoring is then purely deterministic.
    """

    reconstructed_answer: str
    em: int
    f1: float
    recoverable_strict: bool   # EM == 1
    recoverable_lenient: bool  # F1 >= threshold


def score_recoverability(
    reconstructed_answer: str,
    gold_answer: str,
    lenient_f1_threshold: float = 0.6,
) -> RecoverabilityResult:
    """
    Score a reconstruction attempt against the gold answer using HotpotQA's
    official normalization and metrics. No LLM judgment involved.

    Two recoverability definitions are returned:
      - strict:  exact match after normalization (harsh, unambiguous)
      - lenient: token F1 >= threshold (tolerates paraphrase, still rule-based)

    Both are reported in the paper; the strict variant is the primary metric
    because it admits no interpretive latitude.
    """
    em = exact_match(reconstructed_answer, gold_answer)
    f1 = f1_score(reconstructed_answer, gold_answer)
    return RecoverabilityResult(
        reconstructed_answer=reconstructed_answer,
        em=em,
        f1=f1,
        recoverable_strict=bool(em),
        recoverable_lenient=f1 >= lenient_f1_threshold,
    )


RECONSTRUCT_SYSTEM_PROMPT = """You are given a message produced by a research agent, plus the original question it was working on.

Using ONLY the information contained in the message, state the answer to the question.

Rules:
- Answer with the shortest correct span (a name, date, place, or yes/no).
- Do not add explanation, punctuation, or preamble.
- If the message does not contain enough information, respond exactly: INSUFFICIENT"""


def build_reconstruct_prompt(question: str, protocol_message: str) -> str:
    """User prompt for the reconstruction step of Experiment 1."""
    return (
        f"QUESTION: {question}\n\n"
        f"AGENT MESSAGE:\n{protocol_message}\n\n"
        f"Answer:"
    )


def clean_reconstructed_answer(raw: str) -> str:
    """
    Strip common wrappers from a short-answer generation so that EM has a
    fair chance. We deliberately keep this conservative — over-aggressive
    cleaning would inflate EM artificially and is itself a form of
    measurement bias.

    The leading-qualifier list exists because piloting found correct answers
    being scored wrong for phrasing alone: against a gold of "flotilla", the
    reply "part of the flotilla" scored EM=0. That failure was not evenly
    distributed. Structured conditions describe their findings in the
    explanation field ("was part of a flotilla") and the extractor echoed that
    phrasing, so the penalty fell on exactly the conditions under test.

    Only semantically empty prefixes are removed. Anything that could change
    the referent is left alone, and F1 is reported alongside EM regardless.
    """
    if not raw:
        return ""
    s = raw.strip()
    # Remove leading label patterns like "Answer:" or "A:".
    s = re.sub(r"^(answer|a)\s*[:\-]\s*", "", s, flags=re.IGNORECASE)
    # Take only the first line — the prompt asked for a bare span.
    s = s.split("\n")[0].strip()
    # Strip surrounding quotes.
    s = s.strip('"').strip("'").strip()

    # Leading qualifiers that carry no content. Applied repeatedly because
    # they stack: "he was part of the flotilla".
    #
    # Stripping is refused when what remains is a single short word. Titles
    # begin with these same phrases — "One of Us", "It Is Well" — and reducing
    # them to "Us" or "Well" would turn a correct answer into a wrong one,
    # trading one measurement error for another. When the remainder is that
    # thin, the phrase was probably the answer rather than a preamble.
    QUALIFIERS = (
        r"(?:he|she|it|they)\s+(?:was|were|is|are)",
        r"part\s+of",
        r"one\s+of",
        r"a\s+member\s+of",
        r"on\s+(?:a|an|the)",
        r"in\s+(?:a|an|the)",
        r"the\s+answer\s+is",
        r"it\s+is",
    )
    changed = True
    while changed:
        changed = False
        for q in QUALIFIERS:
            new = re.sub(rf"^{q}\b\s*", "", s, flags=re.IGNORECASE).strip()
            if new == s or not new:
                continue
            # Refuse when the remainder is a lone short token — likelier a
            # truncated title than a stripped preamble.
            remainder_words = new.split()
            if len(remainder_words) == 1 and len(remainder_words[0]) <= 4:
                continue
            s = new
            changed = True

    # Trailing period only (keep internal punctuation, e.g. "St. Louis").
    s = re.sub(r"\.$", "", s).strip()
    return s


# =============================================================================
# 2. Error propagation without a judge (Experiment 2)
# =============================================================================


def _token_variants(token: str) -> set[str]:
    """
    Normalized surface variants of an injected token, so that matching is
    robust to casing and simple morphology without becoming so loose that
    it produces false positives.
    """
    if not token:
        return set()
    base = normalize_answer(token)
    variants = {base}
    # Possessive and plural forms that a model might emit.
    if base:
        variants.add(base + "s")
        variants.add(base + "es")
    return {v for v in variants if v}


def _mentions_token(text: str, token: str) -> bool:
    """
    Whether `text` mentions `token` (or a close variant), matched on
    normalized whole words to avoid substring accidents (e.g. 'Oslo'
    matching inside 'Oslobodjenje').
    """
    if not token or not text:
        return False
    norm_text = normalize_answer(text)
    words = set(norm_text.split())
    # Multi-word tokens: check as a contiguous phrase.
    for variant in _token_variants(token):
        if " " in variant:
            if variant in norm_text:
                return True
        elif variant in words:
            return True
    return False


@dataclass
class PropagationResult:
    """Deterministic propagation verdict for one (sample, condition) pair."""

    wrong_vs_gold: bool
    mentions_injected: bool
    mentions_original: bool
    propagated: bool
    final_em: int
    final_f1: float
    # Diagnostic: propagation under a looser wrongness criterion.
    propagated_lenient: bool = False


def score_propagation(
    final_answer: str,
    gold_answer: str,
    injected_token: str,
    original_token: str,
    wrongness_f1_threshold: float = 0.5,
) -> PropagationResult:
    """
    Decide whether the injected error propagated to the final answer.

    Definition (strict, primary):
        propagated := (final answer is wrong vs gold)
                      AND (final answer mentions the injected token)

    The first conjunct uses HotpotQA EM: the answer must not exactly match
    gold. The second conjunct is a whole-word normalized match against the
    token we ourselves injected, so it is fully determined by the
    experimental design rather than by interpretation.

    We additionally compute a lenient variant where wrongness is defined by
    F1 below `wrongness_f1_threshold` instead of EM failure. This matters for
    multi-token gold answers where an agent may produce a partially correct
    answer that still carries the injected error.

    `mentions_original` is reported as a diagnostic: if the final answer
    mentions the ORIGINAL (uncorrupted) token, the agent likely recovered
    the correct fact from elsewhere in the context, which is informative
    about how the protocol handles conflicting evidence.
    """
    em = exact_match(final_answer, gold_answer)
    f1 = f1_score(final_answer, gold_answer)
    wrong_strict = em == 0
    wrong_lenient = f1 < wrongness_f1_threshold

    mentions_inj = _mentions_token(final_answer, injected_token)
    mentions_orig = _mentions_token(final_answer, original_token)

    return PropagationResult(
        wrong_vs_gold=wrong_strict,
        mentions_injected=mentions_inj,
        mentions_original=mentions_orig,
        propagated=bool(wrong_strict and mentions_inj),
        propagated_lenient=bool(wrong_lenient and mentions_inj),
        final_em=em,
        final_f1=f1,
    )


# =============================================================================
# 3. Error detection without a judge (Experiment 2)
# =============================================================================

DetectionStrictness = Literal["strict", "lenient"]


@dataclass
class DetectionResult:
    """Deterministic detection verdict for one (sample, condition) pair."""

    verdict_parsed: bool          # Did we successfully read a verdict at all?
    flagged_unverified: bool      # verified == false
    reasons_text: str             # Concatenated reasons for auditing
    mentions_injected: bool       # Reasons name the injected token
    mentions_original: bool       # Reasons name the token it replaced
    detected_strict: bool         # unverified AND names injected/original token
    detected_lenient: bool        # unverified (regardless of specificity)


def score_detection(
    verified_flag: bool | None,
    reasons: list[str],
    injected_token: str,
    original_token: str,
    full_verifier_text: str = "",
) -> DetectionResult:
    """
    Decide whether the verifier detected the injected error.

    Definition (strict, primary):
        detected := verifier flagged the chain as unverified
                    AND its stated reasons name the injected token
                    (or the original token that was replaced)

    Definition (lenient, secondary):
        detected := verifier flagged the chain as unverified

    Reporting both is important. The strict rule is what the paper's claim
    needs — "SK surfaces errors as specific field inconsistencies" implies
    specificity, not just suspicion. The lenient rule guards against the
    strict rule under-crediting verifiers that identify the problem without
    quoting the exact token. If strict and lenient diverge sharply for one
    protocol, that divergence is itself a finding worth reporting.

    `full_verifier_text` is used as a fallback search space when the
    structured `reasons` list is empty but the verifier clearly wrote about
    the problem in prose.
    """
    if verified_flag is None:
        # Verdict could not be parsed. This is NOT the same as "not detected"
        # and must be excluded from the denominator, not counted as a failure.
        return DetectionResult(
            verdict_parsed=False,
            flagged_unverified=False,
            reasons_text="",
            mentions_injected=False,
            mentions_original=False,
            detected_strict=False,
            detected_lenient=False,
        )

    flagged = not verified_flag
    reasons_text = " ".join(reasons) if reasons else ""
    search_space = reasons_text if reasons_text else full_verifier_text

    mentions_inj = _mentions_token(search_space, injected_token)
    mentions_orig = _mentions_token(search_space, original_token)

    return DetectionResult(
        verdict_parsed=True,
        flagged_unverified=flagged,
        reasons_text=reasons_text[:500],
        mentions_injected=mentions_inj,
        mentions_original=mentions_orig,
        detected_strict=bool(flagged and (mentions_inj or mentions_orig)),
        detected_lenient=bool(flagged),
    )


# =============================================================================
# 4. On-topic scoring without a judge (Experiment 3, secondary metric)
# =============================================================================


@dataclass
class OnTopicResult:
    """Embedding-threshold on-topic label. No LLM judgment."""

    similarity: float
    label: Literal["on_topic", "partial", "off_topic"]


def score_on_topic(
    similarity_to_question: float,
    on_topic_threshold: float = 0.55,
    off_topic_threshold: float = 0.35,
) -> OnTopicResult:
    """
    Convert a question-response cosine similarity into a three-way label
    using fixed thresholds.

    Thresholds must be calibrated once against a small human-labeled sample
    and then FROZEN before the main runs. The defaults here are placeholders
    derived from typical mpnet similarity distributions on QA text; do not
    ship paper numbers without running `calibrate_on_topic_thresholds`
    first and reporting the calibration set.

    This replaces the LLM-judge on-topic labeling of the v2 design. Since
    drift (1 - cosine) is already the primary Experiment 3 metric, this
    secondary label is derived from the same signal and adds interpretability
    rather than new information — which is an honest thing to state in the
    paper rather than presenting it as independent corroboration.
    """
    if similarity_to_question >= on_topic_threshold:
        label: Literal["on_topic", "partial", "off_topic"] = "on_topic"
    elif similarity_to_question >= off_topic_threshold:
        label = "partial"
    else:
        label = "off_topic"
    return OnTopicResult(similarity=similarity_to_question, label=label)


def calibrate_on_topic_thresholds(
    labeled_similarities: list[tuple[float, str]],
) -> tuple[float, float]:
    """
    Given human-labeled (similarity, label) pairs where label is one of
    'on_topic' / 'partial' / 'off_topic', return (on_topic_threshold,
    off_topic_threshold) that maximize agreement.

    Simple grid search over candidate cut points. Intended to be run ONCE on
    a held-out calibration set (~50 turns), with the resulting thresholds
    frozen and reported in the paper.
    """
    if not labeled_similarities:
        return (0.55, 0.35)

    sims = sorted({round(s, 3) for s, _ in labeled_similarities})
    best = (0.55, 0.35)
    best_acc = -1.0

    for off_t in sims:
        for on_t in sims:
            if on_t <= off_t:
                continue
            correct = 0
            for sim, gold_label in labeled_similarities:
                pred = score_on_topic(sim, on_t, off_t).label
                if pred == gold_label:
                    correct += 1
            acc = correct / len(labeled_similarities)
            if acc > best_acc:
                best_acc = acc
                best = (on_t, off_t)

    return best


# =============================================================================
# 5. Verdict extraction that works with constrained decoding
# =============================================================================


VERIFIER_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "verified": {"type": "boolean"},
        "reasons": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["verified", "reasons"],
    "additionalProperties": False,
}
"""
JSON schema constraining verifier verdicts.

With constrained decoding active the model is restricted at the token level
to output matching this schema, which drives verdict parse failures to
essentially zero even on smaller open-weight models. That matters because an
unparsed verdict is a missing observation, not a negative detection, and a
high failure rate silently shrinks the denominator of the detection metric.
"""


def verifier_response_format() -> dict:
    """
    Build the request parameter that constrains output to VERIFIER_JSON_SCHEMA.

    Uses OpenAI's `response_format` with a `json_schema`, which is what current
    vLLM implements. The older vLLM-specific `guided_json` parameter has been
    removed; servers that no longer support it accept the request and then
    silently ignore the constraint, producing free prose. That failure mode is
    quiet — no error, just unparseable verdicts — so prefer this form, and
    verify empirically rather than trusting that a parameter took effect.
    """
    return {
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "verifier_verdict",
                "strict": True,
                "schema": VERIFIER_JSON_SCHEMA,
            },
        }
    }


SK_MESSAGE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        k: {"type": "string"}
        for k in (
            "topic", "who", "when", "where", "what", "how", "why",
            "cause", "effect", "goal", "conditions",
            "evidence", "explanation", "background", "context",
        )
    },
    "additionalProperties": False,
}
"""
Schema for SK protocol messages. Constrains the model to the six Semantic
Keys' field set and forbids invented fields.

IMPORTANT CAVEAT for the paper: applying constrained decoding to the SK condition
but not to the NL conditions would be an asymmetric intervention and would
confound the comparison. Two defensible options:

  (a) Apply constrained decoding ONLY to the verifier verdict turn, which is a
      measurement instrument shared identically by all conditions, and leave
      the agents' primary outputs unconstrained. (Recommended.)
  (b) Apply constrained decoding to both SK and JSON-free conditions, and report
      it as part of what each protocol *is* — since both are structured
      protocols, constrained decoding is a fair implementation of both.

Option (a) is the default in the offline runner. Whichever is chosen must be
stated explicitly in the paper's implementation section.
"""