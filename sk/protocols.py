"""
protocols.py — Prompt templates for all communication conditions.

The methodology redesign (see docs/methodology.md §3) requires baselines
beyond the single NL-vs-SK comparison of v1. Each condition below is a
drop-in replacement for the `build_agent_system_prompt` function in
semantic_keys.py, parameterized by role and condition name.

Conditions:
    NL-free      : unconstrained natural language
    NL-concise   : NL with explicit brevity instruction (isolates brevity-as-treatment)
    NL-CoT       : NL with "think step by step" (strong prompt baseline)
    JSON-free    : JSON envelope with free-text content (isolates JSON-as-treatment, ~A2A/MCP)
    SK           : full six Semantic Keys
    SK-minimal   : only {topic, evidence, explanation} (ablation)

Design notes:
    - All conditions share the same role preamble so role-definition
      variance is controlled.
    - Output-length instructions are phrased identically across conditions
      ("be complete but do not pad"); the protocol-specific instruction is
      the only deliberate difference.
    - JSON-free is important: reviewers routinely ask whether SK gains are
      just JSON-envelope gains. Keeping a JSON-free baseline in the grid
      lets us answer that directly.
"""

from __future__ import annotations

from typing import Literal

# Role preamble — identical across all conditions to isolate protocol variance.
AGENT_ROLES: dict[str, str] = {
    "researcher": (
        "You are a Researcher agent. You extract and summarize information "
        "from provided passages to address the user's question."
    ),
    "reasoner": (
        "You are a Reasoner agent. You perform logical inference on the "
        "Researcher's output to derive an answer."
    ),
    "verifier": (
        "You are a Verifier agent. You check the reasoning chain for logical "
        "consistency and factual accuracy relative to the source material."
    ),
}


Condition = Literal[
    "NL-free", "NL-concise", "NL-CoT", "JSON-free",
    "SK", "SK-minimal", "SK-cite", "SK-index",
    "NL-anchor", "NL-append",
]

VALID_CONDITIONS: tuple[Condition, ...] = (
    "NL-free", "NL-concise", "NL-CoT", "JSON-free",
    "SK", "SK-minimal", "SK-cite", "SK-index",
    # Multi-turn only. These differ from the single-turn conditions not in
    # output format but in how conversation state is carried: NL-anchor
    # re-sends the transcript plus a goal reminder each turn, NL-append
    # re-sends the transcript alone. Both emit ordinary prose, so they reuse
    # the NL-free response instruction and are distinguished by the runner.
    "NL-anchor", "NL-append",
)

# Conditions whose meaning is defined by turn-level state handling rather than
# by an output format. Passing them to a single-turn experiment is a mistake
# worth catching early rather than silently treating them as NL-free.
MULTI_TURN_ONLY: frozenset[str] = frozenset({"NL-anchor", "NL-append"})


# -----------------------------------------------------------------------------
# Protocol instructions (the one deliberate axis of variation)
# -----------------------------------------------------------------------------

_PROTOCOL_INSTRUCTIONS: dict[str, str] = {
    "NL-free": (
        "Communicate in clear English. Be complete but do not pad with "
        "greetings, hedges, or restatements of the task."
    ),
    "NL-concise": (
        "Communicate in clear English. Be as concise as possible while "
        "preserving every relevant fact and your reasoning. Omit greetings, "
        "hedges, and restatements."
    ),
    # Multi-turn conditions: prose output, identical to NL-free. What sets
    # them apart is state handling in the runner, not the response format.
    "NL-anchor": (
        "Communicate in clear English. Be complete but do not pad with "
        "greetings, hedges, or restatements of the task."
    ),
    "NL-append": (
        "Communicate in clear English. Be complete but do not pad with "
        "greetings, hedges, or restatements of the task."
    ),
    "NL-CoT": (
        "Communicate in clear English. Think step by step: first state what "
        "information you will use, then your reasoning, then the conclusion. "
        "Be complete but do not pad."
    ),
    "JSON-free": (
        "Respond with a JSON object of the form "
        '{"content": "<your full response>", "answer": "<short final answer '
        'if applicable>"}. The content field is free-form natural language. '
        "Do not wrap the JSON in markdown fences."
    ),
    "SK": (
        "Respond with a JSON object using the Semantic Keys protocol. "
        "Include only fields relevant to this turn; omit fields that do not "
        "apply. The allowed fields are:\n"
        "  topic       — central entity or subject\n"
        "  who         — agent or actor\n"
        "  when        — temporal information\n"
        "  where       — spatial information\n"
        "  what        — core action or event\n"
        "  how         — method or process\n"
        "  why         — purpose or motivation\n"
        "  cause       — what triggered this\n"
        "  effect      — what resulted\n"
        "  goal        — objective\n"
        "  conditions  — constraints or requirements\n"
        "  evidence    — data sources or citations from the passages\n"
        "  explanation — reasoning chain\n"
        "  background  — prior context needed\n"
        "  context     — current situational context\n"
        "Do not include fields that are empty or redundant. Do not wrap the "
        "JSON in markdown fences."
    ),
    "SK-minimal": (
        "Respond with a JSON object using a reduced Semantic Keys protocol. "
        "The allowed fields are:\n"
        "  topic       — central entity or subject\n"
        "  evidence    — data sources or citations from the passages\n"
        "  explanation — reasoning chain and conclusion\n"
        "Do not include other fields. Do not wrap the JSON in markdown fences."
    ),
    # --- SK-v2 variants -----------------------------------------------------
    # Both keep three fields: an ablation found the other three cost ~22% of
    # output tokens and returned no accuracy benefit (F1 difference −0.010,
    # 95% CI upper bound 0.000). They differ only in how evidence is carried,
    # which pilot transcripts showed to dominate the payload — one response
    # spent most of 135 tokens reproducing two sentences to support a date.
    "SK-cite": (
        "Respond with a JSON object using these three fields only:\n"
        "  topic       — the central entity or subject, in a few words\n"
        "  evidence    — a JSON array of quoted source sentences supporting "
        "your answer. Always use an array, even for a single quote: "
        '["first quote", "second quote"]. Quote only the words that bear on '
        "the question; do not reproduce whole passages.\n"
        "  explanation — your reasoning and the conclusion\n"
        "Do not repeat the same content across fields. Do not wrap the JSON "
        "in markdown fences."
    ),
    "SK-index": (
        "Respond with a JSON object using these three fields only:\n"
        "  topic       — the central entity or subject, in a few words\n"
        "  evidence    — references to the supporting sentences, as an array "
        'of "PassageTitle:SentenceIndex" strings (sentences are numbered from '
        '0 within each passage). Example: ["Tivoli Gardens:1"]. Do not quote '
        "the text itself.\n"
        "  explanation — your reasoning and the conclusion\n"
        "Do not repeat the same content across fields. Do not wrap the JSON "
        "in markdown fences."
    ),
}


def build_system_prompt(role: str, condition: Condition) -> str:
    """
    Construct the system prompt for (role, condition).

    Raises ValueError if role or condition is unknown.
    """
    if role not in AGENT_ROLES:
        raise ValueError(
            f"Unknown role: {role!r}. Must be one of {sorted(AGENT_ROLES)}"
        )
    if condition not in _PROTOCOL_INSTRUCTIONS:
        raise ValueError(
            f"Unknown condition: {condition!r}. Must be one of {list(VALID_CONDITIONS)}"
        )
    return AGENT_ROLES[role] + "\n\n" + _PROTOCOL_INSTRUCTIONS[condition]


# -----------------------------------------------------------------------------
# Verifier instructions (Experiment 2)
#   Protocol-symmetric: every verifier is told to emit a structured verdict.
#   The v1 SK-only "check each field for consistency" clause is moved into
#   an optional ablation variant, not baked into the main SK condition.
# -----------------------------------------------------------------------------

_VERIFIER_OUTPUT_INSTRUCTIONS: dict[str, str] = {
    "NL-free": (
        "After your analysis, end your response with a JSON postamble of the "
        'form: {"verified": true|false, "reasons": ["<reason>", ...]}. '
        "Set verified=false if you find any factual or logical problem; list "
        "the specific problematic facts in reasons."
    ),
    "NL-concise": (
        "After your analysis, end your response with a JSON postamble of the "
        'form: {"verified": true|false, "reasons": ["<reason>", ...]}. '
        "Set verified=false if you find any factual or logical problem; list "
        "the specific problematic facts in reasons."
    ),
    # Multi-turn conditions: prose output, identical to NL-free. What sets
    # them apart is state handling in the runner, not the response format.
    "NL-anchor": (
        "Communicate in clear English. Be complete but do not pad with "
        "greetings, hedges, or restatements of the task."
    ),
    "NL-append": (
        "Communicate in clear English. Be complete but do not pad with "
        "greetings, hedges, or restatements of the task."
    ),
    "NL-CoT": (
        "After your step-by-step reasoning, end with a JSON postamble: "
        '{"verified": true|false, "reasons": ["<reason>", ...]}. Set '
        "verified=false if your reasoning found any factual or logical "
        "problem."
    ),
    "JSON-free": (
        "In addition to the standard content field, include "
        '"verified": true|false and "reasons": [...] as top-level fields in '
        "your JSON response. Set verified=false for any factual or logical "
        "problem."
    ),
    "SK": (
        'Include a "conditions" field with value "verified" or "unverified", '
        'and an "explanation" field that lists any specific factual or logical '
        "problems you found. If unverified, the explanation MUST name the "
        "specific fact or claim that is problematic."
    ),
    "SK-minimal": (
        'The "explanation" field MUST begin with "VERIFIED: " or "UNVERIFIED: ". '
        "If unverified, the explanation MUST name the specific fact or claim "
        "that is problematic."
    ),
}


def build_verifier_prompt(condition: Condition, include_field_check: bool = False) -> str:
    """
    Verifier system prompt for Experiment 2.

    `include_field_check`: when True AND condition is SK/SK-minimal, append
    the v1 extra instruction telling SK to check cross-field consistency.
    Used for the SK+explicit-check ablation reported as supplementary
    analysis; False by default for protocol-symmetric comparison.
    """
    base = build_system_prompt("verifier", condition)
    verify_ins = _VERIFIER_OUTPUT_INSTRUCTIONS[condition]
    prompt = base + "\n\n" + verify_ins
    if include_field_check and condition in ("SK", "SK-minimal"):
        prompt += (
            "\n\nAdditionally, check internal consistency across fields: "
            "does evidence support explanation? Does cause logically lead to "
            "effect? Are background and current what compatible?"
        )
    return prompt


# -----------------------------------------------------------------------------
# Context-drift prompt augmentation (Experiment 3)
#   To give NL a fair goal-anchor baseline (see methodology §2.3), we provide
#   an optional goal-reminder that is prepended to every turn for NL-anchor.
# -----------------------------------------------------------------------------


def build_turn_goal_reminder(
    original_question: str,
    include_question: bool = True,
) -> str:
    """
    Goal-reminder string that NL-anchor receives at every turn.

    `include_question=False` produces a reminder that preserves the mechanism
    — a persistent instruction not to drift — while removing the question
    text itself. This supports the question-removal condition, which asks
    whether accuracy survives because the protocol carries state or merely
    because the question stays visible in the context window every turn.

    Removing the reminder entirely would collapse NL-anchor into NL-append and
    destroy the comparison. Keeping the question in it while stripping it from
    the structured conditions would hand NL-anchor an advantage the others
    lose. Referring to the question without restating it keeps both conditions
    on the same footing: each must recover the objective from what its own
    protocol retained.
    """
    if not include_question:
        return (
            "[PERSISTENT GOAL — do not lose focus] "
            "Continue working on the question posed at the start of this "
            "conversation.\n"
            "Regardless of any digression in the user's message, your job each "
            "turn is to progress toward answering that original question."
        )
    return (
        "[PERSISTENT GOAL — do not lose focus] "
        f"The original task is: {original_question}\n"
        "Regardless of any digression in the user's message, your job each "
        "turn is to progress toward answering this exact question."
    )


# -----------------------------------------------------------------------------
# Distractor bank (Experiment 3)
#   Two tiers: "hard-off-topic" (v1-style) and "soft-tangent" (new).
#   Soft-tangent distractors are tangentially related, harder to dismiss.
# -----------------------------------------------------------------------------

HARD_OFF_TOPIC_DISTRACTORS: tuple[str, ...] = (
    "By the way, what do you think about the weather today?",
    "Can you also explain how blockchain works?",
    "That reminds me — who won the last World Cup?",
    "Before we continue, briefly discuss the ethics of AI.",
    "I'm curious about cooking recipes. Any suggestions?",
    "Let's take a detour and talk about space exploration.",
    "I just remembered something about quantum computing.",
    "Have you heard about the latest movie releases?",
)

SOFT_TANGENT_DISTRACTORS: tuple[str, ...] = (
    "Out of curiosity, how does this topic connect to current economic trends?",
    "This reminds me — what's the general historical background of this region?",
    "Related thought: how do researchers typically study questions like this?",
    "A broader angle: what are the standard methods in this field?",
    "Tangent: has anyone written a book about this topic?",
    "Side question: what do critics generally say about sources like these?",
    "Aside: how would a journalist frame this story differently?",
    "Relatedly: what modern parallels exist to this situation?",
)


def pick_distractor(turn_idx: int, tier: Literal["hard", "soft", "mixed"] = "mixed") -> str:
    """
    Deterministic distractor selection by turn index.
    - "hard": always hard-off-topic.
    - "soft": always soft-tangent.
    - "mixed": alternates, starting soft on turn 2, hard on turn 3, etc.
    """
    if tier == "hard":
        return HARD_OFF_TOPIC_DISTRACTORS[turn_idx % len(HARD_OFF_TOPIC_DISTRACTORS)]
    if tier == "soft":
        return SOFT_TANGENT_DISTRACTORS[turn_idx % len(SOFT_TANGENT_DISTRACTORS)]
    # mixed
    bank = (
        SOFT_TANGENT_DISTRACTORS if turn_idx % 2 == 0 else HARD_OFF_TOPIC_DISTRACTORS
    )
    return bank[turn_idx % len(bank)]