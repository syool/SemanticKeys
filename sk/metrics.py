"""
metrics.py — Redesigned measurement module for the Semantic Keys experiments.

Replaces v1's lexical-overlap and substring-match measurements with:
  - HotpotQA official EM/F1 (normalized)
  - Semantic drift via sentence embeddings (cosine distance)
  - Bootstrap confidence intervals (paired and unpaired)
  - LLM-as-judge wrappers with structured output and caching

Every function here is deterministic given its inputs (embeddings and judge
calls are cached). No measurement here mutates state outside its return value.

References:
  - HotpotQA normalization: Yang et al. 2018 (EMNLP). See `normalize_answer`.
  - Bootstrap CI: Efron & Tibshirani 1993.
  - McNemar's test: via scipy.stats.

This module intentionally has no dependency on the SK protocol itself; it is
protocol-agnostic and can score any string output against any gold answer.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

# Optional heavy deps — imported lazily so metrics stays importable without them.
# numpy is effectively required; sentence_transformers and scipy are required for
# the drift and statistics helpers respectively.


# =============================================================================
# 1. HotpotQA answer normalization and EM/F1
#    Direct port of the official HotpotQA evaluation script, kept self-contained
#    so reviewers can verify it matches Yang et al. 2018 exactly.
# =============================================================================

_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_answer(s: str) -> str:
    """HotpotQA normalization: lowercase, strip articles, remove punct, collapse ws."""
    if s is None:
        return ""
    s = s.lower()
    s = s.translate(_PUNCT_TABLE)
    s = _ARTICLES_RE.sub(" ", s)
    s = " ".join(s.split())
    return s


def exact_match(prediction: str, gold: str) -> int:
    """1 if normalized prediction equals normalized gold, else 0."""
    return int(normalize_answer(prediction) == normalize_answer(gold))


def f1_score(prediction: str, gold: str) -> float:
    """Token-level F1 on normalized strings. 0.0 if either side is empty unless both are."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    # Multiset intersection — HotpotQA counts token multiplicities.
    from collections import Counter
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)




# =============================================================================
# 2. Semantic drift via sentence embeddings
#    Primary drift metric for Experiment 3. Replaces v1's lexical-overlap
#    `compute_drift` function.
# =============================================================================


class EmbeddingBackend:
    """
    Thin wrapper around a sentence embedder with on-disk caching.

    Default backend: sentence-transformers/all-mpnet-base-v2 (768-dim, strong
    on semantic similarity, ~420MB model). Swap the model name via env var
    SK_EMBED_MODEL if needed.

    Usage:
        be = EmbeddingBackend()
        v1 = be.embed("some text")
        sim = cosine(v1, be.embed("other text"))
    """

    def __init__(
        self,
        model_name: str | None = None,
        cache_dir: str | Path | None = None,
    ):
        self.model_name = model_name or os.environ.get(
            "SK_EMBED_MODEL", "sentence-transformers/all-mpnet-base-v2"
        )
        # Default beside the repository rather than an absolute path. An
        # earlier hardcoded development path meant the cache directory was
        # recreated empty on every other machine, so nothing was ever reused.
        if cache_dir is None:
            cache_dir = Path(__file__).resolve().parent.parent / ".embed_cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = None  # lazy load

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            # Device selection matters more than it looks. vLLM reserves most
            # of the GPU up front (--gpu-memory-utilization 0.90), so a second
            # model loading onto the same device competes for what little is
            # left. A sweep died four cells in with CUBLAS_STATUS_ALLOC_FAILED
            # from inside the embedder, at the turn count where the number of
            # texts per batch first got large.
            #
            # CPU is the safe default here: embedding is a small fraction of
            # total runtime next to generation, so the slowdown is negligible
            # against the cost of losing a run. Set SK_EMBED_DEVICE=cuda to
            # override when nothing else holds the GPU.
            device = os.environ.get("SK_EMBED_DEVICE", "cpu")

            # local_files_only when the name is a directory that exists: a
            # sweep loads the embedder once per cell, and resolving through
            # the hub each time turns a transient network failure into a
            # dead run hours in. Passing a path should never touch the
            # network at all.
            path = Path(self.model_name)
            if path.is_dir():
                self._model = SentenceTransformer(
                    str(path), local_files_only=True, device=device
                )
            else:
                self._model = SentenceTransformer(self.model_name, device=device)
        return self._model

    def _cache_path(self, text: str):
        h = hashlib.sha256(
            (self.model_name + "\0" + text).encode("utf-8")
        ).hexdigest()
        return self.cache_dir / f"{h}.npy"

    def embed(self, text: str):
        import numpy as np

        p = self._cache_path(text)
        if p.exists():
            return np.load(p)
        vec = self._load().encode(text, convert_to_numpy=True, normalize_embeddings=True)
        np.save(p, vec)
        return vec

    def embed_batch(self, texts: Sequence[str]):
        import numpy as np

        # Split into cached and uncached.
        results: list = [None] * len(texts)
        to_compute: list[tuple[int, str]] = []
        for i, t in enumerate(texts):
            p = self._cache_path(t)
            if p.exists():
                results[i] = np.load(p)
            else:
                to_compute.append((i, t))
        if to_compute:
            model = self._load()
            idxs, raw = zip(*to_compute)
            vecs = model.encode(
                list(raw), convert_to_numpy=True, normalize_embeddings=True, batch_size=32
            )
            for j, idx in enumerate(idxs):
                results[idx] = vecs[j]
                np.save(self._cache_path(texts[idx]), vecs[j])
        return np.stack(results)


def cosine(u, v) -> float:
    """Cosine similarity. Assumes normalized inputs but doesn't require it."""
    import numpy as np

    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    denom = (np.linalg.norm(u) * np.linalg.norm(v)) or 1e-12
    return float(np.dot(u, v) / denom)


@dataclass
class DriftSeries:
    """
    Drift curve for one multi-turn sample.

    Attributes:
        per_turn: list of (1 - cos(q, r_t)) values, one per turn.
        mean_drift: mean over turns considered "evaluation turns" (default 3..end).
        warmup_turns: number of initial turns excluded from mean (default 2).
    """

    per_turn: list[float]
    warmup_turns: int = 2

    @property
    def mean_drift(self) -> float:
        tail = self.per_turn[self.warmup_turns :]
        return sum(tail) / len(tail) if tail else 0.0




# =============================================================================
# 3. LLM-as-judge harness with structured output and caching
#    Used for: A-SRE answer recoverability check (Exp 1), propagation-vs-error
#    judgment (Exp 2), on-topic labeling (Exp 3 secondary).
# =============================================================================


@dataclass
class JudgeDecision:
    """Structured judge verdict."""

    verdict: dict[str, Any]
    raw_response: str
    model: str
    cached: bool = False


class JudgeClient:
    """
    Wraps an LLM client (AnthropicClient / SimulatedClient from client.py)
    for blinded, structured judgments. Results are cached by prompt hash so
    re-running experiments does not re-pay for judge calls.

    The judge is expected to return a JSON object matching a caller-specified
    schema. If the model returns surrounding text, we extract the first
    JSON object; if extraction fails, the verdict is {"_parse_error": true}
    and callers must handle that explicitly rather than silently treating
    parse failures as one class or another.
    """

    def __init__(
        self,
        llm_client,
        cache_dir: str | Path | None = None,
    ):
        self.client = llm_client
        # Same fix as EmbeddingBackend: default relative to the repository so
        # the cache survives on any machine, not just the one it was written on.
        if cache_dir is None:
            cache_dir = Path(__file__).resolve().parent.parent / ".judge_cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = getattr(llm_client, "model", "unknown")

    def _cache_path(self, key: str):
        h = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{h}.json"

    def judge(
        self,
        system: str,
        user: str,
        max_tokens: int = 400,
    ) -> JudgeDecision:
        cache_key = f"{self.model_name}\0{system}\0{user}"
        cp = self._cache_path(cache_key)
        if cp.exists():
            d = json.loads(cp.read_text())
            return JudgeDecision(
                verdict=d["verdict"],
                raw_response=d["raw_response"],
                model=d["model"],
                cached=True,
            )

        resp = self.client.call(system, user, max_tokens=max_tokens)
        verdict = _extract_first_json(resp.content)
        decision = JudgeDecision(
            verdict=verdict, raw_response=resp.content, model=self.model_name
        )
        cp.write_text(
            json.dumps(
                {"verdict": verdict, "raw_response": resp.content, "model": self.model_name},
                ensure_ascii=False,
            )
        )
        return decision


def _extract_first_json(text: str) -> dict[str, Any]:
    """
    Pull the first balanced JSON object out of a string. Tolerates code fences,
    leading/trailing prose, and partial markdown. Returns {"_parse_error": True,
    "raw": text} if nothing parseable is found.
    """
    if not text:
        return {"_parse_error": True, "raw": ""}
    # Strip common markdown fences.
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z]*\n", "", stripped)
        stripped = re.sub(r"\n```\s*$", "", stripped)
    # Scan for the first {...} with balanced braces.
    start = stripped.find("{")
    if start == -1:
        return {"_parse_error": True, "raw": text}
    depth = 0
    for i in range(start, len(stripped)):
        c = stripped[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                candidate = stripped[start : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    return {"_parse_error": True, "raw": text}
    return {"_parse_error": True, "raw": text}


# --- Judge prompt templates ----------------------------------------------------
# Each prompt is versioned. When a prompt changes, bump the version; the cache
# key includes the prompt text, so stale judgments are automatically recomputed.

JUDGE_ANSWER_RECOVERABILITY_V1_SYSTEM = """You are evaluating whether a given message preserves enough information to answer a factual question.

You will receive:
- QUESTION: a multi-hop factual question
- MESSAGE: a possibly-structured message produced by a research agent
- GOLD ANSWER: the reference answer

Your task: judge whether the MESSAGE by itself (without access to any external source) contains enough information for a downstream agent to arrive at the GOLD ANSWER.

Respond with a JSON object only, no prose:
{"recoverable": true|false, "extracted_answer": "<the answer you would give based on MESSAGE>", "reasoning": "<1-2 sentences>"}"""


JUDGE_PROPAGATION_V1_SYSTEM = """You are evaluating whether an agent's final answer shows evidence of being misled by an injected factual error.

You will receive:
- QUESTION: the original question
- GOLD ANSWER: the correct answer
- INJECTED ERROR: the false claim inserted into the agent's input
- FINAL ANSWER: the agent's output

Judge two things independently:
1. Is the FINAL ANSWER wrong relative to GOLD ANSWER? (Use semantic judgment, not string match.)
2. Does the FINAL ANSWER reflect the INJECTED ERROR specifically? (e.g., naming the wrong entity that was injected, or repeating the false date.)

Respond with a JSON object only:
{"wrong_vs_gold": true|false, "matches_injected_error": true|false, "reasoning": "<1-2 sentences>"}"""


JUDGE_DETECTION_V1_SYSTEM = """You are evaluating whether a verifier agent detected an injected factual error.

You will receive:
- QUESTION: the original question
- INJECTED ERROR: the false claim that was planted in the input
- VERIFIER OUTPUT: what the verifier agent said

Judge whether the VERIFIER OUTPUT identifies, flags, or contradicts the specific INJECTED ERROR. A generic "something seems off" is NOT detection; the verifier must point at the actual poisoned fact (by content or by location) for this to count.

Respond with a JSON object only:
{"detected": true|false, "specificity": "specific"|"generic"|"none", "reasoning": "<1-2 sentences>"}"""


JUDGE_ON_TOPIC_V1_SYSTEM = """You are labeling whether an agent's turn response stays on topic relative to the user's original question.

You will receive:
- ORIGINAL QUESTION: what the user originally asked
- TURN RESPONSE: what the agent said this turn

Label the response as:
- "on_topic": substantively addresses or progresses the original question
- "partial": mixes on-topic content with off-topic digressions
- "off_topic": primarily discusses something other than the original question

Respond with a JSON object only:
{"label": "on_topic"|"partial"|"off_topic", "reasoning": "<1 sentence>"}"""


# =============================================================================
# 4. Bootstrap confidence intervals
# =============================================================================


def bootstrap_ci(
    values: Sequence[float],
    stat: Callable[[Sequence[float]], float] = None,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Return (point, lower, upper) for a sample statistic via percentile bootstrap.

    Default stat is the mean. `seed` makes results reproducible across runs.
    """
    import numpy as np

    if not values:
        return (0.0, 0.0, 0.0)
    if stat is None:
        stat = lambda xs: float(np.mean(xs))
    rng = np.random.default_rng(seed)
    arr = np.asarray(list(values), dtype=float)
    n = len(arr)
    point = stat(arr)
    resamples = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        resamples[i] = stat(arr[idx])
    alpha = (1 - confidence) / 2
    lower = float(np.quantile(resamples, alpha))
    upper = float(np.quantile(resamples, 1 - alpha))
    return (float(point), lower, upper)


def paired_bootstrap_ci(
    a: Sequence[float],
    b: Sequence[float],
    stat: Callable[[Sequence[float]], float] = None,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Bootstrap CI on the difference stat(a) - stat(b) with paired resampling.
    Assumes a[i] and b[i] are paired observations from the same sample i.
    """
    import numpy as np

    assert len(a) == len(b), "paired_bootstrap_ci requires equal-length sequences"
    if not a:
        return (0.0, 0.0, 0.0)
    if stat is None:
        stat = lambda xs: float(np.mean(xs))
    rng = np.random.default_rng(seed)
    arr_a = np.asarray(list(a), dtype=float)
    arr_b = np.asarray(list(b), dtype=float)
    n = len(arr_a)
    point = stat(arr_a) - stat(arr_b)
    diffs = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        diffs[i] = stat(arr_a[idx]) - stat(arr_b[idx])
    alpha = (1 - confidence) / 2
    return (float(point), float(np.quantile(diffs, alpha)), float(np.quantile(diffs, 1 - alpha)))


# =============================================================================
# 5. Hypothesis tests
# =============================================================================


def wilcoxon_paired(a: Sequence[float], b: Sequence[float]) -> dict[str, float]:
    """
    Wilcoxon signed-rank test on paired samples. Returns {statistic, pvalue}.
    Zero-difference pairs are dropped (scipy default 'wilcox' mode).
    """
    from scipy import stats

    assert len(a) == len(b)
    res = stats.wilcoxon(list(a), list(b), zero_method="wilcox")
    return {"statistic": float(res.statistic), "pvalue": float(res.pvalue)}


def mcnemar_paired(
    paired_outcomes: Sequence[tuple[bool, bool]],
) -> dict[str, float]:
    """
    McNemar's test on paired binary outcomes. Input: list of (a_positive, b_positive).
    Returns {b01, b10, statistic, pvalue} where b01/b10 are the off-diagonal counts.

    Uses statsmodels when available. Falls back to computing the test directly
    from the discordant pairs otherwise: McNemar's exact test is a two-sided
    binomial test of b10 against b10+b01 at p=0.5, and the asymptotic form is
    a chi-squared statistic with continuity correction. The fallback keeps the
    analysis runnable on a machine with scipy but not statsmodels rather than
    failing, which previously turned every p-value into nan and made a real
    effect look absent.
    """
    b00 = b01 = b10 = b11 = 0
    for a, b in paired_outcomes:
        if a and b:
            b11 += 1
        elif a and not b:
            b10 += 1
        elif not a and b:
            b01 += 1
        else:
            b00 += 1

    n_disc = b01 + b10
    exact = n_disc < 25

    try:
        from statsmodels.stats.contingency_tables import mcnemar

        table = [[b11, b10], [b01, b00]]
        res = mcnemar(table, exact=exact, correction=not exact)
        statistic, pvalue = float(res.statistic), float(res.pvalue)
    except ImportError:
        from scipy.stats import binomtest, chi2

        if n_disc == 0:
            # No discordant pairs: the conditions never disagreed, so there is
            # no evidence of a difference in either direction.
            statistic, pvalue = 0.0, 1.0
        elif exact:
            statistic = float(min(b01, b10))
            pvalue = float(binomtest(b10, n_disc, 0.5).pvalue)
        else:
            statistic = (abs(b10 - b01) - 1) ** 2 / n_disc
            pvalue = float(chi2.sf(statistic, df=1))

    return {
        "b01": b01,
        "b10": b10,
        "b00": b00,
        "b11": b11,
        "statistic": statistic,
        "pvalue": pvalue,
    }


def holm_bonferroni(pvalues: Sequence[float], alpha: float = 0.05) -> list[bool]:
    """
    Holm-Bonferroni stepdown. Returns list of reject/not-reject decisions
    aligned with input order.
    """
    m = len(pvalues)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvalues[i])
    decisions = [False] * m
    for rank, idx in enumerate(order):
        threshold = alpha / (m - rank)
        if pvalues[idx] <= threshold:
            decisions[idx] = True
        else:
            break  # Holm: stop at first non-reject
    return decisions


# =============================================================================
# 6. Convenience aggregation
# =============================================================================


@dataclass
class ScoredRun:
    """A single scored observation across any experiment."""

    sample_id: str
    condition: str  # "NL-free", "NL-concise", "NL-CoT", "JSON-free", "SK", ...
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)