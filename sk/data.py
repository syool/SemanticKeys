"""
data.py — HotpotQA loading with HuggingFace primary + local synthetic fallback.

The methodology redesign (docs/methodology_v2.md §4) specifies HotpotQA as the
primary dataset. In compute environments where huggingface.co is reachable,
we load via `datasets.load_dataset("hotpot_qa", "distractor", split="validation")`.

If HF is unreachable (restricted network sandboxes), we fall back to a
handcrafted 30-sample synthetic multi-hop set that follows the HotpotQA schema
exactly, so downstream code does not have to branch on provenance.

Schema (matches HotpotQA dev-distractor):
    {
        "_id": str,
        "question": str,
        "answer": str,
        "type": "bridge" | "comparison",
        "level": "easy" | "medium" | "hard",
        "context": {
            "title": [str, ...],           # passage titles
            "sentences": [[str, ...], ...]  # sentences per passage
        },
        "supporting_facts": {
            "title": [str, ...],     # titles of passages with supporting facts
            "sent_id": [int, ...]    # sentence indices within those passages
        }
    }
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


SplitChoice = Literal["validation", "train"]


def load_hotpotqa(
    n: int = 100,
    split: SplitChoice = "validation",
    difficulty: tuple[str, ...] = ("medium", "hard"),
    type_balance: bool = True,
    seed: int = 42,
    allow_fallback: bool = True,
) -> tuple[list[dict], str]:
    """
    Load `n` HotpotQA samples.

    Returns (samples, source) where source is "huggingface" or "synthetic_fallback".

    Args:
        n: number of samples to return
        split: HF split to draw from
        difficulty: keep only samples with `level` in this set
        type_balance: if True, aim for 50/50 bridge vs comparison
        seed: RNG seed
        allow_fallback: if False, raise instead of using synthetic fallback
    """
    try:
        return _load_from_huggingface(n, split, difficulty, type_balance, seed)
    except Exception as e:
        if not allow_fallback:
            raise
        print(f"[data] HF load failed ({type(e).__name__}: {e}); using synthetic fallback.")
        samples = _build_synthetic_multihop_set(n, seed=seed)
        return samples, "synthetic_fallback"


def _load_from_huggingface(
    n: int,
    split: str,
    difficulty: tuple[str, ...],
    type_balance: bool,
    seed: int,
) -> tuple[list[dict], str]:
    from datasets import load_dataset

    ds = load_dataset("hotpot_qa", "distractor", split=split)
    rng = random.Random(seed)

    # HF's HotpotQA has a slightly different schema: `supporting_facts` is a
    # single dict with parallel "title" and "sent_id" lists — same as ours.
    filtered = [
        _normalize_hf_record(rec)
        for rec in ds
        if rec.get("level") in difficulty
    ]

    if type_balance:
        bridge = [s for s in filtered if s["type"] == "bridge"]
        compare = [s for s in filtered if s["type"] == "comparison"]
        rng.shuffle(bridge)
        rng.shuffle(compare)
        half = n // 2
        picked = bridge[:half] + compare[: n - half]
        rng.shuffle(picked)
    else:
        rng.shuffle(filtered)
        picked = filtered[:n]

    return picked[:n], "huggingface"


def _normalize_hf_record(rec: dict) -> dict:
    """Project HF HotpotQA record to our canonical schema (verbatim-compatible)."""
    return {
        "_id": rec.get("id", rec.get("_id", "")),
        "question": rec["question"],
        "answer": rec["answer"],
        "type": rec.get("type", "bridge"),
        "level": rec.get("level", "hard"),
        "context": {
            "title": list(rec["context"]["title"]),
            "sentences": [list(s) for s in rec["context"]["sentences"]],
        },
        "supporting_facts": {
            "title": list(rec["supporting_facts"]["title"]),
            "sent_id": list(rec["supporting_facts"]["sent_id"]),
        },
    }


# =============================================================================
# Synthetic fallback: hand-crafted multi-hop set
# =============================================================================
#
# These are short, self-contained multi-hop questions. Each requires chaining
# across 2-3 passages. Facts chosen from stable, widely-known encyclopedia
# content to reduce risk of being wrong. The set is intentionally small — it
# is for pipeline validation and pilot runs only, NOT for paper numbers.
#
# NOTE: Do not expand this list past ~30 without checking facts carefully.
# Reviewers cannot verify a synthetic set easily; the real HotpotQA is the
# expected dataset for the paper.

_SYNTHETIC_SAMPLES: list[dict] = [
    {
        "_id": "syn_001",
        "question": "What is the capital of the country where the composer of the opera 'The Magic Flute' was born?",
        "answer": "Vienna",
        "type": "bridge",
        "level": "hard",
        "context": {
            "title": ["The Magic Flute", "Wolfgang Amadeus Mozart", "Austria"],
            "sentences": [
                [
                    "The Magic Flute is an opera in two acts by Wolfgang Amadeus Mozart.",
                    "It premiered in Vienna in 1791.",
                ],
                [
                    "Wolfgang Amadeus Mozart was a prolific composer of the Classical period.",
                    "He was born in Salzburg, in what is now Austria.",
                ],
                [
                    "Austria is a landlocked country in Central Europe.",
                    "Its capital and largest city is Vienna.",
                ],
            ],
        },
        "supporting_facts": {
            "title": ["The Magic Flute", "Wolfgang Amadeus Mozart", "Austria"],
            "sent_id": [0, 1, 1],
        },
    },
    {
        "_id": "syn_002",
        "question": "In which country is the university that awarded Marie Curie her doctorate located?",
        "answer": "France",
        "type": "bridge",
        "level": "hard",
        "context": {
            "title": ["Marie Curie", "University of Paris"],
            "sentences": [
                [
                    "Marie Curie was a physicist and chemist of Polish and naturalized-French nationality.",
                    "She received her doctorate from the University of Paris in 1903.",
                ],
                [
                    "The University of Paris, also known as the Sorbonne, is located in Paris.",
                    "Paris is the capital city of France.",
                ],
            ],
        },
        "supporting_facts": {
            "title": ["Marie Curie", "University of Paris"],
            "sent_id": [1, 1],
        },
    },
    {
        "_id": "syn_003",
        "question": "Which actor starred in both 'Forrest Gump' and 'Cast Away'?",
        "answer": "Tom Hanks",
        "type": "comparison",
        "level": "medium",
        "context": {
            "title": ["Forrest Gump", "Cast Away"],
            "sentences": [
                [
                    "Forrest Gump is a 1994 American comedy-drama film.",
                    "It stars Tom Hanks in the title role.",
                ],
                [
                    "Cast Away is a 2000 American survival drama film.",
                    "The lead role was played by Tom Hanks.",
                ],
            ],
        },
        "supporting_facts": {
            "title": ["Forrest Gump", "Cast Away"],
            "sent_id": [1, 1],
        },
    },
    {
        "_id": "syn_004",
        "question": "What is the primary language of the country that hosts CERN's main site?",
        "answer": "French",
        "type": "bridge",
        "level": "hard",
        "context": {
            "title": ["CERN", "Switzerland", "Geneva"],
            "sentences": [
                [
                    "CERN is the European Organization for Nuclear Research.",
                    "Its main site is located near Geneva, Switzerland.",
                ],
                [
                    "Switzerland has four official languages.",
                    "The canton of Geneva is primarily French-speaking.",
                ],
                [
                    "Geneva is the second-most populous city in Switzerland.",
                    "French is the predominant language of the city.",
                ],
            ],
        },
        "supporting_facts": {
            "title": ["CERN", "Geneva"],
            "sent_id": [1, 1],
        },
    },
    {
        "_id": "syn_005",
        "question": "Which of these was founded first: Oxford University or Cambridge University?",
        "answer": "Oxford University",
        "type": "comparison",
        "level": "medium",
        "context": {
            "title": ["University of Oxford", "University of Cambridge"],
            "sentences": [
                [
                    "The University of Oxford is a collegiate research university in Oxford, England.",
                    "Teaching at Oxford existed in some form as early as 1096.",
                ],
                [
                    "The University of Cambridge is a collegiate research university in Cambridge, England.",
                    "Cambridge was founded in 1209.",
                ],
            ],
        },
        "supporting_facts": {
            "title": ["University of Oxford", "University of Cambridge"],
            "sent_id": [1, 1],
        },
    },
    {
        "_id": "syn_006",
        "question": "What year was the author of 'Hamlet' born?",
        "answer": "1564",
        "type": "bridge",
        "level": "medium",
        "context": {
            "title": ["Hamlet", "William Shakespeare"],
            "sentences": [
                [
                    "Hamlet is a tragedy written by William Shakespeare.",
                    "It was probably composed between 1599 and 1601.",
                ],
                [
                    "William Shakespeare was an English playwright and poet.",
                    "He was born in Stratford-upon-Avon in 1564.",
                ],
            ],
        },
        "supporting_facts": {
            "title": ["Hamlet", "William Shakespeare"],
            "sent_id": [0, 1],
        },
    },
    {
        "_id": "syn_007",
        "question": "In which city was the theory of relativity primarily developed by its originator?",
        "answer": "Bern",
        "type": "bridge",
        "level": "hard",
        "context": {
            "title": ["Theory of relativity", "Albert Einstein"],
            "sentences": [
                [
                    "The theory of relativity was developed by Albert Einstein.",
                    "The special theory of relativity was published in 1905.",
                ],
                [
                    "Albert Einstein was a theoretical physicist.",
                    "He was working at the patent office in Bern when he published his special relativity paper.",
                ],
            ],
        },
        "supporting_facts": {
            "title": ["Theory of relativity", "Albert Einstein"],
            "sent_id": [0, 1],
        },
    },
    {
        "_id": "syn_008",
        "question": "Which river flows through both the capital of France and the capital of England?",
        "answer": "None (trick)",
        "type": "comparison",
        "level": "hard",
        "context": {
            "title": ["Seine", "Thames"],
            "sentences": [
                [
                    "The Seine is a river in northern France.",
                    "It flows through Paris, the capital of France.",
                ],
                [
                    "The Thames is a river in southern England.",
                    "It flows through London, the capital of England.",
                ],
            ],
        },
        "supporting_facts": {
            "title": ["Seine", "Thames"],
            "sent_id": [1, 1],
        },
    },
    {
        "_id": "syn_009",
        "question": "What is the nationality of the scientist who formulated the three laws of motion?",
        "answer": "English",
        "type": "bridge",
        "level": "medium",
        "context": {
            "title": ["Laws of motion", "Isaac Newton"],
            "sentences": [
                [
                    "Newton's laws of motion are three basic laws of classical mechanics.",
                    "They were formulated by Isaac Newton.",
                ],
                [
                    "Sir Isaac Newton was an English mathematician and physicist.",
                    "He was born in Woolsthorpe, England in 1643.",
                ],
            ],
        },
        "supporting_facts": {
            "title": ["Laws of motion", "Isaac Newton"],
            "sent_id": [1, 0],
        },
    },
    {
        "_id": "syn_010",
        "question": "Which is larger by area: Canada or Russia?",
        "answer": "Russia",
        "type": "comparison",
        "level": "easy",
        "context": {
            "title": ["Canada", "Russia"],
            "sentences": [
                [
                    "Canada is a country in North America.",
                    "It has a total area of approximately 9.98 million square kilometers.",
                ],
                [
                    "Russia is a country in Eastern Europe and Northern Asia.",
                    "It has a total area of approximately 17.1 million square kilometers.",
                ],
            ],
        },
        "supporting_facts": {
            "title": ["Canada", "Russia"],
            "sent_id": [1, 1],
        },
    },
]


def _build_synthetic_multihop_set(n: int, seed: int = 42) -> list[dict]:
    """Return up to n samples from the hand-crafted synthetic set."""
    rng = random.Random(seed)
    pool = list(_SYNTHETIC_SAMPLES)
    rng.shuffle(pool)
    if n > len(pool):
        print(
            f"[data] Requested n={n} but synthetic pool has only "
            f"{len(pool)} samples. Returning all."
        )
        return pool
    return pool[:n]


# =============================================================================
# Utility
# =============================================================================


def extract_context_text(
    sample: dict,
    max_chars: int | None = None,
    number_sentences: bool = False,
) -> str:
    """
    Flatten context passages to a plain string for LLM input.

    When `max_chars` is given, passages containing supporting facts are kept
    first and distractors are dropped from the end. This ordering matters more
    than it looks: HotpotQA scatters supporting passages anywhere among the
    ten, so slicing the joined string blindly deletes the evidence needed to
    answer. A pilot lost the answer this way on a majority of samples — every
    condition dutifully replied "the passages do not contain this information",
    which is correct behaviour on a broken input and tells us nothing about the
    protocols being compared.

    `number_sentences` prefixes each sentence with its index, giving the
    SK-index condition stable targets to reference instead of quoting. It must
    be enabled for every condition in a comparison or none: numbering changes
    the prompt, and applying it only where it helps would confound the
    contrast it is meant to support.

    Passing max_chars=None returns everything, which is the right default.
    Truncate only when a context genuinely exceeds the model's window, and
    prefer raising max_model_len over discarding evidence.
    """
    titles = sample["context"]["title"]
    sentences = sample["context"]["sentences"]

    supporting = set(sample.get("supporting_facts", {}).get("title", []))

    blocks: list[tuple[bool, str]] = []
    for title, sents in zip(titles, sentences):
        if number_sentences:
            body = " ".join(f"[{i}] {s}" for i, s in enumerate(sents))
        else:
            body = " ".join(sents)
        blocks.append((title in supporting, f"[{title}] " + body))

    if max_chars is None:
        return "\n".join(text for _, text in blocks)

    # Supporting passages first, distractors after, then fill to the budget.
    ordered = [t for is_sup, t in blocks if is_sup] + \
              [t for is_sup, t in blocks if not is_sup]

    kept: list[str] = []
    used = 0
    for text in ordered:
        cost = len(text) + 1  # newline
        if used + cost > max_chars:
            continue  # skip this one, a later shorter passage may still fit
        kept.append(text)
        used += cost

    # If even the first supporting passage exceeds the budget, keep it anyway
    # in truncated form: a clipped piece of the right passage beats a complete
    # set of irrelevant ones.
    if not kept and ordered:
        return ordered[0][:max_chars]

    return "\n".join(kept)


def supporting_facts_present(sample: dict, context_text: str) -> bool:
    """
    Whether every supporting passage survived into `context_text`.

    Call this after building a prompt to confirm the sample is answerable at
    all. A sample failing this check should be excluded or the budget raised —
    scoring it would measure prompt construction, not protocol behaviour.
    """
    supporting = sample.get("supporting_facts", {}).get("title", [])
    return all(f"[{title}]" in context_text for title in supporting)


def save_samples(samples: list[dict], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)


def load_samples(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)