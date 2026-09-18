"""Self-tests for sk/scoring.py (fully deterministic, no model calls)."""
import sys
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sk"))

from scoring import (
    score_recoverability, clean_reconstructed_answer,
    score_propagation, score_detection, score_on_topic,
    calibrate_on_topic_thresholds, _mentions_token, _token_variants,
)

# =============================================================================
# 1. Recoverability
# =============================================================================
r = score_recoverability("Vienna", "Vienna")
assert r.em == 1 and r.f1 == 1.0 and r.recoverable_strict and r.recoverable_lenient

r = score_recoverability("the Vienna", "Vienna")
assert r.em == 1, "article stripping should make this exact"

r = score_recoverability("Vienna, Austria", "Vienna")
assert r.em == 0, "extra token breaks EM"
assert r.f1 > 0.5, f"but F1 should be decent, got {r.f1}"
assert r.recoverable_lenient, "lenient should accept partial"

r = score_recoverability("Paris", "Vienna")
assert r.em == 0 and r.f1 == 0.0
assert not r.recoverable_strict and not r.recoverable_lenient

r = score_recoverability("INSUFFICIENT", "Vienna")
assert r.em == 0 and not r.recoverable_strict
print("score_recoverability: PASS")

# --- Answer cleaning ---
assert clean_reconstructed_answer("Answer: Vienna") == "Vienna"
assert clean_reconstructed_answer("A: Vienna.") == "Vienna"
assert clean_reconstructed_answer('  "Vienna"  ') == "Vienna"
assert clean_reconstructed_answer("Vienna\nSome extra prose") == "Vienna"
assert clean_reconstructed_answer("St. Louis") == "St. Louis", "internal period preserved"
assert clean_reconstructed_answer("") == ""
assert clean_reconstructed_answer("INSUFFICIENT") == "INSUFFICIENT"
print("clean_reconstructed_answer: PASS")

# =============================================================================
# 2. Token matching (foundation for propagation/detection)
# =============================================================================
assert _mentions_token("She was born in Vienna in 1867.", "Vienna")
assert _mentions_token("vienna is the capital", "Vienna"), "case insensitive"
assert not _mentions_token("Viennese pastry", "Vienna"), "no substring false positive"
assert not _mentions_token("The answer is Oslobodjenje", "Oslo"), "no substring match"
assert _mentions_token("They visited Oslo yesterday", "Oslo")
assert _mentions_token("New York City is large", "New York"), "multi-word phrase"
assert not _mentions_token("", "Vienna")
assert not _mentions_token("Vienna", "")
print("_mentions_token: PASS")

# =============================================================================
# 3. Propagation
# =============================================================================
# Case: agent gave the injected wrong answer → propagated
p = score_propagation(
    final_answer="Vienna",
    gold_answer="Warsaw",
    injected_token="Vienna",
    original_token="Warsaw",
)
assert p.wrong_vs_gold and p.mentions_injected and p.propagated
assert not p.mentions_original
print(f"  propagated case: propagated={p.propagated}, em={p.final_em}")

# Case: agent gave the correct answer → not propagated
p = score_propagation(
    final_answer="Warsaw",
    gold_answer="Warsaw",
    injected_token="Vienna",
    original_token="Warsaw",
)
assert not p.wrong_vs_gold and not p.propagated
assert p.mentions_original, "should note the agent recovered the original fact"
print(f"  recovered case: propagated={p.propagated}, mentions_original={p.mentions_original}")

# Case: agent wrong but for an unrelated reason → NOT propagated
# (this is the key discriminating case — v1's substring approach would
#  have miscounted anything wrong as propagation)
p = score_propagation(
    final_answer="Berlin",
    gold_answer="Warsaw",
    injected_token="Vienna",
    original_token="Warsaw",
)
assert p.wrong_vs_gold, "answer is wrong"
assert not p.mentions_injected, "but not because of our injection"
assert not p.propagated, "so NOT propagated — this is the crucial distinction"
print(f"  wrong-but-unrelated: propagated={p.propagated} (correctly excluded)")

# Case: partial answer carrying injected error (lenient variant)
p = score_propagation(
    final_answer="Vienna, Austria",
    gold_answer="Warsaw",
    injected_token="Vienna",
    original_token="Warsaw",
    wrongness_f1_threshold=0.5,
)
assert p.propagated, "strict should catch it (EM fails, mentions injected)"
assert p.propagated_lenient
print("score_propagation: PASS")

# =============================================================================
# 4. Detection
# =============================================================================
# Case: verifier flagged AND named the injected token → strict detection
d = score_detection(
    verified_flag=False,
    reasons=["The claim that she was born in Vienna contradicts the other passage."],
    injected_token="Vienna",
    original_token="Warsaw",
)
assert d.verdict_parsed and d.flagged_unverified
assert d.mentions_injected and d.detected_strict and d.detected_lenient
print(f"  specific detection: strict={d.detected_strict}")

# Case: verifier flagged but was vague → lenient only
d = score_detection(
    verified_flag=False,
    reasons=["Something in the source material seems inconsistent."],
    injected_token="Vienna",
    original_token="Warsaw",
)
assert d.flagged_unverified and d.detected_lenient
assert not d.detected_strict, "vague flag should not count as strict detection"
print(f"  vague detection: strict={d.detected_strict}, lenient={d.detected_lenient}")

# Case: verifier passed the chain → no detection either way
d = score_detection(
    verified_flag=True,
    reasons=[],
    injected_token="Vienna",
    original_token="Warsaw",
)
assert not d.detected_strict and not d.detected_lenient
print(f"  no detection: strict={d.detected_strict}, lenient={d.detected_lenient}")

# Case: verdict unparseable → must be distinguishable from "not detected"
d = score_detection(
    verified_flag=None,
    reasons=[],
    injected_token="Vienna",
    original_token="Warsaw",
)
assert not d.verdict_parsed, "must flag as unparsed so caller can exclude it"
print(f"  unparsed verdict: verdict_parsed={d.verdict_parsed} (excluded from denominator)")

# Case: reasons empty but prose mentions it → fallback search space
d = score_detection(
    verified_flag=False,
    reasons=[],
    injected_token="Vienna",
    original_token="Warsaw",
    full_verifier_text="I notice the Vienna birthplace conflicts with the source.",
)
assert d.detected_strict, "should fall back to full text when reasons list is empty"
print(f"  prose fallback: strict={d.detected_strict}")

# Case: verifier names the ORIGINAL token (equally valid detection)
d = score_detection(
    verified_flag=False,
    reasons=["The passage should say Warsaw based on the other source."],
    injected_token="Vienna",
    original_token="Warsaw",
)
assert d.mentions_original and d.detected_strict
print("score_detection: PASS")

# =============================================================================
# 5. On-topic
# =============================================================================
assert score_on_topic(0.80).label == "on_topic"
assert score_on_topic(0.45).label == "partial"
assert score_on_topic(0.10).label == "off_topic"
assert score_on_topic(0.55).label == "on_topic", "boundary inclusive"
assert score_on_topic(0.35).label == "partial", "boundary inclusive"
print("score_on_topic: PASS")

# --- Threshold calibration ---
labeled = [
    (0.85, "on_topic"), (0.78, "on_topic"), (0.72, "on_topic"),
    (0.50, "partial"), (0.45, "partial"), (0.42, "partial"),
    (0.20, "off_topic"), (0.15, "off_topic"), (0.08, "off_topic"),
]
on_t, off_t = calibrate_on_topic_thresholds(labeled)
print(f"  calibrated thresholds: on={on_t:.3f}, off={off_t:.3f}")
# Verify the calibrated thresholds actually reproduce the labels well
correct = sum(
    1 for sim, gold in labeled
    if score_on_topic(sim, on_t, off_t).label == gold
)
assert correct >= 8, f"calibration should fit at least 8/9, got {correct}"
print(f"  calibration accuracy: {correct}/{len(labeled)}")

# Empty input → defaults
assert calibrate_on_topic_thresholds([]) == (0.55, 0.35)
print("calibrate_on_topic_thresholds: PASS")

print("\nAll scoring self-tests PASSED")
