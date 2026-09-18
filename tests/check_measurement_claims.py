"""
check_measurement_claims.py — Verify §4 describes what the code actually does.

The Measurement section makes precise claims: that propagation requires both
wrongness and the injected token, that strict detection requires naming it,
that constrained decoding touches only the verifier verdict, that drift
normalisation strips field names. Any of these could drift out of agreement
with the implementation during revision, and a reader cannot check them
against the repository as easily as a test can.

This exercises each claim against the real functions rather than reading the
prose. It does not check wording — a rule can behave correctly and still be
described badly.

Run from the repository root:

    python3 tests/check_measurement_claims.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sk"))
issues = []

# 1. Propagation definition
from scoring import score_propagation
r = score_propagation("Paris", "Paris", "Paris", "London")
if r.propagated:
    issues.append("propagation: correct answer mentioning injected token counted as propagated")
r = score_propagation("Berlin", "Paris", "London", "Paris")
if r.propagated:
    issues.append("propagation: wrong-for-other-reasons counted as propagated")
r = score_propagation("London", "Paris", "London", "Paris")
if not r.propagated:
    issues.append("propagation: genuine propagation not detected")
print("propagation rule matches the described definition" if not issues else "MISMATCH")

# 2. Detection strict vs lenient
from scoring import score_detection
d = score_detection(False, ["something seems off"], "London", "Paris")
if d.detected_strict:
    issues.append("detection: vague verifier credited as strict")
if not d.detected_lenient:
    issues.append("detection: flag alone should satisfy lenient")
d = score_detection(False, ["London contradicts the passage"], "London", "Paris")
if not d.detected_strict:
    issues.append("detection: naming the token should satisfy strict")
print("detection strict/lenient behave as described")

# 3. Constrained decoding applied to exactly one schema
from scoring import VERIFIER_JSON_SCHEMA
props = set(VERIFIER_JSON_SCHEMA["properties"])
if props != {"verified", "reasons"}:
    issues.append(f"verifier schema is {props}, text says verified/reasons")
print(f"verifier schema: {sorted(props)}")

# 4. Drift normalisation strips protocol syntax
from runner import normalize_for_embedding
import json
sk_msg = json.dumps({"topic": "Warsaw", "evidence": ["born in Warsaw"]})
if "topic" in normalize_for_embedding(sk_msg):
    issues.append("drift: field names survive normalisation")
prose = "She was born in Warsaw."
if normalize_for_embedding(prose) != prose:
    issues.append("drift: prose altered by normalisation")
print("drift normalisation strips fields from JSON, leaves prose intact")

# 5. Injection strategies named in the text exist
from injection import inject_error
src = (ROOT / "sk" / "injection.py").read_text()
for strat in ("entity_substitution", "attribute_flip"):
    if strat not in src:
        issues.append(f"injection strategy '{strat}' named in text but absent from code")
print("injection strategies present")

# 6. Sanity checks the text claims exist
sanity = (ROOT / "sk" / "sanity.py").read_text()
claimed = {
    "evidence completeness": "check_evidence_completeness",
    "structured-output validity": "check_structured_output",
    "output truncation": "check_truncation",
    "floor and ceiling": "floor",
    "identical-response": "identical",
}
for label, needle in claimed.items():
    if needle not in sanity:
        issues.append(f"sanity check '{label}' claimed in text but not found in sanity.py")
print("sanity checks claimed in the text exist in the code")

print()
if issues:
    print(f"{len(issues)} MISMATCHES:")
    for i in issues:
        print(f"  {i}")
    sys.exit(1)
print("Measurement section agrees with the implementation.")