"""
check_paper_numbers.py — Keep the manuscript's numbers tied to the data.

Every figure in the Results section was transcribed by hand from a run. The
manuscript will be revised many times before submission, and each revision is
a chance for a number to drift away from what was measured — or for a value
to be updated in one file and not the other.

This checks two things: that every headline figure quoted in results.md also
appears in observed_values.md, and that the ratios stated in the text agree
with the token counts they are derived from.

It does not check prose. A number can be correct and still be described
wrongly; that needs a reader.

Run from the repository root:

    python3 tests/check_paper_numbers.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
results = (ROOT / "paper" / "results.md").read_text()
values = (ROOT / "paper" / "observed_values.md").read_text()

# Claims that must appear identically in both files.
CLAIMS = [
    ("114,301", "Qwen SK-index input tokens"),
    ("757,931", "Qwen NL-anchor input tokens"),
    ("109,094", "Gemma SK-index input"),
    ("379,115", "Gemma NL-anchor input"),
    ("0.15",    "Qwen input ratio"),
    ("0.29",    "Gemma input ratio"),
    ("2.78",    "Gemma output ratio"),
    ("0.67",    "Qwen output ratio"),
    ("73.5%",   "Qwen 6-key malformed"),
    ("30.4%",   "E4B malformed at 96 turns"),
    ("28.3%",   "E4B malformed at 6 turns"),
    ("−0.949",  "verbosity correlation"),
    ("−0.230",  "size correlation"),
    ("−0.914",  "delta-size correlation"),
    ("0.598",   "N=400 SK-index EM"),
    ("0.502",   "N=400 baseline EM"),
    ("+0.095",  "N=400 delta"),
    ("0.0010",  "6-key p-value at N=400"),
    ("108,180", "E4B SK-index input"),
    ("346,455", "E4B baseline input"),
    ("195.8",   "Qwen baseline verbosity"),
    ("28.5",    "E4B baseline verbosity"),
]

missing_in_values = []
missing_in_results = []
for token, desc in CLAIMS:
    if token not in results:
        missing_in_results.append((token, desc))
    if token not in values:
        missing_in_values.append((token, desc))

print("Claims in results.md not backed by observed_values.md:")
if missing_in_values:
    for t, d in missing_in_values:
        print(f"  MISSING  {t:<10} {d}")
else:
    print("  (none)")

print("\nValues recorded but not used in results.md:")
if missing_in_results:
    for t, d in missing_in_results:
        print(f"  unused   {t:<10} {d}")
else:
    print("  (none)")

# Sanity: ratios should be consistent with the token counts quoted
checks = [
    ("Qwen input ratio", 114301 / 757931, 0.15),
    ("Gemma input ratio", 109094 / 379115, 0.29),
    ("Qwen output ratio", 12500 / 18797, 0.67),
    ("Gemma output ratio", 9300 / 3336, 2.78),
    ("E4B input ratio", 108180 / 346455, 0.31),
]
print("\nArithmetic consistency:")
failed = 0
for name, computed, stated in checks:
    ok = abs(computed - stated) < 0.011
    if not ok:
        failed += 1
    print(f"  {'ok  ' if ok else 'FAIL'} {name:<22} computed {computed:.3f} vs stated {stated}")

print()
if missing_in_values or failed:
    print(f"{len(missing_in_values)} unbacked claims, {failed} arithmetic errors.")
    sys.exit(1)
print("Manuscript numbers agree with the recorded values.")