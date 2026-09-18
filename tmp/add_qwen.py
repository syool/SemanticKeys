"""Add the existing Qwen sweep cells to the ladder for the cost analysis."""
import json
from pathlib import Path

src = Path("results/sweep/qwen3.5-27b_")
dst = Path("results/ladder/qwen3.5-27b_")
keep = {"NL-anchor", "SK", "SK-index"}

for turns in (6, 24, 96):
    s = src / f"win16k_turns{turns}" / "exp3.jsonl"
    d = dst / f"turns{turns}"
    if not s.exists():
        print(f"  MISSING {s}")
        continue
    if (d / "exp3.jsonl").exists():
        print(f"  turns{turns}: already present")
        continue
    d.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in s.read_text().splitlines() if l.strip()]
    kept = [r for r in rows if r["condition"] in keep]
    (d / "exp3.jsonl").write_text("\n".join(json.dumps(r) for r in kept) + "\n")
    print(f"  turns{turns}: {len(kept)} rows")
