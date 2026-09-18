"""Reuse the 31B sweep cells as ladder cells.

The sweep stored them as win16k_turnsN; the ladder expects turnsN. Same runs,
same settings (16K window, N=40), so copying rather than re-running is both
correct and saves hours. Only the three turn counts the ladder uses are taken.
"""
import shutil, json
from pathlib import Path

src_root = Path("results/sweep/gemma-4-31b-it-qat-w4a16-ct_")
dst_root = Path("results/ladder/gemma-4-31b-it-qat-w4a16-ct_")

if not src_root.exists():
    raise SystemExit(f"sweep data not found at {src_root}")

for turns in (6, 24, 96):
    src = src_root / f"win16k_turns{turns}" / "exp3.jsonl"
    dst = dst_root / f"turns{turns}"
    if not src.exists():
        print(f"  MISSING {src}")
        continue
    if (dst / "exp3.jsonl").exists():
        print(f"  turns{turns}: already present, skipping")
        continue
    dst.mkdir(parents=True, exist_ok=True)

    # The sweep ran five conditions; the ladder uses three. Filter so the
    # comparison is like-for-like across models.
    keep = {"NL-anchor", "SK", "SK-index"}
    rows = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
    kept = [r for r in rows if r["condition"] in keep]
    (dst / "exp3.jsonl").write_text(
        "\n".join(json.dumps(r) for r in kept) + "\n")
    conds = sorted({r["condition"] for r in kept})
    print(f"  turns{turns}: {len(kept)} rows, conditions {conds}")
