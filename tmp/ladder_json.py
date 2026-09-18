import json
from pathlib import Path

root = Path("results/ladder")
print(f"{'model':<30} {'turns':>6} {'SK':>8} {'SK-index':>10}")
print("-" * 58)
for model in sorted(root.iterdir()):
    if not model.is_dir():
        continue
    for turns in (6, 24, 96):
        p = model / f"turns{turns}" / "exp3.jsonl"
        if not p.exists():
            continue
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        out = {}
        for cond in ("SK", "SK-index"):
            g = [r for r in rows if r["condition"] == cond]
            tot = bad = 0
            for r in g:
                for t in r["turn_responses"]:
                    tot += 1
                    try:
                        json.loads(t.strip())
                    except Exception:
                        bad += 1
            out[cond] = bad / tot if tot else 0.0
        print(f"{model.name[:30]:<30} {turns:>6} "
              f"{out.get('SK', 0):>7.1%} {out.get('SK-index', 0):>9.1%}")
