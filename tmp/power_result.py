import json
from pathlib import Path
sys_path = Path("results_power/ladder/gemma-4-e4b-it_/turns24/exp3.jsonl")
rows = [json.loads(l) for l in sys_path.read_text().splitlines() if l.strip()]

def paired(a, b):
    av = {r["sample_id"]: r["final_em"] for r in rows if r["condition"] == a}
    bv = {r["sample_id"]: r["final_em"] for r in rows if r["condition"] == b}
    common = sorted(set(av) & set(bv))
    return [av[s] for s in common], [bv[s] for s in common]

import sys
sys.path.insert(0, "sk")
from metrics import mcnemar_paired

for sk in ("SK-index", "SK"):
    a, b = paired(sk, "NL-anchor")
    if not a:
        continue
    diff = sum(a)/len(a) - sum(b)/len(b)
    m = mcnemar_paired([(bool(x), bool(y)) for x, y in zip(a, b)])
    print(f"{sk:<10} vs NL-anchor   n={len(a)}")
    print(f"   EM {sum(a)/len(a):.3f} vs {sum(b)/len(b):.3f}   diff {diff:+.3f}")
    print(f"   McNemar p = {m['pvalue']:.4f}   b10={m['b10']} b01={m['b01']}")
    print()
