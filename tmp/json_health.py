import json

path = 'results/sweep/qwen3.5-27b_/win16k_turns96/exp3.jsonl'
rows = [json.loads(l) for l in open(path) if l.strip()]

for cond in ('SK', 'SK-cite', 'SK-index'):
    g = [r for r in rows if r['condition'] == cond]
    if not g:
        continue
    total = bad = 0
    for r in g:
        for t in r['turn_responses']:
            total += 1
            try:
                json.loads(t.strip())
            except Exception:
                bad += 1
    em = sum(r['final_em'] for r in g) / len(g)
    print(f'{cond:<10} malformed {bad:>5}/{total:<6} ({bad/total:>5.1%})   EM={em:.3f}')
