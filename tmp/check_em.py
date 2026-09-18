import json

path = '/root/PAPER/results/sweep/qwen3.5-27b_/win16k_turns96/exp3.jsonl'
rows = [json.loads(l) for l in open(path)]

for cond in ('SK', 'SK-index', 'NL-anchor', 'NL-append'):
    g = [x for x in rows if x['condition'] == cond]
    if not g:
        continue
    em = sum(x['final_em'] for x in g) / len(g)
    f1 = sum(x['final_f1'] for x in g) / len(g)
    near = [x for x in g if x['final_em'] == 0 and x['final_f1'] > 0.5]
    print(f'{cond:<12} EM={em:.3f}  F1={f1:.3f}  gap={f1-em:+.3f}  near-miss={len(near)}/{len(g)}')

print()
print('=== EM=0 but F1>0.5 (near misses, SK) ===')
sk = [x for x in rows
      if x['condition'] == 'SK' and x['final_em'] == 0 and x['final_f1'] > 0.5]
for r in sk[:10]:
    print('  gold=%-28s got=%-28s F1=%.2f' % (
        repr(r['gold_answer']), repr(r['final_answer']), r['final_f1']))
