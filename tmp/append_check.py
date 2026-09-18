import json, glob, os

root = 'results/sweep/qwen3.5-27b_'
print(f"{'cell':<22} {'NL-append EM':>13} {'refusals':>10}")
print("-" * 48)
for d in sorted(glob.glob(f'{root}/win*_turns*')):
    p = os.path.join(d, 'exp3.jsonl')
    if not os.path.exists(p):
        continue
    rows = [json.loads(l) for l in open(p) if l.strip()]
    g = [r for r in rows if r['condition'] == 'NL-append']
    if not g:
        continue
    em = sum(r['final_em'] for r in g) / len(g)
    # First-turn refusals: the failure mode we just fixed
    refuse = sum(
        1 for r in g
        if r['turn_responses'] and any(
            m in r['turn_responses'][0].lower()
            for m in ('provide the question', 'have not provided', 'no question')
        )
    )
    print(f"{os.path.basename(d):<22} {em:>13.3f} {refuse:>7}/{len(g)}")
