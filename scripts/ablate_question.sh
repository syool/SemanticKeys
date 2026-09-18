#!/usr/bin/env bash
# ablate_question.sh — Does accuracy survive on protocol state, or on the
# question simply staying visible?
#
# The horizon sweep found no accuracy difference between structured and
# natural-language conditions, and 6-key SK held EM ~0.70 on Qwen despite 56%
# of its turns emitting unparseable JSON. State propagation was failing on
# half the turns and nothing happened. That is the tell: if the state tuple
# were carrying the objective, breaking it should have cost something.
#
# The likely explanation is that the question stays in context every turn, so
# the model never has to rely on what the protocol preserved. This experiment
# removes it: the question is stated once, then withheld from goal reminders,
# SK state fields, and the retained transcript. Whatever the protocol carried
# is all that remains.
#
# A reviewer will ask this of any paper claiming compression without loss, and
# without it the cost argument rests on an accuracy result whose mechanism is
# unexplained.
#
# Design (settled in earlier planning):
#   96 turns only   — shorter horizons never accumulate enough history for
#                     state anchoring to matter
#   16K window only — 4K mixes in transcript dropping, which muddies the read
#   3 conditions    — NL-anchor, SK, SK-index. NL-append is excluded: its
#                     question lives only in the transcript, so "removal" is
#                     not defined for it. SK-cite tracks SK-index closely
#                     enough to omit.
#
# Both arms are run so the comparison is within-experiment rather than against
# the earlier sweep, which differed in other ways.
#
# Usage:
#   ./scripts/ablate_question.sh              # both arms
#   ARMS=removed ./scripts/ablate_question.sh # just the removal arm
#   N_SAMPLES=20 ./scripts/ablate_question.sh # quick look

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/models.conf"

BASE_URL="${VLLM_BASE_URL:-http://localhost:${VLLM_PORT_DEFAULT}/v1}"
DATA_PATH="${SK_DATA:-$ROOT/data}/$HOTPOTQA_LOCAL_FILE"
RESULTS_ROOT="${SK_RESULTS:-$ROOT/results}/question_ablation"

N_SAMPLES="${N_SAMPLES:-40}"
TURNS="${TURNS:-96}"
BUDGET="${BUDGET:-48000}"           # 16K window
CONCURRENCY="${CONCURRENCY:-32}"
CONDITIONS="${CONDITIONS:-NL-anchor SK SK-index}"
ARMS="${ARMS:-kept removed}"

export SK_EMBED_DEVICE="${SK_EMBED_DEVICE:-cpu}"

# --- Embedder, resolved locally so a network blip cannot end the run -------
EMBEDDER="${SK_EMBED_MODEL:-}"
if [[ -z "$EMBEDDER" ]]; then
    EMBEDDER=$(python3 - <<'PYEOF' 2>/dev/null || true
from huggingface_hub import scan_cache_dir
for repo in scan_cache_dir().repos:
    if "all-mpnet-base-v2" in repo.repo_id:
        revs = sorted(repo.revisions, key=lambda r: r.last_modified, reverse=True)
        if revs:
            print(revs[0].snapshot_path)
        break
PYEOF
)
fi
if [[ -n "$EMBEDDER" && -d "$EMBEDDER" ]]; then
    EMBED_ARGS=(--embedder-path "$EMBEDDER")
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    echo "[embedder] local snapshot: $EMBEDDER (device: $SK_EMBED_DEVICE)"
else
    EMBED_ARGS=()
    echo "[embedder] WARNING: no local snapshot; will resolve via the hub." >&2
fi

# --- Preflight -------------------------------------------------------------
[[ -f "$DATA_PATH" ]] || { echo "ERROR: dataset not found at $DATA_PATH" >&2; exit 1; }

if ! curl -sf "$BASE_URL/models" -o /tmp/abl_models.json --max-time 10; then
    echo "ERROR: vLLM unreachable at $BASE_URL" >&2
    echo "  Start it with: ./scripts/serve.sh qwen" >&2
    exit 1
fi
MODEL=$(python3 -c "
import json; d=json.load(open('/tmp/abl_models.json'))
ids=[m['id'] for m in d.get('data',[])]; print(ids[0] if ids else '')")
[[ -n "$MODEL" ]] || { echo "ERROR: no model loaded" >&2; exit 1; }

MODEL_TAG=$(basename "$MODEL" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9._-' '_')
read -r -a COND_ARR <<< "$CONDITIONS"
read -r -a ARM_ARR <<< "$ARMS"

echo "=========================================="
echo "Question-removal ablation"
echo "  Model:       $MODEL"
echo "  Turns:       $TURNS"
echo "  Window:      ${BUDGET} chars (~$((BUDGET / 4)) tokens)"
echo "  Conditions:  ${COND_ARR[*]}"
echo "  Arms:        ${ARM_ARR[*]}"
echo "  N:           $N_SAMPLES"
echo "  Output:      $RESULTS_ROOT/$MODEL_TAG"
echo "=========================================="

EST=$(python3 -c "
arms=${#ARM_ARR[@]}; conds=${#COND_ARR[@]}
print(f'{$N_SAMPLES*conds*($TURNS*250+48)*(1000/308)/1000/3600*arms:.1f}')")
echo ""
echo "Estimated runtime: ~${EST} h"
echo ""
if [[ "${ASSUME_YES:-0}" != "1" ]]; then
    read -rp "Proceed? [y/N] " confirm
    [[ "${confirm,,}" == "y" ]] || { echo "Aborted."; exit 0; }
fi

cd "$ROOT/sk"

for arm in "${ARM_ARR[@]}"; do
    case "$arm" in
        kept)    FLAG=() ;;
        removed) FLAG=(--remove-question) ;;
        *) echo "ERROR: arm must be 'kept' or 'removed', got '$arm'" >&2; exit 2 ;;
    esac

    OUT="$RESULTS_ROOT/$MODEL_TAG/question_${arm}"
    if [[ -f "$OUT/exp3.jsonl" ]]; then
        echo ""
        echo "[$arm] already done, skipping"
        continue
    fi
    mkdir -p "$OUT"

    echo ""
    echo "[$arm] running ($TURNS turns, ${#COND_ARR[@]} conditions)"
    if python3 runner.py \
        --base-url "$BASE_URL" --model "$MODEL" \
        --data-path "$DATA_PATH" \
        "${EMBED_ARGS[@]}" \
        --n-samples "$N_SAMPLES" \
        --conditions "${COND_ARR[@]}" \
        --exp 3 \
        --num-turns "$TURNS" \
        --prompt-budget-chars "$BUDGET" \
        --distractor-tier mixed \
        --concurrency "$CONCURRENCY" \
        --output-dir "$OUT" --seed 42 \
        "${FLAG[@]}" \
        > "$OUT/run.log" 2>&1
    then
        tail -3 "$OUT/run.log" | sed 's/^/    /'
    else
        echo "  ERROR: arm '$arm' failed. Last 20 lines:" >&2
        tail -20 "$OUT/run.log" | sed 's/^/    /' >&2
        exit 1
    fi
done

# --- Read the result -------------------------------------------------------
echo ""
echo "=========================================="
echo "Result"
echo "=========================================="
python3 - "$RESULTS_ROOT/$MODEL_TAG" <<'PYEOF'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])

def load(arm):
    p = root / f"question_{arm}" / "exp3.jsonl"
    if not p.exists():
        return None
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]

kept, removed = load("kept"), load("removed")
if not (kept and removed):
    print("  Need both arms to compare.")
    raise SystemExit

def em_by_cond(rows):
    out = {}
    for r in rows:
        out.setdefault(r["condition"], []).append(r["final_em"])
    return {c: sum(v) / len(v) for c, v in out.items()}

def json_health(rows, cond):
    g = [r for r in rows if r["condition"] == cond]
    tot = bad = 0
    for r in g:
        for t in r["turn_responses"]:
            tot += 1
            try:
                json.loads(t.strip())
            except Exception:
                bad += 1
    return bad / tot if tot else 0.0

k, rm = em_by_cond(kept), em_by_cond(removed)
print(f"\n  {'condition':<12} {'EM kept':>9} {'EM removed':>12} {'delta':>9} "
      f"{'malformed':>11}")
print("  " + "-" * 58)
for c in sorted(set(k) | set(rm)):
    a, b = k.get(c), rm.get(c)
    if a is None or b is None:
        continue
    mal = json_health(removed, c) if c.startswith("SK") else None
    mal_s = f"{mal:>10.1%}" if mal is not None else f"{'—':>10}"
    print(f"  {c:<12} {a:>9.3f} {b:>12.3f} {b-a:>+9.3f} {mal_s:>11}")

print()
print("  Reading this:")
print("    Large negative deltas for SK conditions mean the state tuple was")
print("    carrying the objective, and accuracy in the main sweep was a real")
print("    protocol effect.")
print("    Deltas near zero mean the models recovered the objective from the")
print("    passages or their own earlier responses, and the protocol was not")
print("    what held accuracy up.")
print("    A 6-key SK that stays flat despite a high malformed rate is the")
print("    strongest form of the second reading.")
PYEOF

echo ""
echo "Full analysis:"
echo "  python3 sk/crossover.py --root $RESULTS_ROOT/$MODEL_TAG --sk-condition SK-index"
echo "=========================================="