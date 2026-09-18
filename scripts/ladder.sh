#!/usr/bin/env bash
# ladder.sh — How do the cost gap and any accuracy effect move with model size?
#
# Two questions, one experiment, because both fall out of the same run:
#
#   Cost.     The input-token saving differed between the 27B and 31B models
#             (0.15x vs 0.29x), and the reason was baseline verbosity: Gemma's
#             NL-anchor answered in ~35 tokens per turn where Qwen's used far
#             more, so there was less transcript for structure to avoid
#             re-sending. If that is the mechanism, the saving should track how
#             verbose a model is rather than how large it is.
#
#   Accuracy. At 27B and 31B, removing the question barely moved EM even when
#             73.5% of 6-key SK turns emitted unparseable JSON — the models
#             recovered the objective from the passages regardless of what the
#             protocol carried. A smaller model may not manage that, and if so
#             structure would start to matter for accuracy at the low end.
#
# Ladder: Gemma 4 E4B, 12B, 31B. One family, so architecture is held fixed.
#
# PRECISION CAVEAT: Google publishes QAT w4a16 checkpoints for 12B and 31B but
# not for E4B, which runs bf16. Full precision favours the small end, so its
# accuracy is an upper bound. Token counts are unaffected.
#
# Turn counts 6, 24, 96. Not 96 alone: if a small model holds up early and
# fails late, where it fails is the finding, and a single horizon cannot show
# that. Window fixed at 16K — 4K mixes in transcript dropping, which confounds
# the cost read this script exists to take.
#
# Usage:
#   ./scripts/ladder.sh gemma-e4b        # serve it first, then run this
#   TURNS="6 96" ./scripts/ladder.sh gemma-12b
#
# 31B does not need re-running: point compare_ladder.py at the existing sweep.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/models.conf"

BASE_URL="${VLLM_BASE_URL:-http://localhost:${VLLM_PORT_DEFAULT}/v1}"
DATA_PATH="${SK_DATA:-$ROOT/data}/$HOTPOTQA_LOCAL_FILE"
RESULTS_ROOT="${SK_RESULTS:-$ROOT/results}/ladder"

N_SAMPLES="${N_SAMPLES:-40}"
TURNS="${TURNS:-6 24 96}"
BUDGET="${BUDGET:-48000}"                       # 16K window
CONCURRENCY="${CONCURRENCY:-32}"
CONDITIONS="${CONDITIONS:-NL-anchor SK SK-index}"

export SK_EMBED_DEVICE="${SK_EMBED_DEVICE:-cpu}"

# --- Embedder from local cache so a network blip cannot end the run --------
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
    echo "[embedder] local snapshot (device: $SK_EMBED_DEVICE)"
else
    EMBED_ARGS=()
    echo "[embedder] WARNING: no local snapshot; will resolve via the hub." >&2
fi

# --- Preflight -------------------------------------------------------------
[[ -f "$DATA_PATH" ]] || { echo "ERROR: dataset not found at $DATA_PATH" >&2; exit 1; }

if ! curl -sf "$BASE_URL/models" -o /tmp/ladder_models.json --max-time 10; then
    echo "ERROR: vLLM unreachable at $BASE_URL" >&2
    echo "  Start one of: ./scripts/serve.sh {gemma-e4b|gemma-12b|gemma}" >&2
    exit 1
fi
MODEL=$(python3 -c "
import json; d=json.load(open('/tmp/ladder_models.json'))
ids=[m['id'] for m in d.get('data',[])]; print(ids[0] if ids else '')")
[[ -n "$MODEL" ]] || { echo "ERROR: no model loaded" >&2; exit 1; }

MODEL_TAG=$(basename "$MODEL" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9._-' '_')
read -r -a TURN_ARR <<< "$TURNS"
read -r -a COND_ARR <<< "$CONDITIONS"

echo "=========================================="
echo "Scale ladder"
echo "  Model:       $MODEL"
echo "  Turns:       ${TURN_ARR[*]}"
echo "  Window:      ${BUDGET} chars (~$((BUDGET / 4)) tokens)"
echo "  Conditions:  ${COND_ARR[*]}"
echo "  N:           $N_SAMPLES"
echo "  Output:      $RESULTS_ROOT/$MODEL_TAG"
echo "=========================================="

EST=$(python3 -c "
turns=[${TURNS// /,}]
print(f'{sum($N_SAMPLES*${#COND_ARR[@]}*(t*250+48)*(1000/308)/1000 for t in turns)/3600:.1f}')")
echo ""
echo "Estimated runtime: ~${EST} h (small models will be faster than this)"
echo ""
if [[ "${ASSUME_YES:-0}" != "1" ]]; then
    read -rp "Proceed? [y/N] " confirm
    [[ "${confirm,,}" == "y" ]] || { echo "Aborted."; exit 0; }
fi

cd "$ROOT/sk"

for turns in "${TURN_ARR[@]}"; do
    OUT="$RESULTS_ROOT/$MODEL_TAG/turns${turns}"
    if [[ -f "$OUT/exp3.jsonl" ]]; then
        echo ""
        echo "[${turns} turns] already done, skipping"
        continue
    fi
    mkdir -p "$OUT"

    echo ""
    echo "[${turns} turns] running"
    if python3 runner.py \
        --base-url "$BASE_URL" --model "$MODEL" \
        --data-path "$DATA_PATH" \
        "${EMBED_ARGS[@]}" \
        --n-samples "$N_SAMPLES" \
        --conditions "${COND_ARR[@]}" \
        --exp 3 \
        --num-turns "$turns" \
        --prompt-budget-chars "$BUDGET" \
        --distractor-tier mixed \
        --concurrency "$CONCURRENCY" \
        --output-dir "$OUT" --seed 42 \
        > "$OUT/run.log" 2>&1
    then
        python3 - "$OUT/exp3.jsonl" <<'PYEOF'
import json, sys, collections
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
em = collections.defaultdict(list)
tok = collections.defaultdict(list)
for r in rows:
    em[r["condition"]].append(r["final_em"])
    tok[r["condition"]].append(r["cumulative_in_tokens"])
for c in sorted(em):
    print(f"    {c:<12} EM={sum(em[c])/len(em[c]):.3f}  "
          f"in-tok={sum(tok[c])/len(tok[c]):>9,.0f}")
PYEOF
    else
        echo "  ERROR: failed. Last 20 lines:" >&2
        tail -20 "$OUT/run.log" | sed 's/^/    /' >&2
        exit 1
    fi
done

echo ""
echo "=========================================="
echo "Done: $RESULTS_ROOT/$MODEL_TAG"
echo ""
echo "Next model:"
echo "  ./scripts/serve.sh gemma-12b     # then re-run this"
echo ""
echo "When the ladder is complete:"
echo "  python3 sk/compare_ladder.py --root $RESULTS_ROOT"
echo "=========================================="