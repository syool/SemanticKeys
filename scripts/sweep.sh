#!/usr/bin/env bash
# sweep.sh — Horizon sweep: goal maintenance and cost versus turn count.
#
# The central experiment of the revised direction. Structured communication
# carries a fixed per-message overhead; its claimed benefit — propagating a
# bounded state tuple instead of an ever-growing transcript — only accrues as
# turns accumulate. Earlier experiments measured one turn, three turns, and
# six, which is the regime where structure cannot win. This sweeps to 96.
#
# Two window conditions. At ~250 output tokens per turn an NL transcript
# fills a 16K window near turn 53 and a 4K window near turn 11; the SK state
# tuple stays around 1K throughout. Past those points NL conditions drop
# their oldest turns to fit, which is what a real deployment does and is
# recorded per run as turns_dropped.
#
# Usage:
#   ./scripts/sweep.sh                    # both windows, current model
#   WINDOWS=16k ./scripts/sweep.sh        # one window only
#   TURNS="6 12 24" ./scripts/sweep.sh    # shorter sweep
#
# Resumable: completed (window, turns) cells are skipped, so an interrupted
# run continues rather than restarting.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/models.conf"

BASE_URL="${VLLM_BASE_URL:-http://localhost:${VLLM_PORT_DEFAULT}/v1}"
DATA_PATH="${SK_DATA:-$ROOT/data}/$HOTPOTQA_LOCAL_FILE"
RESULTS_ROOT="${SK_RESULTS:-$ROOT/results}/sweep"

N_SAMPLES="${N_SAMPLES:-40}"
TURNS="${TURNS:-6 12 24 48 96}"
WINDOWS="${WINDOWS:-16k 4k}"
CONCURRENCY="${CONCURRENCY:-32}"
# Both three-field variants are included: SK-cite quotes its evidence and
# SK-index references it. Running both separates "fewer fields helps" from
# "compact evidence helps", which a single variant cannot distinguish.
CONDITIONS="${CONDITIONS:-NL-append NL-anchor SK SK-cite SK-index}"

# The embedder is loaded fresh for every cell. Without an explicit local path
# it resolves through the HuggingFace hub each time, which works while the
# cache is warm and the network is up — and takes the whole sweep down when
# either stops being true. A run died this way six hours in.
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
    echo "[embedder] using local snapshot: $EMBEDDER"
    EMBED_ARGS=(--embedder-path "$EMBEDDER")
    # Belt and braces: even with a local path, transformers may probe the hub
    # for metadata unless told the machine is offline.
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
else
    echo "[embedder] WARNING: no local snapshot found; will resolve via the hub." >&2
    echo "           A network interruption will abort the sweep mid-run." >&2
    echo "           Set SK_EMBED_MODEL to a local directory to avoid this." >&2
    EMBED_ARGS=()
fi

# Character budgets, ~4 chars per token with margin for the system prompt.
BUDGET_16k=48000
BUDGET_4k=12000

# ---------------------------------------------------------------------------
[[ -f "$DATA_PATH" ]] || { echo "ERROR: dataset not found at $DATA_PATH" >&2; exit 1; }

if ! curl -sf "$BASE_URL/models" -o /tmp/sweep_models.json --max-time 10; then
    echo "ERROR: vLLM unreachable at $BASE_URL" >&2
    echo "  Start it with: ./scripts/serve.sh qwen" >&2
    exit 1
fi
MODEL=$(python3 -c "
import json; d=json.load(open('/tmp/sweep_models.json'))
ids=[m['id'] for m in d.get('data',[])]; print(ids[0] if ids else '')")
[[ -n "$MODEL" ]] || { echo "ERROR: no model loaded" >&2; exit 1; }

MODEL_TAG=$(basename "$MODEL" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9._-' '_')

read -r -a TURN_ARR <<< "$TURNS"
read -r -a WIN_ARR <<< "$WINDOWS"
read -r -a COND_ARR <<< "$CONDITIONS"

TOTAL_CELLS=$(( ${#TURN_ARR[@]} * ${#WIN_ARR[@]} ))

echo "=========================================="
echo "Horizon sweep"
echo "  Model:       $MODEL"
echo "  Turns:       ${TURN_ARR[*]}"
echo "  Windows:     ${WIN_ARR[*]}"
echo "  Conditions:  ${COND_ARR[*]}"
echo "  N per cell:  $N_SAMPLES"
echo "  Concurrency: $CONCURRENCY"
echo "  Cells:       $TOTAL_CELLS"
echo "  Output:      $RESULTS_ROOT/$MODEL_TAG"
echo "=========================================="

# Rough estimate from measured throughput (~308 tok/s at concurrency 32).
EST=$(python3 -c "
turns=[${TURNS// /,}]; wins=${#WIN_ARR[@]}; conds=${#COND_ARR[@]}
tot=sum(${N_SAMPLES}*conds*(t*250+48)*(1000/308)/1000 for t in turns)*wins
print(f'{tot/3600:.1f}')")
echo ""
echo "Estimated runtime: ~${EST} h (this model only)"
echo ""
# ASSUME_YES skips the prompt, for running detached (nohup, tmux, cron).
# Interactive use still gets it: the estimate above is worth reading before
# committing a GPU for most of a day.
if [[ "${ASSUME_YES:-0}" != "1" ]]; then
    read -rp "Proceed? [y/N] " confirm
    [[ "${confirm,,}" == "y" ]] || { echo "Aborted."; exit 0; }
else
    echo "ASSUME_YES=1 — proceeding without confirmation."
fi

# The embedder runs on CPU by default so it does not contend with vLLM for
# device memory — a sweep died four cells in when it tried to allocate on a
# GPU that vLLM had already reserved 90% of. Report headroom anyway: if it is
# very tight, generation itself may struggle on the longer cells, and that is
# better known now than hours from now.
export SK_EMBED_DEVICE="${SK_EMBED_DEVICE:-cpu}"
if command -v nvidia-smi >/dev/null; then
    GPU_LINE=$(nvidia-smi --query-gpu=memory.used,memory.total \
        --format=csv,noheader,nounits | head -1)
    GPU_USED=$(echo "$GPU_LINE" | cut -d',' -f1 | tr -d ' ')
    GPU_TOTAL=$(echo "$GPU_LINE" | cut -d',' -f2 | tr -d ' ')
    GPU_FREE=$(( GPU_TOTAL - GPU_USED ))
    echo "[gpu] ${GPU_USED} MiB used of ${GPU_TOTAL} MiB — ${GPU_FREE} MiB free"
    echo "[gpu] embedder device: $SK_EMBED_DEVICE"
    if (( GPU_FREE < 2000 )); then
        echo "      Headroom is thin; long-turn cells may be tight." >&2
    fi
fi
cd "$ROOT/sk"
CELL=0
STARTED=$(date +%s)

for win in "${WIN_ARR[@]}"; do
    case "$win" in
        16k) BUDGET=$BUDGET_16k ;;
        4k)  BUDGET=$BUDGET_4k ;;
        *)   echo "ERROR: unknown window '$win' (expected 16k or 4k)" >&2; exit 2 ;;
    esac

    for turns in "${TURN_ARR[@]}"; do
        CELL=$((CELL + 1))
        OUT="$RESULTS_ROOT/$MODEL_TAG/win${win}_turns${turns}"

        if [[ -f "$OUT/exp3.jsonl" ]]; then
            # A cell is only done if it contains every condition being asked
            # for now. Checking file existence alone silently skips cells that
            # were completed under a shorter condition list, so adding a
            # condition to an existing sweep would appear to succeed while
            # producing no new data for it.
            MISSING=$(python3 -c "
import json, sys
want = set(sys.argv[1:])
have = set()
for line in open('$OUT/exp3.jsonl'):
    line = line.strip()
    if line:
        have.add(json.loads(line)['condition'])
print(' '.join(sorted(want - have)))
" "${COND_ARR[@]}")
            if [[ -z "$MISSING" ]]; then
                echo "[$CELL/$TOTAL_CELLS] win=$win turns=$turns — already done, skipping"
                continue
            fi
            echo "[$CELL/$TOTAL_CELLS] win=$win turns=$turns — re-running, missing: $MISSING"
            # Preserve the earlier data: a re-run overwrites exp3.jsonl, and
            # losing 18 hours of results to a rerun would be unrecoverable.
            mv "$OUT/exp3.jsonl" "$OUT/exp3.previous.jsonl"
        fi

        echo ""
        echo "[$CELL/$TOTAL_CELLS] win=$win turns=$turns budget=${BUDGET}ch"
        mkdir -p "$OUT"

        # Show progress lines but keep the full output in the log. Filtering
        # the terminal is fine; filtering away a traceback is not, so failures
        # are detected explicitly rather than left to scroll past.
        if python3 runner.py \
            --base-url "$BASE_URL" --model "$MODEL" \
            --data-path "$DATA_PATH" \
            --n-samples "$N_SAMPLES" \
            --conditions "${COND_ARR[@]}" \
            ${EMBED_ARGS[@]+"${EMBED_ARGS[@]}"} \
            --exp 3 \
            --num-turns "$turns" \
            --prompt-budget-chars "$BUDGET" \
            --distractor-tier mixed \
            --concurrency "$CONCURRENCY" \
            --output-dir "$OUT" --seed 42 \
            > "$OUT/run.log" 2>&1
        then
            tail -3 "$OUT/run.log" | sed 's/^/    /'
        else
            echo "" >&2
            echo "  ERROR: cell failed. Last 20 lines of $OUT/run.log:" >&2
            tail -20 "$OUT/run.log" | sed 's/^/    /' >&2
            echo "" >&2
            echo "  Stopping: later cells would fail the same way." >&2
            exit 1
        fi

        # Surface how far each condition degraded, per cell rather than only
        # at the end — a cell where every condition drops turns is one where
        # the comparison has stopped being about protocol.
        python3 -c "
import json, collections
rows=[json.loads(l) for l in open('$OUT/exp3.jsonl')]
d=collections.defaultdict(list)
for r in rows: d[r['condition']].append(r.get('turns_dropped',0))
for c in sorted(d):
    v=d[c]; n=sum(1 for x in v if x>0)
    print(f'    {c:<12} dropped turns in {n}/{len(v)} runs, max {max(v)}')
" 2>/dev/null || true
    done
done

ELAPSED=$(( $(date +%s) - STARTED ))
echo ""
echo "=========================================="
echo "Sweep complete in $((ELAPSED / 3600))h $(((ELAPSED % 3600) / 60))m"
echo "  Results: $RESULTS_ROOT/$MODEL_TAG"
echo ""
echo "Next:"
echo "  python3 sk/crossover.py --root $RESULTS_ROOT/$MODEL_TAG"
echo ""
echo "  Then switch models and repeat:"
echo "    ./scripts/serve.sh gemma"
echo "    ./scripts/sweep.sh"
echo "=========================================="