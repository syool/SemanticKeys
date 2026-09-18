#!/usr/bin/env bash
# online_dryrun.sh — Exercise the entire pipeline in minutes with a small model.
#
# WHY THIS EXISTS
# A full run takes 5-9 hours per model. A bug that surfaces in hour six is
# expensive, and if it surfaces on the offline machine it is expensive AND
# hard to diagnose. This script runs every code path end to end with a 4B
# model on N=3 samples, so pipeline bugs surface in about five minutes while
# you still have internet to look things up.
#
# It is NOT a scientific run. The numbers are meaningless. What it verifies:
#   - vLLM serves and responds
#   - guided decoding produces parseable verdicts
#   - all three experiments complete without exceptions
#   - the embedder loads and produces drift scores
#   - JSONL outputs are well formed
#   - analyze.py, sanity.py, and compare.py all consume them
#   - validation sheet generation works
#
# Usage:
#   ./scripts/serve.sh dryrun      # terminal A — starts the 4B model
#   ./online_dryrun.sh           # terminal B

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="$ROOT/sk"
# shellcheck source=models.conf
source "$ROOT/models.conf"

BASE_URL="${VLLM_BASE_URL:-http://localhost:${VLLM_PORT_DEFAULT}/v1}"
OUT_DIR="${DRYRUN_OUT:-$ROOT/results/dryrun}"
DATA_CACHE="$ROOT/data"

FAILURES=0
step() { echo ""; echo "── $* ──"; }
pass() { echo "   PASS: $*"; }
fail() { echo "   FAIL: $*" >&2; FAILURES=$((FAILURES + 1)); }

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR" "$DATA_CACHE"

echo "=========================================="
echo "Pipeline dry run"
echo "  Server:  $BASE_URL"
echo "  Output:  $OUT_DIR"
echo "  Purpose: catch bugs, not produce results"
echo "=========================================="

# ---------------------------------------------------------------------------
step "1. Server reachable"
if curl -sf "$BASE_URL/models" -o /tmp/dryrun_models.json --max-time 10; then
    LOADED_MODEL=$(python3 -c "
import json; d=json.load(open('/tmp/dryrun_models.json'))
ids=[m['id'] for m in d.get('data',[])]; print(ids[0] if ids else '')")
    if [[ -n "$LOADED_MODEL" ]]; then
        pass "model loaded: $LOADED_MODEL"
    else
        fail "server responded but reports no models"; exit 1
    fi
else
    fail "server unreachable at $BASE_URL"
    echo "" >&2
    echo "   Diagnosis:" >&2
    # Distinguish "nothing is listening" from "listening but wrong path",
    # since those have completely different fixes.
    HOSTPORT="${BASE_URL#*://}"; HOSTPORT="${HOSTPORT%%/*}"
    H="${HOSTPORT%%:*}"; P="${HOSTPORT##*:}"
    if command -v curl >/dev/null && curl -s --max-time 3 "http://$H:$P/" >/dev/null 2>&1; then
        echo "     Something IS listening on $H:$P but /v1/models did not answer." >&2
        echo "     Check that it is vLLM and not another service on that port." >&2
    else
        echo "     Nothing is listening on $H:$P." >&2
        echo "" >&2
        echo "     1. Did serve.sh actually start? A rejected command-line flag" >&2
        echo "        makes vllm exit before binding the port. Scroll up in the" >&2
        echo "        serve.sh terminal and look for 'unrecognized arguments'." >&2
        echo "        Wait for the 'Uvicorn running on ...' line before running" >&2
        echo "        this script — large models take minutes to load." >&2
        echo "" >&2
        echo "     2. Are the two scripts in the SAME container?" >&2
        echo "        localhost does not cross container boundaries. If serve.sh" >&2
        echo "        runs in another container, point this script at it:" >&2
        echo "          VLLM_BASE_URL=http://<container-name>:8000/v1 $0" >&2
        echo "        and make sure both are on the same Docker network." >&2
        echo "" >&2
        echo "     3. Is the port published? If vLLM runs on the host or in" >&2
        echo "        another container, that container needs -p 8000:8000," >&2
        echo "        and vLLM must bind 0.0.0.0 (serve.sh already does)." >&2
    fi
    exit 1
fi

# ---------------------------------------------------------------------------
step "2. Basic generation"
R=$(curl -s -X POST "$BASE_URL/chat/completions" -H "Content-Type: application/json" \
    -d "{\"model\":\"$LOADED_MODEL\",\"max_tokens\":16,\"temperature\":0,
         \"messages\":[{\"role\":\"system\",\"content\":\"Reply with exactly: PONG\"},
                       {\"role\":\"user\",\"content\":\"Reply.\"}]}" --max-time 60)
echo "$R" | grep -qi pong && pass "model follows a simple instruction" \
    || fail "unexpected: $(echo "$R" | head -c 200)"

# ---------------------------------------------------------------------------
step "3. Guided decoding"
G=$(curl -s -X POST "$BASE_URL/chat/completions" -H "Content-Type: application/json" \
    -d "{\"model\":\"$LOADED_MODEL\",\"max_tokens\":128,\"temperature\":0,
         \"chat_template_kwargs\":{\"enable_thinking\":false},
         \"messages\":[{\"role\":\"system\",\"content\":\"Emit a verdict.\"},
                       {\"role\":\"user\",\"content\":\"Is 2+2=5?\"}],
         \"response_format\":{\"type\":\"json_schema\",
           \"json_schema\":{\"name\":\"verifier_verdict\",\"strict\":true,
             \"schema\":{\"type\":\"object\",
               \"properties\":{\"verified\":{\"type\":\"boolean\"},
                               \"reasons\":{\"type\":\"array\",\"items\":{\"type\":\"string\"}}},
               \"required\":[\"verified\",\"reasons\"],
               \"additionalProperties\":false}}}}" --max-time 60)
if echo "$G" | python3 -c "
import json,sys
try:
    obj=json.loads(json.load(sys.stdin)['choices'][0]['message']['content'])
    sys.exit(0 if isinstance(obj.get('verified'), bool) else 1)
except Exception: sys.exit(1)" 2>/dev/null; then
    pass "constrained decoding returns schema-conformant JSON"
    GUIDED_FLAG=""
else
    fail "schema not applied — Experiment 2 will not be interpretable"
    GUIDED_FLAG="--no-guided-decoding"
fi

# ---------------------------------------------------------------------------
step "4. Reasoning-token check"
T=$(curl -s -X POST "$BASE_URL/chat/completions" -H "Content-Type: application/json" \
    -d "{\"model\":\"$LOADED_MODEL\",\"max_tokens\":200,\"temperature\":0,
         \"messages\":[{\"role\":\"system\",\"content\":\"Answer concisely.\"},
                       {\"role\":\"user\",\"content\":\"What is 17 times 23?\"}]}" \
    --max-time 60)
if echo "$T" | grep -qE '<think>|reasoning_content'; then
    fail "reasoning tokens present — these confound Experiment 1 token counts"
else
    pass "no reasoning tokens in output"
fi

# ---------------------------------------------------------------------------
step "5. Dataset available"
DATA_PATH="$DATA_CACHE/$HOTPOTQA_LOCAL_FILE"
if [[ -f "$DATA_PATH" ]]; then
    N=$(python3 -c "import json,sys;print(len(json.load(open(sys.argv[1]))))" "$DATA_PATH")
    pass "cached dataset: $N records"
else
    echo "   Downloading HotpotQA (first run)..."
    if python3 - "$DATA_PATH" "$HOTPOTQA_HF_REPO" "$HOTPOTQA_HF_CONFIG" "$HOTPOTQA_HF_SPLIT" <<'PYEOF'
import json, sys
out, repo, cfg, split = sys.argv[1:5]
from datasets import load_dataset
ds = load_dataset(repo, cfg, split=split)
recs = [{
    "_id": r.get("id", ""), "question": r["question"], "answer": r["answer"],
    "type": r.get("type","bridge"), "level": r.get("level","hard"),
    "context": {"title": list(r["context"]["title"]),
                "sentences": [list(s) for s in r["context"]["sentences"]]},
    "supporting_facts": {"title": list(r["supporting_facts"]["title"]),
                         "sent_id": list(r["supporting_facts"]["sent_id"])},
} for r in ds]
json.dump(recs, open(out,"w"), ensure_ascii=False)
print(f"   {len(recs)} records cached")
PYEOF
    then pass "dataset downloaded"; else fail "dataset download failed"; fi
fi

# ---------------------------------------------------------------------------
step "5b. Embedder loads"
# Experiment 3 is the only stage that touches the embedder, so a broken
# install here would otherwise surface only after experiments 1 and 2 have
# already run. On a full run that is hours of wasted compute, so load it now.
EMBED_ARG_TEST="${SK_EMBED_MODEL:-sentence-transformers/all-mpnet-base-v2}"
if python3 - "$EMBED_ARG_TEST" <<'PYEOF' 2>/tmp/embed_err.txt
import sys
from sentence_transformers import SentenceTransformer
m = SentenceTransformer(sys.argv[1])
v = m.encode("smoke test", convert_to_numpy=True, normalize_embeddings=True)
assert v.ndim == 1 and v.shape[0] > 0, f"unexpected embedding shape {v.shape}"
print(f"   embedding dim {v.shape[0]}")
PYEOF
then
    pass "embedder loaded and produced a vector"
else
    fail "embedder failed to load"
    echo "   ---" >&2
    tail -20 /tmp/embed_err.txt >&2
    echo "   ---" >&2
    # This specific failure is common enough in slim containers to name.
    if grep -q "libavutil\|torchcodec" /tmp/embed_err.txt; then
        echo "   This is the torchcodec/FFmpeg problem: datasets 4.x imports" >&2
        echo "   torchcodec, which needs FFmpeg shared libraries this container" >&2
        echo "   lacks. We only do text, so the dependency is unnecessary." >&2
        echo "" >&2
        echo "   Fix (preferred, keeps the offline bundle wheel-only):" >&2
        echo "     pip install 'datasets<4.0'" >&2
        echo "" >&2
        echo "   Fix (alternative, adds an apt dependency):" >&2
        echo "     apt-get update && apt-get install -y ffmpeg" >&2
    fi
fi

# ---------------------------------------------------------------------------
step "6. Full experiment pipeline (N=3)"
cd "$SRC_DIR"
if python3 runner.py \
    --base-url "$BASE_URL" --model "$LOADED_MODEL" \
    --data-path "$DATA_PATH" --n-samples 3 \
    --conditions NL-free SK --exp all --num-turns 3 \
    --output-dir "$OUT_DIR" --seed 42 $GUIDED_FLAG \
    > "$OUT_DIR/run.log" 2>&1; then
    pass "all three experiments completed"
else
    fail "runner exited non-zero — see $OUT_DIR/run.log"
    tail -25 "$OUT_DIR/run.log" >&2
fi

# ---------------------------------------------------------------------------
step "7. Output integrity"
for f in exp1 exp2 exp3; do
    path="$OUT_DIR/$f.jsonl"
    if [[ -f "$path" ]]; then
        if python3 -c "
import json,sys
rows=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
assert rows, 'empty'
print(f'   {sys.argv[1].split(\"/\")[-1]}: {len(rows)} rows, '
      f'{len(rows[0])} fields')" "$path"; then
            pass "$f.jsonl well formed"
        else
            fail "$f.jsonl malformed or empty"
        fi
    else
        fail "$f.jsonl not written"
    fi
done

# Verify the reconstruction stage actually ran — a silent failure here would
# make Exp 1 report zeros without any obvious error.
python3 -c "
import json, sys
rows=[json.loads(l) for l in open('$OUT_DIR/exp1.jsonl') if l.strip()]
empty=[r for r in rows if not r.get('reconstructed_answer')]
if empty:
    print(f'   WARNING: {len(empty)}/{len(rows)} rows have empty reconstructions')
    sys.exit(1)
print(f'   all {len(rows)} reconstructions non-empty')" \
    && pass "reconstruction stage produced output" \
    || fail "reconstruction stage produced empty answers"

# Verify drift scores are in a plausible range — but only if exp3 actually
# ran. Reporting a missing file as "drift out of range" sends you looking at
# the embedder when the real problem is upstream.
if [[ -f "$OUT_DIR/exp3.jsonl" ]]; then
    python3 -c "
import json, sys
rows=[json.loads(l) for l in open('$OUT_DIR/exp3.jsonl') if l.strip()]
if not rows:
    print('   exp3.jsonl is empty'); sys.exit(1)
bad=[r for r in rows if not (0.0 <= r['mean_drift'] <= 2.0)]
if bad:
    print(f'   {len(bad)} rows with implausible drift'); sys.exit(1)
ds=[r['mean_drift'] for r in rows]
print(f'   drift range: {min(ds):.3f} to {max(ds):.3f}')" \
        && pass "drift scores plausible" \
        || fail "drift scores out of range — check the embedder"
else
    fail "exp3 did not produce output — drift check skipped (see step 6 errors)"
fi

# ---------------------------------------------------------------------------
step "8. Downstream tools consume the outputs"
python3 analyze.py --results-dir "$OUT_DIR" --output "$OUT_DIR/analysis.txt" \
    > /dev/null 2>&1 && pass "analyze.py ran" || fail "analyze.py failed"

python3 "sanity.py" "$OUT_DIR" > "$OUT_DIR/sanity.txt" 2>&1
# sanity exits 1 on red flags, which is expected at N=3
[[ -s "$OUT_DIR/sanity.txt" ]] && pass "sanity.py produced output" \
    || fail "sanity.py produced nothing"

python3 validation.py --results-dir "$OUT_DIR" \
    --output-dir "$OUT_DIR/validation" --n-propagation 3 --n-detection 3 \
    --n-ontopic 3 > /dev/null 2>&1 \
    && pass "validation sheets generated" || fail "validation.py failed"

# compare needs two model dirs; fake a second to exercise the path.
mkdir -p "$OUT_DIR/../dryrun_compare"/{model_a,model_b}
for m in model_a model_b; do
    cp "$OUT_DIR"/exp*.jsonl "$OUT_DIR/../dryrun_compare/$m/" 2>/dev/null || true
done
python3 compare.py --root "$OUT_DIR/../dryrun_compare" \
    > "$OUT_DIR/compare.txt" 2>&1 \
    && pass "compare.py ran" || fail "compare.py failed"
rm -rf "$OUT_DIR/../dryrun_compare"

# ---------------------------------------------------------------------------
echo ""
echo "=========================================="
if (( FAILURES == 0 )); then
    echo "DRY RUN CLEAN — pipeline is ready."
    echo ""
    echo "Next:"
    echo "  ./scripts/serve.sh qwen      # swap to a real model"
    echo "  ./scripts/run.sh            # pilot at N=20"
else
    echo "DRY RUN FOUND $FAILURES PROBLEM(S)."
    echo "Fix these before starting a long run — debugging is much cheaper now,"
    echo "while you still have internet and a fast model."
fi
echo ""
echo "Artifacts: $OUT_DIR"
echo "=========================================="
exit $(( FAILURES > 0 ? 1 : 0 ))