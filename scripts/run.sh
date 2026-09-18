#!/usr/bin/env bash
# run.sh — Run the experiments against a local vLLM server.
#
# Works identically online and offline. The only difference is where the
# dataset comes from: online it is fetched from HuggingFace and cached;
# offline it must already be present (staged by bundle_create.sh).
#
# Usage:
#   ./scripts/run.sh                       # pilot, N=20
#   MODE=full ./scripts/run.sh             # full run, N=100
#   ./scripts/run.sh --offline             # air-gapped
#   CROSS_VALIDATE=1 ./scripts/run.sh      # also check rules against an API judge
#
# Results are written per-model, so running once per model produces the
# cross-model replication data that compare.py expects.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../models.conf
source "$ROOT/models.conf"

OFFLINE=0
EXP="all"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --offline) OFFLINE=1; shift ;;
        --exp)     EXP="$2"; shift 2 ;;
        --exp=*)   EXP="${1#*=}"; shift ;;
        *)         shift ;;
    esac
done
(( OFFLINE )) && export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

BASE_URL="${VLLM_BASE_URL:-http://localhost:${VLLM_PORT_DEFAULT}/v1}"
DATA_DIR="${SK_DATA:-$ROOT/data}"
DATA_PATH="$DATA_DIR/$HOTPOTQA_LOCAL_FILE"
RESULTS_ROOT="${SK_RESULTS:-$ROOT/results}"
EMBEDDER="${SK_EMBED_MODEL:-}"
CROSS_VALIDATE="${CROSS_VALIDATE:-0}"

MODE="${MODE:-pilot}"
case "$MODE" in
    pilot)
        N_SAMPLES="${N_SAMPLES:-20}"
        DEFAULT_CONDITIONS="NL-free NL-concise SK"
        NUM_TURNS="${NUM_TURNS:-6}" ;;
    full)
        N_SAMPLES="${N_SAMPLES:-100}"
        # SK-minimal is included: the pilot showed the six-field spec costs
        # tokens without an accuracy return, so the ablation is part of the
        # confirmatory design rather than a follow-up.
        DEFAULT_CONDITIONS="NL-free NL-concise NL-CoT JSON-free SK SK-minimal"
        NUM_TURNS="${NUM_TURNS:-8}" ;;
    *)
        echo "ERROR: MODE must be 'pilot' or 'full'" >&2; exit 2 ;;
esac

# Word-split on purpose so callers can pass a space-separated list:
#   CONDITIONS="NL-free SK SK-minimal" ./scripts/run.sh
read -r -a CONDITIONS <<< "${CONDITIONS:-$DEFAULT_CONDITIONS}"

mkdir -p "$DATA_DIR"

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
if [[ -f "$DATA_PATH" ]]; then
    N_AVAIL=$(python3 -c "import json,sys;print(len(json.load(open(sys.argv[1]))))" "$DATA_PATH")
    echo "[data] $DATA_PATH — $N_AVAIL records"
elif (( OFFLINE )); then
    echo "ERROR: dataset not found at $DATA_PATH" >&2
    echo "Offline mode cannot download it. Stage it with bundle_create.sh" >&2
    echo "on an online machine, then bundle_install.sh here." >&2
    exit 1
else
    echo "[data] Downloading HotpotQA (first run only)..."
    python3 - "$DATA_PATH" "$HOTPOTQA_HF_REPO" "$HOTPOTQA_HF_CONFIG" "$HOTPOTQA_HF_SPLIT" <<'PYEOF'
import json, sys
out, repo, cfg, split = sys.argv[1:5]
from datasets import load_dataset
ds = load_dataset(repo, cfg, split=split)
recs = [{
    "_id": r.get("id", ""), "question": r["question"], "answer": r["answer"],
    "type": r.get("type", "bridge"), "level": r.get("level", "hard"),
    "context": {"title": list(r["context"]["title"]),
                "sentences": [list(s) for s in r["context"]["sentences"]]},
    "supporting_facts": {"title": list(r["supporting_facts"]["title"]),
                         "sent_id": list(r["supporting_facts"]["sent_id"])},
} for r in ds]
json.dump(recs, open(out, "w"), ensure_ascii=False)
print(f"  Cached {len(recs)} records")
PYEOF
    N_AVAIL=$(python3 -c "import json,sys;print(len(json.load(open(sys.argv[1]))))" "$DATA_PATH")
fi
(( N_AVAIL < N_SAMPLES )) && \
    echo "  WARNING: requested N=$N_SAMPLES but only $N_AVAIL available." >&2

# ---------------------------------------------------------------------------
# Server preflight
# ---------------------------------------------------------------------------
echo "[preflight] Querying vLLM at $BASE_URL ..."
if ! curl -sf "$BASE_URL/models" -o /tmp/sk_models.json --max-time 10; then
    echo "ERROR: vLLM unreachable at $BASE_URL" >&2
    echo "" >&2
    HOSTPORT="${BASE_URL#*://}"; HOSTPORT="${HOSTPORT%%/*}"
    H="${HOSTPORT%%:*}"; P="${HOSTPORT##*:}"
    if curl -s --max-time 3 "http://$H:$P/" >/dev/null 2>&1; then
        echo "  Something is listening on $H:$P but /v1/models did not answer." >&2
        echo "  Check that it is vLLM and not another service on that port." >&2
    else
        echo "  Nothing is listening on $H:$P. Common causes:" >&2
        echo "" >&2
        echo "  1. The server exited on startup. A rejected command-line flag" >&2
        echo "     makes vllm quit before binding the port — check the serve.sh" >&2
        echo "     terminal for 'unrecognized arguments'. Otherwise wait for" >&2
        echo "     'Uvicorn running on ...'; large models take minutes to load." >&2
        echo "" >&2
        echo "  2. Different containers. localhost does not cross container" >&2
        echo "     boundaries — set VLLM_BASE_URL to the serving container:" >&2
        echo "       VLLM_BASE_URL=http://<container-name>:8000/v1 $0" >&2
        echo "     with both containers on the same Docker network." >&2
        echo "" >&2
        echo "  Start the server with:" >&2
        echo "    ./scripts/serve.sh qwen$( (( OFFLINE )) && echo ' --offline')" >&2
    fi
    exit 1
fi
MODEL=$(python3 -c "
import json; d=json.load(open('/tmp/sk_models.json'))
ids=[m['id'] for m in d.get('data',[])]; print(ids[0] if ids else '')")
[[ -n "$MODEL" ]] || { echo "ERROR: no model loaded." >&2; exit 1; }
echo "  Loaded: $MODEL"

MODEL_TAG=$(basename "$MODEL" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9._-' '_')
OUT_DIR="$RESULTS_ROOT/$MODE/$MODEL_TAG"
mkdir -p "$OUT_DIR"

_chat() {
    curl -s -X POST "$BASE_URL/chat/completions" \
        -H "Content-Type: application/json" -d "$1" --max-time 90
}

echo "[preflight] Generation ..."
# Parse the response rather than grepping the raw envelope: JSON escaping and
# surrounding fields make substring matching on the wire format unreliable,
# which previously reported a correct 'PONG' as a failure.
GEN_RESP=$(_chat "{\"model\":\"$MODEL\",\"max_tokens\":16,\"temperature\":0,
    \"chat_template_kwargs\":{\"enable_thinking\":false},
    \"messages\":[{\"role\":\"system\",\"content\":\"Reply with exactly: PONG\"},
                  {\"role\":\"user\",\"content\":\"Reply.\"}]}")
GEN_CONTENT=$(python3 -c "
import json,sys
try:
    d=json.load(sys.stdin)
    print(d['choices'][0]['message']['content'].strip())
except Exception:
    print('')" <<<"$GEN_RESP")

if [[ "${GEN_CONTENT^^}" == *PONG* ]]; then
    echo "  OK"
else
    echo "  WARNING: expected PONG, got: ${GEN_CONTENT:0:80}" >&2
    echo "  If this is empty, the server may be erroring — check its log." >&2
fi

# Reasoning-capable models emit chain-of-thought as ordinary output unless
# told otherwise, which inflates Experiment 1 token counts and pushes real
# answers past max_tokens. The client sends enable_thinking=false, but not
# every model honours that key, so verify the effect rather than the setting.
#
# Checking for "<think>" alone is not enough: models also emit plain prose
# like "Thinking Process:" with no markup at all. Truncation is the sturdier
# signal — a 200-token budget is ample for an arithmetic question, so hitting
# the cap means something is padding the output.
echo "[preflight] Reasoning tokens ..."
THINK_RESP=$(_chat "{\"model\":\"$MODEL\",\"max_tokens\":200,\"temperature\":0,
    \"chat_template_kwargs\":{\"enable_thinking\":false},
    \"messages\":[{\"role\":\"system\",\"content\":\"Answer with the number only.\"},
                  {\"role\":\"user\",\"content\":\"What is 17 times 23?\"}]}")

THINK_VERDICT=$(python3 -c "
import json, sys, re
try:
    d = json.load(sys.stdin)
    ch = d['choices'][0]
    msg = ch.get('message', {}) or {}
    content = (msg.get('content') or '').strip()
    reasoning = msg.get('reasoning') or msg.get('reasoning_content') or ''
    finish = ch.get('finish_reason', '')
    ntok = (d.get('usage') or {}).get('completion_tokens', 0)
except Exception as e:
    print(f'PARSE_FAIL|{e}'); raise SystemExit

problems = []
if finish == 'length':
    problems.append(f'output hit the {ntok}-token cap (finish_reason=length)')
if reasoning:
    problems.append('backend returned a separate reasoning field')
if re.search(r'<think>|thinking process|let me think|step 1[:.]', content, re.I):
    problems.append('answer text contains reasoning preamble')
if ntok > 40:
    problems.append(f'{ntok} tokens for a one-number answer')

print(('BAD|' if problems else 'OK|') + '; '.join(problems) + '|' + content[:80])
" <<<"$THINK_RESP")

THINK_STATUS="${THINK_VERDICT%%|*}"
THINK_DETAIL=$(cut -d'|' -f2 <<<"$THINK_VERDICT")
THINK_SAMPLE=$(cut -d'|' -f3 <<<"$THINK_VERDICT")

if [[ "$THINK_STATUS" == "OK" ]]; then
    echo "  Clean — answered in few tokens without reasoning"
else
    echo "  WARNING: reasoning output appears to be enabled." >&2
    echo "    $THINK_DETAIL" >&2
    echo "    Response began: $THINK_SAMPLE" >&2
    echo "" >&2
    echo "  This invalidates Experiment 1: reasoning tokens are counted as" >&2
    echo "  protocol cost, and long reasoning truncates the real answer," >&2
    echo "  which zeroes every accuracy metric downstream." >&2
    echo "" >&2
    echo "  The client already sends chat_template_kwargs.enable_thinking=false." >&2
    echo "  If that is not working for this model, check its model card for" >&2
    echo "  the correct switch and set it via extra_body in sk/client.py." >&2
    echo "" >&2
    read -rp "  Continue anyway? [y/N] " c
    [[ "${c,,}" == "y" ]] || exit 1
fi

# Experiment 3 is the only stage that loads the embedder. Without this check
# a broken install surfaces only after experiments 1 and 2 have run, which on
# a full run is hours of wasted compute.
echo "[preflight] Embedder ..."
EMBED_TARGET="${EMBEDDER:-sentence-transformers/all-mpnet-base-v2}"
if python3 - "$EMBED_TARGET" <<'PYEOF' 2>/tmp/sk_embed_err.txt
import sys
from sentence_transformers import SentenceTransformer
m = SentenceTransformer(sys.argv[1])
v = m.encode("smoke test", convert_to_numpy=True, normalize_embeddings=True)
assert v.ndim == 1 and v.shape[0] > 0
print(f"  OK (dim {v.shape[0]})")
PYEOF
then
    :
else
    echo "  ERROR: embedder failed to load." >&2
    tail -15 /tmp/sk_embed_err.txt >&2
    if grep -q "libavutil\|torchcodec" /tmp/sk_embed_err.txt; then
        echo "" >&2
        echo "  datasets 4.x imports torchcodec, which needs FFmpeg libraries" >&2
        echo "  this container lacks. We use text only, so:" >&2
        echo "    pip install 'datasets<4.0'        # preferred" >&2
        echo "    apt-get install -y ffmpeg          # alternative" >&2
    fi
    exit 1
fi

echo "[preflight] Constrained decoding ..."
# The older vLLM-specific `guided_json` parameter was removed; servers that
# lack it accept the request and silently return free prose. That is exactly
# what broke a previous pilot — 61% of verdicts unparseable with no error
# anywhere. So test the parameter we actually send, and judge by whether the
# output parses, not by whether the request succeeded.
CD_RESP=$(_chat "{\"model\":\"$MODEL\",\"max_tokens\":128,\"temperature\":0,
    \"chat_template_kwargs\":{\"enable_thinking\":false},
    \"messages\":[{\"role\":\"system\",\"content\":\"Emit a verdict.\"},
                  {\"role\":\"user\",\"content\":\"Is 2+2=5?\"}],
    \"response_format\":{\"type\":\"json_schema\",
      \"json_schema\":{\"name\":\"verifier_verdict\",\"strict\":true,
        \"schema\":{\"type\":\"object\",
          \"properties\":{\"verified\":{\"type\":\"boolean\"},
                          \"reasons\":{\"type\":\"array\",\"items\":{\"type\":\"string\"}}},
          \"required\":[\"verified\",\"reasons\"],
          \"additionalProperties\":false}}}}")

if python3 -c "
import json,sys
try:
    d=json.load(sys.stdin)
    obj=json.loads(d['choices'][0]['message']['content'])
    sys.exit(0 if isinstance(obj.get('verified'), bool)
                 and isinstance(obj.get('reasons'), list) else 1)
except Exception:
    sys.exit(1)" <<<"$CD_RESP" 2>/dev/null; then
    echo "  OK — schema honoured"
    GUIDED_FLAG=""
else
    echo "  WARNING: the schema was not applied." >&2
    echo "  Response began: $(python3 -c "
import json,sys
try:
    d=json.load(sys.stdin)
    print(repr(d['choices'][0]['message']['content'][:120]))
except Exception as e:
    print(f'(unparseable response: {e})')" <<<"$CD_RESP")" >&2
    echo "" >&2
    echo "  Verdicts will fall back to prose parsing. Expect a substantial" >&2
    echo "  unparsed rate, and note that it will not fall equally across" >&2
    echo "  conditions: SK emits JSON natively while NL must append it, so" >&2
    echo "  the detection comparison becomes a test of format compliance" >&2
    echo "  rather than of protocol. Treat Experiment 2 as uninterpretable" >&2
    echo "  unless this passes." >&2
    GUIDED_FLAG="--no-guided-decoding"
fi

# ---------------------------------------------------------------------------
# Confirm
# ---------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "Configuration ($MODE, $( (( OFFLINE )) && echo OFFLINE || echo ONLINE ))"
echo "  Model:        $MODEL"
echo "  N samples:    $N_SAMPLES"
echo "  Conditions:   ${CONDITIONS[*]}"
echo "  Exp3 turns:   $NUM_TURNS"
echo "  Experiments:  $EXP"
echo "  Output:       $OUT_DIR"
echo "  Scoring:      deterministic (no LLM judge)"
echo "=========================================="
[[ -f "$OUT_DIR/exp1.jsonl" ]] && \
    echo "" && echo "WARNING: existing results here will be overwritten." >&2
echo ""
read -rp "Proceed? [y/N] " confirm
[[ "${confirm,,}" == "y" ]] || { echo "Aborted."; exit 0; }

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
cd "$ROOT/sk"
EMBED_ARG=()
[[ -n "$EMBEDDER" ]] && EMBED_ARG=(--embedder-path "$EMBEDDER")

echo ""
echo "[1/4] Experiments ..."
python3 runner.py \
    --base-url "$BASE_URL" --model "$MODEL" \
    --data-path "$DATA_PATH" "${EMBED_ARG[@]}" \
    --n-samples "$N_SAMPLES" --conditions "${CONDITIONS[@]}" \
    --exp "$EXP" --num-turns "$NUM_TURNS" --distractor-tier mixed \
    --output-dir "$OUT_DIR" --seed 42 $GUIDED_FLAG \
    2>&1 | tee "$OUT_DIR/run.log"

echo ""
echo "[2/4] Hypothesis tests ..."
python3 analyze.py --results-dir "$OUT_DIR" \
    --output "$OUT_DIR/analysis.txt" --alpha 0.05 \
    2>&1 | tee -a "$OUT_DIR/run.log"

echo ""
echo "[3/4] Sanity checks ..."
python3 sanity.py "$OUT_DIR" 2>&1 | tee -a "$OUT_DIR/run.log" || true

echo ""
if [[ "$CROSS_VALIDATE" == "1" ]] && (( ! OFFLINE )); then
    echo "[4/4] Cross-validating rules against an API judge ..."
    if [[ -z "${ANTHROPIC_API_KEY:-}${OPENAI_API_KEY:-}${GOOGLE_API_KEY:-}" ]]; then
        echo "  SKIPPED: no judge API key set." >&2
    else
        python3 judge_check.py --results-dir "$OUT_DIR" \
            --output "$OUT_DIR/rule_validation.txt" \
            2>&1 | tee -a "$OUT_DIR/run.log"
    fi
else
    echo "[4/4] Rule cross-validation skipped."
fi

# ---------------------------------------------------------------------------
DONE_COUNT=$(find "$RESULTS_ROOT/$MODE" -name exp1.jsonl 2>/dev/null | wc -l)
echo ""
echo "=========================================="
echo "Done. Results in $OUT_DIR"
echo "Models completed in this $MODE run: $DONE_COUNT"
if (( DONE_COUNT < 2 )); then
    echo ""
    echo "Next: switch models and re-run for cross-model replication."
    echo "  ./scripts/serve.sh gemma$( (( OFFLINE )) && echo ' --offline')"
    echo "  ./scripts/run.sh$( (( OFFLINE )) && echo ' --offline')"
else
    echo ""
    echo "Both models done. Compare:"
    echo "  python3 sk/compare.py --root $RESULTS_ROOT/$MODE \\"
    echo "      --output $RESULTS_ROOT/$MODE/comparison.txt"
fi
echo "=========================================="