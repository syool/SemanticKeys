#!/usr/bin/env bash
# serve.sh — Start a vLLM server for the experiments.
#
# Handles both modes:
#   online  (default) — pulls weights from HuggingFace
#   offline (--offline) — loads from local paths under $SK_MODELS
#
# Usage:
#   ./scripts/serve.sh qwen                # online
#   ./scripts/serve.sh gemma --offline     # air-gapped
#   ./scripts/serve.sh dryrun              # small model for pipeline testing
#
# Model identity comes from models.conf. Edit there, not here.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../models.conf
source "$ROOT/models.conf"

CHOICE="${1:-qwen}"
OFFLINE=0
for arg in "$@"; do
    [[ "$arg" == "--offline" ]] && OFFLINE=1
done

PORT="${VLLM_PORT:-$VLLM_PORT_DEFAULT}"
HOST="${VLLM_HOST:-$VLLM_HOST_DEFAULT}"
GPU_UTIL="${VLLM_GPU_UTIL:-$VLLM_GPU_UTIL_DEFAULT}"
GUIDED_BACKEND="${VLLM_GUIDED_BACKEND:-$VLLM_GUIDED_BACKEND_DEFAULT}"
MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-$MAX_NUM_SEQS_DEFAULT}"
MODELS_DIR="${SK_MODELS:-$ROOT/models}"

case "$CHOICE" in
    qwen)
        REPO="$QWEN_REPO";           LOCAL="$QWEN_LOCAL_DIR"
        MAX_LEN="$QWEN_MAX_LEN";     QUANT="$QWEN_QUANT"
        APPROX_GB="$QWEN_APPROX_GB" ;;
    gemma)
        REPO="$GEMMA_REPO";          LOCAL="$GEMMA_LOCAL_DIR"
        MAX_LEN="$GEMMA_MAX_LEN";    QUANT="$GEMMA_QUANT"
        APPROX_GB="$GEMMA_APPROX_GB" ;;
    gemma-e4b)
        REPO="$GEMMA_E4B_REPO";      LOCAL="$GEMMA_E4B_LOCAL_DIR"
        MAX_LEN="$GEMMA_E4B_MAX_LEN"; QUANT="$GEMMA_E4B_QUANT"
        APPROX_GB="$GEMMA_E4B_APPROX_GB" ;;
    gemma-12b)
        REPO="$GEMMA_12B_REPO";      LOCAL="$GEMMA_12B_LOCAL_DIR"
        MAX_LEN="$GEMMA_12B_MAX_LEN"; QUANT="$GEMMA_12B_QUANT"
        APPROX_GB="$GEMMA_12B_APPROX_GB" ;;
    gemma-moe)
        REPO="$GEMMA_MOE_REPO";      LOCAL="$GEMMA_MOE_LOCAL_DIR"
        MAX_LEN="$GEMMA_MOE_MAX_LEN"; QUANT="$GEMMA_MOE_QUANT"
        APPROX_GB="$GEMMA_MOE_APPROX_GB" ;;
    dryrun)
        REPO="$DRYRUN_REPO";         LOCAL="$DRYRUN_LOCAL_DIR"
        MAX_LEN="$DRYRUN_MAX_LEN";   QUANT="$DRYRUN_QUANT"
        APPROX_GB="$DRYRUN_APPROX_GB" ;;
    *)
        echo "Unknown model: $CHOICE" >&2
        echo "Usage: $0 {qwen|gemma|gemma-e4b|gemma-12b|gemma-moe|dryrun} [--offline]" >&2
        exit 2 ;;
esac

if (( OFFLINE )); then
    MODEL="$MODELS_DIR/$LOCAL"
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    if [[ ! -d "$MODEL" ]]; then
        echo "ERROR: model directory not found: $MODEL" >&2
        echo "Available under $MODELS_DIR:" >&2
        ls "$MODELS_DIR" 2>/dev/null | sed 's/^/  /' >&2 || echo "  (none)" >&2
        exit 1
    fi
    MODE_LABEL="OFFLINE (local weights)"
else
    MODEL="$REPO"
    MODE_LABEL="ONLINE (HuggingFace)"
    # Fail fast on a stale or gated repo rather than after a long download.
    # SKIP_VERIFY exists because once the weights are cached this check adds
    # nothing but a dependency on the network being up, and a sweep left
    # running overnight should not die because a hub request timed out. An
    # earlier run lost six hours to exactly that.
    if [[ "${SKIP_VERIFY:-0}" == "1" ]]; then
        echo "[preflight] SKIP_VERIFY=1 — not contacting the hub for $REPO"
    else
        echo "[preflight] Verifying $REPO ..."
        if ! python3 "$ROOT/scripts/verify_models.py" --repo "$REPO"; then
            echo "" >&2
            echo "Repository verification failed. Fix models.conf before serving." >&2
            echo "If the weights are already cached, retry with SKIP_VERIFY=1." >&2
            exit 1
        fi
    fi
fi

# --- GPU headroom check ----------------------------------------------------
if command -v nvidia-smi >/dev/null; then
    VRAM_GB=$(( $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1) / 1024 ))
    echo ""
    echo "[preflight] GPU: ${VRAM_GB} GB VRAM"
    # Weights plus KV cache and activations. Rough guide, not a guarantee.
    NEEDED=$(( APPROX_GB * 13 / 10 ))
    if (( VRAM_GB < NEEDED )); then
        echo "  WARNING: ~${APPROX_GB} GB of weights plus KV cache may not fit." >&2
        echo "  Consider a lower --max-model-len, a smaller" >&2
        echo "  --gpu-memory-utilization, or a quantized variant." >&2
    fi
fi

# --- Guided decoding flag detection ----------------------------------------
# The flag controlling structured-output backends has changed name across
# vLLM releases, and on recent versions xgrammar is the default so no flag is
# needed at all. Passing an unrecognised flag makes vllm exit before it ever
# binds a port, which then shows up downstream as "server unreachable" and
# sends you looking in the wrong place. So: ask vllm what it accepts.
GUIDED_ARGS=()
VLLM_HELP="$(vllm serve --help 2>&1 || true)"

if grep -q -- "--guided-decoding-backend" <<<"$VLLM_HELP"; then
    GUIDED_ARGS=(--guided-decoding-backend "$GUIDED_BACKEND")
    GUIDED_NOTE="via --guided-decoding-backend"
elif grep -q -- "--structured-outputs-config" <<<"$VLLM_HELP"; then
    # Newer releases group structured-output settings under one config object.
    GUIDED_ARGS=(--structured-outputs-config.backend "$GUIDED_BACKEND")
    GUIDED_NOTE="via --structured-outputs-config.backend"
else
    # No flag available. Modern vLLM defaults to xgrammar and still honours
    # per-request guided_json, so this is fine — run.sh smoke-tests it anyway.
    GUIDED_NOTE="engine default (no flag on this vLLM build)"
fi

echo ""
echo "=========================================="
echo "Serving: $MODEL"
echo "  Mode:            $MODE_LABEL"
echo "  Host:            $HOST:$PORT"
echo "  Max len:         $MAX_LEN"
echo "  GPU util:        $GPU_UTIL"
echo "  Quantization:    ${QUANT:-none or baked into checkpoint}"
echo "  Guided decoding: $GUIDED_NOTE"
echo "  Max concurrent:  $MAX_NUM_SEQS sequences"
echo "=========================================="
echo ""
echo "If startup aborts with 'exceeds available Mamba cache blocks (N)',"
echo "set VLLM_MAX_NUM_SEQS to N or lower and retry:"
echo "  VLLM_MAX_NUM_SEQS=32 $0 $CHOICE"
echo ""
echo "When loaded, run experiments in another terminal:"
if (( OFFLINE )); then
    echo "  ./scripts/run.sh --offline"
else
    echo "  ./scripts/run.sh"
fi
echo ""

exec vllm serve "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --max-model-len "$MAX_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --dtype auto \
    --trust-remote-code \
    ${GUIDED_ARGS[@]+"${GUIDED_ARGS[@]}"} \
    $QUANT