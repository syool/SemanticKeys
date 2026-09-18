#!/usr/bin/env bash
# prepare_offline_bundle.sh
#
# Run this on a machine WITH internet access. It downloads everything the
# experiments need, then packs it into a directory you can transfer to the
# offline environment (USB, internal file share, scp through a jump host).
#
# What gets bundled:
#   1. Main model weights   (Qwen 3.5 27B, Gemma 4 31B)  — ~100 GB
#   2. Embedder weights     (all-mpnet-base-v2)          — ~420 MB
#   3. HotpotQA dev set     (distractor split)           — ~90 MB
#   4. Python wheels        (vllm, scipy, statsmodels, …) — ~2 GB
#   5. The experiment code itself
#
# Total: roughly 105 GB. Plan storage and transfer time accordingly.
#
# Usage:
#   ./prepare_offline_bundle.sh /path/to/bundle_output
#
# Then on the offline machine:
#   ./scripts/bundle_install.sh /path/to/bundle_output

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=models.conf
source "$ROOT/models.conf"

BUNDLE_DIR="${1:-./sk_offline_bundle}"
mkdir -p "$BUNDLE_DIR"/{models,data,wheels,code}

echo "=========================================="
echo "Preparing offline bundle at: $BUNDLE_DIR"
echo "This will download ~105 GB. Ensure you have space and time."
echo "=========================================="
read -rp "Proceed? [y/N] " confirm
[[ "${confirm,,}" == "y" ]] || { echo "Aborted."; exit 0; }

# -----------------------------------------------------------------------------
# 1. Main model weights
# -----------------------------------------------------------------------------
echo ""
echo "[1/5] Downloading main model weights..."

export HF_HUB_ENABLE_HF_TRANSFER=1

# Verify every configured repo before spending hours on downloads.
echo "  Verifying repository IDs..."
if ! python3 "$ROOT/scripts/verify_models.py" \
        --conf "$ROOT/models.conf" \
        --target-dir "$BUNDLE_DIR"; then
    echo "" >&2
    echo "Repository verification failed. Fix models.conf and re-run." >&2
    exit 1
fi

download_repo() {
    local repo="$1" local_dir="$2"
    local target="$BUNDLE_DIR/models/$local_dir"
    if [[ -d "$target" ]] && [[ -n "$(ls -A "$target" 2>/dev/null)" ]]; then
        echo "  $repo already present, verifying completeness..."
    else
        echo "  Downloading $repo ..."
    fi
    # huggingface-cli resumes partial downloads, so re-running after an
    # interruption continues rather than restarting.
    huggingface-cli download "$repo" \
        --local-dir "$target" \
        --local-dir-use-symlinks False \
        --resume-download
}

download_repo "$QWEN_REPO"       "$QWEN_LOCAL_DIR"
download_repo "$GEMMA_REPO"      "$GEMMA_LOCAL_DIR"

# The MoE variant is optional; include it if you want a fast iteration model.
if [[ "${INCLUDE_MOE:-0}" == "1" ]]; then
    download_repo "$GEMMA_MOE_REPO" "$GEMMA_MOE_LOCAL_DIR"
fi

# Small model for offline dry runs — cheap insurance, ~8 GB.
if [[ "${INCLUDE_DRYRUN_MODEL:-1}" == "1" ]]; then
    download_repo "$DRYRUN_REPO" "$DRYRUN_LOCAL_DIR"
fi

# -----------------------------------------------------------------------------
# 2. Embedder
# -----------------------------------------------------------------------------
echo ""
echo "[2/5] Downloading embedder (all-mpnet-base-v2)..."
huggingface-cli download "$EMBEDDER_REPO" \
    --local-dir "$BUNDLE_DIR/models/$EMBEDDER_LOCAL_DIR" \
    --local-dir-use-symlinks False --resume-download

# -----------------------------------------------------------------------------
# 3. HotpotQA dataset
# -----------------------------------------------------------------------------
echo ""
echo "[3/5] Downloading HotpotQA dev-distractor split..."

# Primary source: the official CMU mirror.
if curl -fL --max-time 600 \
    "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json" \
    -o "$BUNDLE_DIR/data/hotpot_dev_distractor_v1.json"; then
    echo "  Downloaded from CMU mirror."
else
    echo "  CMU mirror failed; falling back to HuggingFace datasets export..."
    python3 - <<'PYEOF'
import json, os, sys
from datasets import load_dataset

out_path = os.environ["BUNDLE_DIR"] + "/data/hotpot_dev_distractor_v1.json"
ds = load_dataset("hotpot_qa", "distractor", split="validation")

records = []
for rec in ds:
    records.append({
        "_id": rec.get("id", ""),
        "question": rec["question"],
        "answer": rec["answer"],
        "type": rec.get("type", "bridge"),
        "level": rec.get("level", "hard"),
        "context": {
            "title": list(rec["context"]["title"]),
            "sentences": [list(s) for s in rec["context"]["sentences"]],
        },
        "supporting_facts": {
            "title": list(rec["supporting_facts"]["title"]),
            "sent_id": list(rec["supporting_facts"]["sent_id"]),
        },
    })

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False)
print(f"  Wrote {len(records)} records to {out_path}")
PYEOF
fi

# -----------------------------------------------------------------------------
# 4. Python wheels
# -----------------------------------------------------------------------------
echo ""
echo "[4/5] Downloading Python wheels for offline install..."

# Single source of truth: the project's requirements.txt. Copying rather
# than duplicating means the online and offline dependency sets cannot drift.
cp "$ROOT/requirements.txt" "$BUNDLE_DIR/requirements_offline.txt"

# Wheels must match the OFFLINE machine's platform and Python version, not
# this machine's. Override these if the two differ.
TARGET_PY="${TARGET_PYTHON_VERSION:-$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')}"
TARGET_PLATFORM="${TARGET_PLATFORM:-manylinux2014_x86_64}"

echo "  Target Python:   $TARGET_PY"
echo "  Target platform: $TARGET_PLATFORM"
echo "  (override with TARGET_PYTHON_VERSION / TARGET_PLATFORM if the offline"
echo "   machine differs from this one)"

# --only-binary=:all: forces wheels rather than source distributions, which
# would need a compiler on the air-gapped machine.
if ! pip download -r "$BUNDLE_DIR/requirements_offline.txt" \
        -d "$BUNDLE_DIR/wheels" \
        --only-binary=:all: \
        --python-version "$TARGET_PY" \
        --platform "$TARGET_PLATFORM" \
        2>&1 | tail -8; then
    echo "  WARNING: wheel download had failures." >&2
    echo "  Common cause: a package has no wheel for $TARGET_PLATFORM/$TARGET_PY." >&2
    echo "  Check which one, then either relax the constraint or install it" >&2
    echo "  separately on the offline machine." >&2
fi

WHEEL_COUNT=$(find "$BUNDLE_DIR/wheels" -name "*.whl" | wc -l)
echo "  Downloaded $WHEEL_COUNT wheels"

# -----------------------------------------------------------------------------
# 5. Experiment code
# -----------------------------------------------------------------------------
echo ""
echo "[5/5] Copying experiment code..."
cp -r "$ROOT/sk" "$BUNDLE_DIR/code/"
cp -r "$ROOT/docs" "$BUNDLE_DIR/code/" 2>/dev/null || true
cp "$ROOT"/scripts/*.sh "$BUNDLE_DIR/code/" 2>/dev/null || true
cp "$ROOT"/scripts/*.py "$BUNDLE_DIR/code/" 2>/dev/null || true
cp "$ROOT"/*.md "$BUNDLE_DIR/code/" 2>/dev/null || true
cp "$ROOT/models.conf" "$BUNDLE_DIR/code/"

# -----------------------------------------------------------------------------
# Manifest
# -----------------------------------------------------------------------------
echo ""
echo "Writing manifest..."
{
    echo "Semantic Keys — Offline Bundle Manifest"
    echo "Generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo ""
    echo "Contents:"
    du -sh "$BUNDLE_DIR"/* 2>/dev/null
    echo ""
    echo "Model checksums (for reproducibility documentation in the paper):"
    for m in "$BUNDLE_DIR"/models/*/; do
        name=$(basename "$m")
        # config.json hash is a stable, small proxy for the model identity
        if [[ -f "$m/config.json" ]]; then
            h=$(sha256sum "$m/config.json" | cut -d' ' -f1)
            echo "  $name  config.json sha256=$h"
        fi
    done
} > "$BUNDLE_DIR/MANIFEST.txt"

cat "$BUNDLE_DIR/MANIFEST.txt"

echo ""
echo "=========================================="
echo "Bundle ready at: $BUNDLE_DIR"
echo ""
echo "Transfer this whole directory to the offline machine, then run:"
echo "  ./scripts/bundle_install.sh $BUNDLE_DIR"
echo "=========================================="
