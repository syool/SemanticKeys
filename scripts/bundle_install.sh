#!/usr/bin/env bash
# install_offline_bundle.sh
#
# Run this on the OFFLINE machine after transferring the bundle produced by
# prepare_offline_bundle.sh.
#
# Installs Python packages from the bundled wheels (no PyPI access needed),
# places model weights and data where the experiment code expects them, and
# verifies everything is ready.
#
# Usage:
#   ./install_offline_bundle.sh /path/to/sk_offline_bundle [install_root]
#
# install_root defaults to ~/sk_experiments

set -euo pipefail

BUNDLE_DIR="${1:?Usage: $0 <bundle_dir> [install_root]}"
INSTALL_ROOT="${2:-$HOME/sk_experiments}"

if [[ ! -d "$BUNDLE_DIR" ]]; then
    echo "ERROR: bundle directory not found: $BUNDLE_DIR" >&2
    exit 1
fi

echo "=========================================="
echo "Installing offline bundle"
echo "  Source: $BUNDLE_DIR"
echo "  Target: $INSTALL_ROOT"
echo "=========================================="

mkdir -p "$INSTALL_ROOT"/{models,data,results}

# -----------------------------------------------------------------------------
# 1. Python packages from bundled wheels
# -----------------------------------------------------------------------------
echo ""
echo "[1/5] Installing Python packages from bundled wheels..."

if [[ ! -d "$BUNDLE_DIR/wheels" ]]; then
    echo "ERROR: $BUNDLE_DIR/wheels not found." >&2
    exit 1
fi

# --no-index prevents any attempt to reach PyPI, which would fail (and hang)
# on an air-gapped machine.
pip install \
    --no-index \
    --find-links "$BUNDLE_DIR/wheels" \
    -r "$BUNDLE_DIR/requirements_offline.txt" \
    2>&1 | tail -10

echo "  Verifying imports..."
python3 - <<'PYEOF'
import sys
missing = []
for mod in ("numpy", "scipy", "statsmodels", "sentence_transformers", "torch"):
    try:
        __import__(mod)
    except ImportError as e:
        missing.append(f"{mod}: {e}")
if missing:
    print("  MISSING:")
    for m in missing:
        print(f"    {m}")
    sys.exit(1)
print("  All core imports OK.")

try:
    import vllm
    print(f"  vllm {vllm.__version__} OK.")
except ImportError:
    print("  WARNING: vllm not importable. Serving will not work.")
PYEOF

# -----------------------------------------------------------------------------
# 2. Model weights
# -----------------------------------------------------------------------------
echo ""
echo "[2/5] Placing model weights..."
if [[ -d "$BUNDLE_DIR/models" ]]; then
    for m in "$BUNDLE_DIR"/models/*/; do
        name=$(basename "$m")
        target="$INSTALL_ROOT/models/$name"
        if [[ -d "$target" ]]; then
            echo "  $name already present, skipping copy."
        else
            echo "  Copying $name (this may take a while)..."
            cp -r "$m" "$target"
        fi
        # Report size so the user can spot truncated transfers.
        echo "    $(du -sh "$target" | cut -f1)  $target"
    done
else
    echo "  WARNING: no models directory in bundle."
fi

# -----------------------------------------------------------------------------
# 3. Dataset
# -----------------------------------------------------------------------------
echo ""
echo "[3/5] Placing dataset..."
if [[ -d "$BUNDLE_DIR/data" ]]; then
    cp -n "$BUNDLE_DIR"/data/*.json "$INSTALL_ROOT/data/" 2>/dev/null || true
    for f in "$INSTALL_ROOT"/data/*.json; do
        [[ -f "$f" ]] || continue
        n=$(python3 -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "$f" 2>/dev/null || echo "?")
        echo "  $(basename "$f"): $n records"
    done
else
    echo "  WARNING: no data directory in bundle."
fi

# -----------------------------------------------------------------------------
# 4. Experiment code
# -----------------------------------------------------------------------------
echo ""
echo "[4/5] Placing experiment code..."
if [[ -d "$BUNDLE_DIR/code" ]]; then
    cp -r "$BUNDLE_DIR"/code/* "$INSTALL_ROOT/"
    chmod +x "$INSTALL_ROOT"/*.sh 2>/dev/null || true
    echo "  Code installed at $INSTALL_ROOT"
    ls "$INSTALL_ROOT/src" 2>/dev/null | sed 's/^/    /'
else
    echo "  WARNING: no code directory in bundle."
fi

# -----------------------------------------------------------------------------
# 5. Offline environment configuration
# -----------------------------------------------------------------------------
echo ""
echo "[5/5] Writing offline environment config..."

cat > "$INSTALL_ROOT/env_offline.sh" <<EOF
# Source this before running experiments:  source env_offline.sh
#
# These variables prevent any library from attempting a network call, which
# on an air-gapped machine would hang until timeout rather than failing fast.

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

export SK_ROOT="$INSTALL_ROOT"
export SK_MODELS="\$SK_ROOT/models"
export SK_DATA="\$SK_ROOT/data"
export SK_RESULTS="\$SK_ROOT/results"

# Embedder path (used by --embedder-path)
export SK_EMBED_MODEL="\$SK_MODELS/all-mpnet-base-v2"

echo "Offline environment configured:"
echo "  SK_ROOT    = \$SK_ROOT"
echo "  SK_MODELS  = \$SK_MODELS"
echo "  SK_DATA    = \$SK_DATA"
echo "  HF_HUB_OFFLINE=1 (no network calls will be attempted)"
EOF

chmod +x "$INSTALL_ROOT/env_offline.sh"

# -----------------------------------------------------------------------------
# Done
# -----------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "Installation complete."
echo ""
echo "Next steps:"
echo "  cd $INSTALL_ROOT"
echo "  source env_offline.sh"
echo ""
echo "  # Terminal A — start the model server"
echo "  ./scripts/serve.sh qwen --offline"
echo ""
echo "  # Terminal B — run the experiments"
echo "  ./scripts/run.sh --offline"
echo "=========================================="
