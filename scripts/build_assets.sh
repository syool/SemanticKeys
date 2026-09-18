#!/usr/bin/env bash
# build_assets.sh — Generate every table and figure from the result files.
#
# The manuscript references five tables and three figures. Producing them by
# hand and pasting them in would mean the paper disagrees with the data the
# moment anything is re-run, so they are generated instead, straight from the
# JSONL files, into paper/tex/ where the LaTeX picks them up.
#
# Run this before to_latex.py. It needs the result directories, so it runs on
# the machine that has them.
#
# Usage:
#   ./scripts/build_assets.sh
#   RESULTS=/path/to/results ./scripts/build_assets.sh

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS="${RESULTS:-$ROOT/results}"
OUT="$ROOT/paper/tex"
TABLES="$OUT/tables"
FIGURES="$OUT/figures"

SWEEP_QWEN="$RESULTS/sweep/qwen3.5-27b_"
LADDER="$RESULTS/ladder"
ABLATION="$RESULTS/question_ablation"

mkdir -p "$TABLES" "$FIGURES"
cd "$ROOT/sk"

MISSING=0
require() {
    if [[ ! -d "$1" ]]; then
        echo "  missing: $1" >&2
        MISSING=$((MISSING + 1))
        return 1
    fi
}

echo "Tables"
if require "$LADDER"; then
    python3 tables.py --table cost --ladder "$LADDER" --format latex \
        --output "$TABLES/cost.tex" >/dev/null && echo "  cost.tex"
fi
if require "$SWEEP_QWEN"; then
    python3 tables.py --table accuracy --sweep "$SWEEP_QWEN" --window 16k \
        --format latex --output "$TABLES/accuracy.tex" >/dev/null \
        && echo "  accuracy.tex"
    python3 tables.py --table fields --sweep "$SWEEP_QWEN" --turns 96 \
        --format latex --output "$TABLES/fields.tex" >/dev/null \
        && echo "  fields.tex"
fi
if require "$ABLATION"; then
    python3 tables.py --table removal --ablation "$ABLATION" --format latex \
        --output "$TABLES/removal.tex" >/dev/null && echo "  removal.tex"
fi

# Table 1 describes the conditions rather than reporting data, so it is
# written by hand rather than generated.
if [[ ! -f "$TABLES/conditions.tex" ]]; then
    cat > "$TABLES/conditions.tex" <<'TEX'
\begin{table}[t]
\centering
\caption{Conditions and how each retains the objective across turns.}
\label{tab:conditions}
\begin{tabular}{lll}
\toprule
Condition & Output & State handling \\
\midrule
NL-append  & prose & transcript accumulates \\
NL-anchor  & prose & transcript accumulates, goal restated each turn \\
SK         & JSON, open schema  & bounded state tuple, updated \\
SK-cite    & JSON, three fields & bounded state tuple, quoted evidence \\
SK-index   & JSON, three fields & bounded state tuple, evidence by reference \\
\bottomrule
\end{tabular}
\end{table}
TEX
    echo "  conditions.tex (static)"
fi

echo ""
echo "Figures"
if require "$SWEEP_QWEN"; then
    python3 figures.py --figure cost --sweep "$SWEEP_QWEN" --window 16k \
        --out "$FIGURES/cost.pdf" >/dev/null 2>&1 && echo "  cost.pdf"
fi
if require "$LADDER"; then
    python3 figures.py --figure verbosity --ladder "$LADDER" \
        --out "$FIGURES/verbosity.pdf" >/dev/null 2>&1 && echo "  verbosity.pdf"
    python3 figures.py --figure scale --ladder "$LADDER" \
        --confirm-size 5 --confirm-delta 0.095 \
        --out "$FIGURES/scale.pdf" >/dev/null 2>&1 && echo "  scale.pdf"
fi

echo ""
if (( MISSING )); then
    echo "$MISSING result directory/directories not found."
    echo "Set RESULTS to where they live, or run the experiments first."
    echo "to_latex.py will emit placeholders for anything absent."
    exit 1
fi
echo "Assets written to $OUT"