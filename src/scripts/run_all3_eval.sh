#!/usr/bin/env bash
set -euo pipefail

ROOT=${OPERA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
if [[ -z "${PYTHON:-}" && -x "$ROOT/.conda_env/bin/python" ]]; then
  PYTHON="$ROOT/.conda_env/bin/python"
else
  PYTHON=${PYTHON:-python}
fi
LIMIT=${1:-500}
OUT=${2:-$ROOT/eval_runs/all3_eval${LIMIT}_final}
EVAL_ROOT=${OPERA_EVAL_ROOT:-$ROOT/eval_sets/sample}
EVAL_SIZE=${OPERA_EVAL_SIZE:-500}
HOTPOT_FILE=${OPERA_HOTPOT_FILE:-$EVAL_ROOT/hotpotqa_${EVAL_SIZE}.jsonl}
TWOWIKI_FILE=${OPERA_2WIKI_FILE:-$EVAL_ROOT/2wiki_${EVAL_SIZE}.jsonl}
MUSIQUE_FILE=${OPERA_MUSIQUE_FILE:-$EVAL_ROOT/musique_${EVAL_SIZE}.jsonl}

cd "$ROOT"
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error

"$PYTHON" -m opera.evaluate \
  --dataset hotpotqa="$HOTPOT_FILE" \
  --dataset 2wiki="$TWOWIKI_FILE" \
  --dataset musique="$MUSIQUE_FILE" \
  --eval-limit "$LIMIT" \
  --llm-backend transformers \
  --retriever-backend http \
  --retriever-url "${OPERA_RETRIEVER_URL:-http://localhost:8110}" \
  --top-k 5 \
  --top-k-schedule 5,10,15 \
  --max-docs-in-prompt 15 \
  --max-doc-chars "${OPERA_MAX_DOC_CHARS:-1500}" \
  --max-steps 6 \
  --max-rewrites 2 \
  --multi-query-dependencies \
  --include-original-query \
  --final-synthesis \
  --continue-on-step-failure \
  --eval-output-dir "$OUT"
