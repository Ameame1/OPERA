#!/usr/bin/env bash
set -euo pipefail

ROOT=${OPERA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}

cd "$ROOT"
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

if [[ -z "${PYTHON:-}" && -x "$ROOT/.conda_env/bin/python" ]]; then
  PYTHON="$ROOT/.conda_env/bin/python"
else
  PYTHON=${PYTHON:-python}
fi

"$PYTHON" -m opera \
  --question "${1:-What books about finance has the author of The Big Short written?}" \
  --llm-backend "${OPERA_LLM_BACKEND:-transformers}" \
  --retriever-backend "${OPERA_RETRIEVER_BACKEND:-http}" \
  --retriever-url "${OPERA_RETRIEVER_URL:-http://localhost:8110}" \
  --top-k 5 \
  --top-k-schedule 5,10,15 \
  --max-docs-in-prompt 15 \
  --max-doc-chars 1500 \
  --max-steps 6 \
  --max-rewrites 2 \
  --multi-query-dependencies \
  --include-original-query \
  --final-synthesis \
  --continue-on-step-failure \
  "${@:2}"
