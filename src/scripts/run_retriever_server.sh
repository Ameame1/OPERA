#!/usr/bin/env bash
set -euo pipefail

ROOT=${OPERA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
if [[ -z "${PYTHON:-}" && -x "$ROOT/.conda_env/bin/python" ]]; then
  PYTHON="$ROOT/.conda_env/bin/python"
else
  PYTHON=${PYTHON:-python}
fi
PORT=${OPERA_RETRIEVER_PORT:-8110}
REPO_INDEX=$ROOT/indexes/OPERA-index.index
REPO_META=$ROOT/indexes/OPERA-index.meta
INDEX_PATH="${OPERA_INDEX_PATH:-$REPO_INDEX}"
METADATA_PATH="${OPERA_METADATA_PATH:-$REPO_META}"

cd "$ROOT"
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

exec "$PYTHON" -m opera.retriever_server \
  --host "${OPERA_RETRIEVER_HOST:-0.0.0.0}" \
  --port "$PORT" \
  --gpu-id 0 \
  --index-path "$INDEX_PATH" \
  --metadata-path "$METADATA_PATH" \
  "$@"
