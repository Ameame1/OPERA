#!/usr/bin/env bash
set -euo pipefail

ROOT=${OPERA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
if [[ -z "${PYTHON:-}" && -x "$ROOT/.conda_env/bin/python" ]]; then
  PYTHON="$ROOT/.conda_env/bin/python"
else
  PYTHON=${PYTHON:-python}
fi

cd "$ROOT"
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

"$PYTHON" - <<'PY'
import json
import os

import faiss
import numpy as np
import torch
import transformers
from FlagEmbedding import BGEM3FlagModel

from opera.cli import (
    DEFAULT_BGE_LOCAL,
    DEFAULT_INDEX,
    DEFAULT_META,
    DEFAULT_QWEN25_3B_LOCAL,
    DEFAULT_QWEN25_7B_LOCAL,
)
from opera.retriever import BGEM3FaissRetriever

print(json.dumps({
    "python": os.sys.version.split()[0],
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "cuda_capability": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
    "faiss": getattr(faiss, "__version__", "unknown"),
    "faiss_num_gpus": faiss.get_num_gpus() if hasattr(faiss, "get_num_gpus") else 0,
    "transformers": transformers.__version__,
}, ensure_ascii=False, indent=2))

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
_ = torch.ones((2, 2), device="cuda").matmul(torch.ones((2, 2), device="cuda"))

for path in [DEFAULT_INDEX, DEFAULT_META]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

for model_ref in [DEFAULT_BGE_LOCAL, DEFAULT_QWEN25_7B_LOCAL, DEFAULT_QWEN25_3B_LOCAL]:
    looks_like_path = model_ref.startswith(("/", ".")) or os.sep in model_ref
    if looks_like_path and not os.path.exists(model_ref):
        raise FileNotFoundError(model_ref)

retriever = BGEM3FaissRetriever(
    index_path=DEFAULT_INDEX,
    metadata_path=DEFAULT_META,
    bge_model=DEFAULT_BGE_LOCAL,
    use_fp16=True,
    faiss_gpu=True,
    gpu_id=0,
)
docs = retriever.search("The Big Short author Michael Lewis finance books", top_k=3)
print(json.dumps({
    "index_size": retriever.info()["index_size"],
    "metadata_size": retriever.info()["metadata_size"],
    "smoke_query_hits": [doc.to_dict(max_chars=180) for doc in docs],
}, ensure_ascii=False, indent=2))

# Keep this import check explicit; it catches broken FlagEmbedding installs before runtime.
assert BGEM3FlagModel is not None
assert np.__version__.startswith("1.")
PY
