# OPERA Source Guide

This directory contains the inference-only OPERA implementation. It includes the runtime package, helper scripts, dependency files, and the expected local index directory. Training code is not included here.

## Directory Layout

- `opera/`: Python package for OPERA inference, retrieval, evaluation, and serving.
- `scripts/`: shell and Python helpers for environment checks, running inference, evaluation, index construction, and diagnostics.
- `requirements.txt`: pip dependency list. This installs CPU FAISS by default; use `environment.yml` for GPU FAISS.
- `environment.yml`: conda environment spec for the full GPU retrieval pipeline.
- `eval_sets/`: local evaluation splits generated with `scripts/build_eval_sets.py`. This directory is optional and can be recreated from official dev files.
- `indexes/`: optional local placement for the released FAISS index files. This directory is git-ignored.

Expected index filenames:

```text
indexes/OPERA-index.index
indexes/OPERA-index.meta
```

The index path can also be overridden:

```bash
export OPERA_INDEX_PATH=/path/to/OPERA-index.index
export OPERA_METADATA_PATH=/path/to/OPERA-index.meta
```

## Package: `opera/`

### `__init__.py`

Exports the main package-level classes:

- `OperaPipeline`
- `PipelineConfig`
- `BGEM3FaissRetriever`

### `__main__.py`

Entry point for single-question or JSONL batch inference:

```bash
python -m opera --question "What books about finance has the author of The Big Short written?"
```

### `cli.py`

Command-line interface and runtime wiring. It:

- parses CLI arguments;
- selects local `transformers` or OpenAI-compatible vLLM backend;
- builds Plan, Analysis-Answer, and Rewrite LLM clients;
- builds the local or HTTP BGE-M3 retriever;
- creates `PipelineConfig`;
- writes run outputs under `runs/<timestamp>/`.

Important environment variables:

```bash
export OPERA_BGE_MODEL=/path/to/bge-m3
export OPERA_QWEN25_7B_MODEL=/path/to/Qwen2.5-7B-Instruct
export OPERA_QWEN25_3B_MODEL=/path/to/Qwen2.5-3B-Instruct
export OPERA_INDEX_PATH=/path/to/OPERA-index.index
export OPERA_METADATA_PATH=/path/to/OPERA-index.meta
```

Default inference settings used by the helper scripts:

```text
top-k schedule: 5,10,15
max docs in prompt: 15
max doc chars: 1500
max steps: 6
max rewrites: 2
final synthesis: enabled
continue on step failure: enabled
```

### `pipeline.py`

Core OPERA inference loop:

1. Plan Agent decomposes the original question into dependent sub-goals.
2. Each sub-goal is filled with previous answers when placeholders are present.
3. BGE-M3 retrieves evidence for the current sub-goal.
4. Analysis-Answer Agent decides whether the retrieved documents are sufficient.
5. Rewrite Agent reformulates the query when evidence is insufficient.
6. A final synthesis step combines executed sub-goal answers into the final answer.

The implementation is inference-only. It has no reflection module and no training logic.

### `prompts.py`

Prompt templates for:

- Plan Agent;
- Analysis-Answer Agent;
- Rewrite Agent;
- final synthesis prompt used as a non-agent aggregation step.

The prompts follow the OPERA role design and include grounding constraints used in the final pipeline. They do not inject gold answers or supporting facts.

### `llm.py`

LLM client abstractions:

- `TransformersLLM`: local Hugging Face chat generation.
- `OpenAICompatibleLLM`: vLLM/OpenAI-compatible HTTP chat client.

The default local models are:

```text
Plan / Analysis-Answer: Qwen/Qwen2.5-7B-Instruct
Rewrite: Qwen/Qwen2.5-3B-Instruct
```

### `retriever.py`

Retrieval backends:

- `BGEM3FaissRetriever`: local BGE-M3 + FAISS index retriever.
- `BGEM3MultiFaissRetriever`: optional multi-index retriever.
- `HTTPRetrieverClient`: client for the persistent retriever server.

The local retriever expects a FAISS index file and matching metadata pickle.

### `retriever_server.py`

Persistent HTTP retriever service. It keeps BGE-M3 and the FAISS index loaded, usually on GPU0.

Endpoints:

```text
GET  /health
GET  /search?query=...&top_k=5
POST /search
POST /batch_search
```

CLI example:

```bash
CUDA_VISIBLE_DEVICES=0 python -m opera.retriever_server \
  --index-path indexes/OPERA-index.index \
  --metadata-path indexes/OPERA-index.meta \
  --port 8110
```

### `evaluate.py`

Evaluation runner and scorer. It:

- runs OPERA over one or more JSONL datasets;
- supports multiple gold answers / aliases;
- computes EM and F1;
- writes `results.jsonl`, traces, and summaries;
- enforces the invariant `F1 >= EM`.

Entry point:

```bash
python -m opera.evaluate --dataset hotpotqa=/path/to/hotpotqa.jsonl --eval-limit 500
```

### `schema.py`

Dataclasses for the pipeline:

- retrieved documents;
- plan steps;
- analysis results;
- rewrite results.

### `utils.py`

Shared utilities for:

- JSON / JSONL IO;
- timestamped run directories;
- response JSON extraction;
- GPU selection.

## Scripts: `scripts/`

Run all commands from `src/` unless noted otherwise.

### `validate_env.sh`

Checks whether the current environment can run the pipeline. It verifies:

- Python / PyTorch / CUDA;
- FAISS GPU availability;
- Transformers version;
- BGE-M3 import;
- model/index paths;
- a small retrieval smoke test.

Usage:

```bash
cd src
CUDA_VISIBLE_DEVICES=0 ./scripts/validate_env.sh
```

Optional overrides:

```bash
PYTHON=/path/to/python ./scripts/validate_env.sh
OPERA_INDEX_PATH=/path/to/index OPERA_METADATA_PATH=/path/to/meta ./scripts/validate_env.sh
```

### `run_retriever_server.sh`

Starts the persistent BGE-M3 + FAISS retriever server on GPU0.

Usage:

```bash
cd src
CUDA_VISIBLE_DEVICES=0 ./scripts/run_retriever_server.sh
```

Useful overrides:

```bash
export OPERA_RETRIEVER_PORT=8110
export OPERA_RETRIEVER_HOST=0.0.0.0
export OPERA_INDEX_PATH=/path/to/OPERA-index.index
export OPERA_METADATA_PATH=/path/to/OPERA-index.meta
```

Direct health check:

```bash
curl http://localhost:8110/health
```

Direct retrieval check:

```bash
curl "http://localhost:8110/search?query=The%20Big%20Short%20Michael%20Lewis&top_k=5"
```

### `run_single.sh`

Runs one question through the final OPERA pipeline. It assumes the retriever server is already running unless `OPERA_RETRIEVER_BACKEND=local` is set.

Usage:

```bash
cd src
CUDA_VISIBLE_DEVICES=0 ./scripts/run_single.sh \
  "What books about finance has the author of The Big Short written?"
```

Optional extra CLI arguments can be appended:

```bash
./scripts/run_single.sh "Question text" --output-dir runs/manual_test
```

### `run_all3_eval.sh`

Runs the standard three-dataset evaluation over HotpotQA, 2WikiMultiHopQA, and MuSiQue JSONL files.

By default, the helper reads local evaluation files from `eval_sets/sample/`.

Expected filenames:

```text
hotpotqa_500.jsonl
2wiki_500.jsonl
musique_500.jsonl
```

Usage:

```bash
cd src
CUDA_VISIBLE_DEVICES=0 ./scripts/run_all3_eval.sh 500
```

With custom eval set root:

```bash
OPERA_EVAL_ROOT=/path/to/eval_sets/sample ./scripts/run_all3_eval.sh 500
```

Build or refresh the local evaluation files:

```bash
python scripts/build_eval_sets.py \
  --seed <int> \
  --output-dir eval_sets/sample \
  --hotpot-dev /path/to/hotpot_dev_distractor_v1.json \
  --2wiki-dev /path/to/2wiki/dev.json \
  --musique-dev /path/to/musique_ans_v1.0_dev.jsonl
```

With custom output directory:

```bash
./scripts/run_all3_eval.sh 500 eval_runs/final_eval
```

### `analyze_eval_run.py`

Recomputes diagnostics from an existing evaluation run. It reads dataset subdirectories containing `results.jsonl` and optional trace files.

Usage:

```bash
cd src
python scripts/analyze_eval_run.py eval_runs/final_eval \
  --output eval_runs/final_eval/diagnostic_recomputed.json
```

The report includes:

- recomputed EM/F1;
- success and not-found rates;
- partial-match rate;
- wrong examples where gold text appears in retrieved documents;
- MuSiQue hop-level breakdown when available.

### `build_eval_sets.py`

Builds reproducible evaluation splits from official HotpotQA, 2WikiMultiHopQA, and MuSiQue dev files.

Usage:

```bash
cd src
python scripts/build_eval_sets.py \
  --seed <int> \
  --output-dir eval_sets/sample \
  --hotpot-dev /path/to/hotpot_dev_distractor_v1.json \
  --2wiki-dev /path/to/2wiki/dev.json \
  --musique-dev /path/to/musique_ans_v1.0_dev.jsonl
```

By default this writes:

```text
eval_sets/sample/hotpotqa_500.jsonl
eval_sets/sample/2wiki_500.jsonl
eval_sets/sample/musique_500.jsonl
```

### `build_opera_index.py`

Builds the final OPERA 1.78M-style FAISS index variant. It replaces old MuSiQue sentence chunks with a MuSiQue paragraph plus sentence-window representation:

- paragraph chunks;
- sentence-window chunks.

It keeps the total corpus size fixed by pruning low-information, unprotected 2Wiki train sentences. HotpotQA and 2Wiki dev support documents can be protected during balancing.

Dry-run plan:

```bash
cd src
python scripts/build_opera_index.py \
  --mixed-index /path/to/base.index \
  --mixed-meta /path/to/base.meta \
  --musique-meta /path/to/musique.meta \
  --musique-arrow-dir /path/to/musique_arrow_dir \
  --wiki-train-json /path/to/2wiki_train.json \
  --hotpot-dev-json /path/to/hotpot_dev.json \
  --2wiki-dev-json /path/to/2wiki_dev.json \
  --dry-run
```

Build index:

```bash
python scripts/build_opera_index.py \
  --mixed-index /path/to/base.index \
  --mixed-meta /path/to/base.meta \
  --musique-meta /path/to/musique.meta \
  --musique-arrow-dir /path/to/musique_arrow_dir \
  --wiki-train-json /path/to/2wiki_train.json \
  --hotpot-dev-json /path/to/hotpot_dev.json \
  --2wiki-dev-json /path/to/2wiki_dev.json \
  --output-dir indexes \
  --output-name OPERA-index
```

Important options:

```text
--mixed-index /path/to/base.index
--mixed-meta /path/to/base.meta
--musique-meta /path/to/musique.meta
--musique-arrow-dir /path/to/musique_arrow_dir
--target-total 1780294
--protect-eval-set /path/to/eval.jsonl
--window-radius 1
--min-window-chars 35
```

### `index_build_utils.py`

Supporting index-construction module used by the OPERA index builder. It contains shared utilities for:

- metadata loading;
- support-key extraction;
- MuSiQue paragraph selection;
- old-vector retention;
- BGE-M3 encoding;
- FAISS writing.

This module also contains a legacy paragraph-only index builder, but the recommended final builder is `build_opera_index.py`.

### `verify_gold_coverage.py`

Verifies that support/gold documents from evaluation JSONL files remain covered by the planned rebuilt corpus before writing the new index.

Usage:

```bash
cd src
python scripts/verify_gold_coverage.py \
  --mixed-index /path/to/base.index \
  --mixed-meta /path/to/base.meta \
  --musique-meta /path/to/musique.meta \
  --musique-arrow-dir /path/to/musique_arrow_dir \
  --wiki-train-json /path/to/2wiki_train.json \
  --hotpot-dev-json /path/to/hotpot_dev.json \
  --2wiki-dev-json /path/to/2wiki_dev.json \
  --musique-dev-jsonl /path/to/musique_dev.jsonl \
  --eval-set /path/to/hotpotqa_500.jsonl \
  --eval-set /path/to/2wiki_500.jsonl \
  --eval-set /path/to/musique_500.jsonl
```

The script exits with non-zero status if any required support document is missing.

## Common End-to-End Commands

Create environment:

```bash
conda env create -f environment.yml
conda activate opera
```

Start retriever:

```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/run_retriever_server.sh
```

Run one question:

```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/run_single.sh "Question text"
```

Run evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 OPERA_EVAL_ROOT=/path/to/eval_sets/sample \
  ./scripts/run_all3_eval.sh 500 eval_runs/final_eval
```

## Generated Files

These outputs are intentionally git-ignored:

- `runs/`
- `eval_runs/`
- `tuning_runs/`
- `logs/`
- `.conda_env/`
- `indexes/`
- FAISS / metadata files such as `*.index`, `*.faiss`, `*.meta`, `*.pkl`
