from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from .llm import OpenAICompatibleLLM, TransformersLLM
from .pipeline import OperaPipeline, PipelineConfig
from .retriever import BGEM3FaissRetriever, HTTPRetrieverClient
from .utils import force_gpu0, read_jsonl, timestamp, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_INDEX = str(PROJECT_ROOT / "indexes" / "OPERA-index.index")
REPO_META = str(PROJECT_ROOT / "indexes" / "OPERA-index.meta")
DEFAULT_INDEX = os.environ.get("OPERA_INDEX_PATH", REPO_INDEX)
DEFAULT_META = os.environ.get("OPERA_METADATA_PATH", REPO_META)
DEFAULT_BGE_LOCAL = os.environ.get("OPERA_BGE_MODEL", "BAAI/bge-m3")
DEFAULT_QWEN25_7B_LOCAL = os.environ.get("OPERA_QWEN25_7B_MODEL", "Qwen/Qwen2.5-7B-Instruct")
DEFAULT_QWEN25_3B_LOCAL = os.environ.get("OPERA_QWEN25_3B_MODEL", "Qwen/Qwen2.5-3B-Instruct")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inference-only OPERA pipeline: plan -> BGE-M3 retrieval -> rewrite -> reasoning."
    )
    parser.add_argument("--question", type=str, default=None, help="Single question to run.")
    parser.add_argument("--input-jsonl", type=str, default=None, help="Batch input JSONL.")
    parser.add_argument("--question-field", type=str, default="question")
    parser.add_argument("--id-field", type=str, default="id")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "runs"))

    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--llm-backend", choices=["transformers", "vllm"], default="transformers")
    parser.add_argument("--planner-model", type=str, default=_default_qwen25_7b_model())
    parser.add_argument("--analysis-model", type=str, default=_default_qwen25_7b_model())
    parser.add_argument("--rewrite-model", type=str, default=_default_qwen25_3b_model())
    parser.add_argument("--torch-dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16"])

    parser.add_argument("--planner-url", type=str, default="http://localhost:8001")
    parser.add_argument("--analysis-url", type=str, default="http://localhost:8001")
    parser.add_argument("--rewrite-url", type=str, default="http://localhost:8003")
    parser.add_argument("--planner-vllm-model", type=str, default=None)
    parser.add_argument("--analysis-vllm-model", type=str, default=None)
    parser.add_argument("--rewrite-vllm-model", type=str, default=None)

    parser.add_argument("--index-path", type=str, default=DEFAULT_INDEX)
    parser.add_argument("--metadata-path", type=str, default=DEFAULT_META)
    parser.add_argument("--bge-model", type=str, default=None)
    parser.add_argument("--retriever-backend", choices=["local", "http"], default="local")
    parser.add_argument("--retriever-url", type=str, default="http://localhost:8110")
    parser.add_argument("--retriever-dataset", type=str, default=None)
    parser.add_argument("--no-faiss-gpu", action="store_true", help="Keep FAISS on CPU.")
    parser.add_argument("--bge-fp32", action="store_true", help="Disable fp16 for BGE-M3.")

    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--top-k-schedule",
        type=str,
        default=None,
        help="Comma-separated per-attempt top-k schedule, e.g. 5,10,15.",
    )
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--max-rewrites", type=int, default=2)
    parser.add_argument("--max-docs-in-prompt", type=int, default=5)
    parser.add_argument("--max-doc-chars", type=int, default=900)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--plan-max-tokens", type=int, default=512)
    parser.add_argument("--analysis-max-tokens", type=int, default=512)
    parser.add_argument("--rewrite-max-tokens", type=int, default=256)
    parser.add_argument("--final-synthesis", action="store_true")
    parser.add_argument("--continue-on-step-failure", action="store_true")
    parser.add_argument("--multi-query-dependencies", action="store_true")
    parser.add_argument("--max-dependency-queries", type=int, default=4)
    parser.add_argument(
        "--include-original-query",
        action="store_true",
        help="Also use the original question as a retrieval query for each sub-goal.",
    )
    parser.add_argument(
        "--direct-answer-fallback",
        action="store_true",
        help="When the decomposed chain returns Not found, try one whole-question retrieve-and-answer pass.",
    )
    return parser


def _default_bge_model() -> str:
    return DEFAULT_BGE_LOCAL


def _default_qwen25_7b_model() -> str:
    return DEFAULT_QWEN25_7B_LOCAL


def _default_qwen25_3b_model() -> str:
    return DEFAULT_QWEN25_3B_LOCAL


def _iter_inputs(args: argparse.Namespace) -> Iterable[Tuple[str, str]]:
    if args.question:
        yield "single", args.question
        return
    if not args.input_jsonl:
        raise SystemExit("Provide --question or --input-jsonl.")

    count = 0
    for row in read_jsonl(Path(args.input_jsonl)):
        question = row.get(args.question_field)
        if not question:
            continue
        qid = str(row.get(args.id_field) or count)
        yield qid, str(question)
        count += 1
        if args.limit is not None and count >= args.limit:
            break


def _build_llms(args: argparse.Namespace):
    if args.llm_backend == "vllm":
        planner = OpenAICompatibleLLM(args.planner_url, model=args.planner_vllm_model)
        if args.analysis_url == args.planner_url and args.analysis_vllm_model == args.planner_vllm_model:
            analysis = planner
        else:
            analysis = OpenAICompatibleLLM(args.analysis_url, model=args.analysis_vllm_model)
        rewrite = OpenAICompatibleLLM(args.rewrite_url, model=args.rewrite_vllm_model)
        return planner, analysis, rewrite

    planner = TransformersLLM(
        args.planner_model,
        device="cuda:0",
        torch_dtype=args.torch_dtype,
    )
    if args.analysis_model == args.planner_model:
        analysis = planner
    else:
        analysis = TransformersLLM(
            args.analysis_model,
            device="cuda:0",
            torch_dtype=args.torch_dtype,
        )
    rewrite = TransformersLLM(
        args.rewrite_model,
        device="cuda:0",
        torch_dtype=args.torch_dtype,
    )
    return planner, analysis, rewrite


def _build_pipeline(args: argparse.Namespace) -> OperaPipeline:
    force_gpu0(args.gpu_id)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    top_k_schedule = None
    if args.top_k_schedule:
        top_k_schedule = [int(item.strip()) for item in args.top_k_schedule.split(",") if item.strip()]

    if args.retriever_backend == "http":
        retriever = HTTPRetrieverClient(args.retriever_url, dataset=args.retriever_dataset)
    else:
        retriever = BGEM3FaissRetriever(
            index_path=args.index_path,
            metadata_path=args.metadata_path,
            bge_model=args.bge_model or _default_bge_model(),
            use_fp16=not args.bge_fp32,
            faiss_gpu=not args.no_faiss_gpu,
            gpu_id=0,
        )
    planner_llm, analysis_llm, rewrite_llm = _build_llms(args)
    config = PipelineConfig(
        top_k=args.top_k,
        top_k_schedule=top_k_schedule,
        max_steps=args.max_steps,
        max_rewrites=args.max_rewrites,
        max_docs_in_prompt=args.max_docs_in_prompt,
        max_doc_chars=args.max_doc_chars,
        plan_max_tokens=args.plan_max_tokens,
        analysis_max_tokens=args.analysis_max_tokens,
        rewrite_max_tokens=args.rewrite_max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        final_synthesis=args.final_synthesis,
        stop_on_step_failure=not args.continue_on_step_failure,
        multi_query_dependencies=args.multi_query_dependencies,
        max_dependency_queries=args.max_dependency_queries,
        include_original_query=args.include_original_query,
        direct_answer_fallback=args.direct_answer_fallback,
    )
    return OperaPipeline(
        planner_llm=planner_llm,
        analysis_llm=analysis_llm,
        rewrite_llm=rewrite_llm,
        retriever=retriever,
        config=config,
    )


def main() -> None:
    args = build_arg_parser().parse_args()
    run_dir = Path(args.output_dir) / timestamp()
    traces_dir = run_dir / "traces"
    results_path = run_dir / "results.jsonl"
    run_dir.mkdir(parents=True, exist_ok=True)
    traces_dir.mkdir(parents=True, exist_ok=True)

    write_json(run_dir / "args.json", vars(args))
    pipeline = _build_pipeline(args)

    results: List[Dict[str, Any]] = []
    with results_path.open("w", encoding="utf-8") as results_file:
        for qid, question in _iter_inputs(args):
            trace = pipeline.run(question, question_id=qid)
            safe_id = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in qid)
            trace_path = traces_dir / f"{safe_id}.json"
            write_json(trace_path, trace)

            row = {
                "id": qid,
                "question": question,
                "answer": trace["final_answer"],
                "success": trace["success"],
                "trace_path": str(trace_path),
                "stats": trace["stats"],
            }
            results_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            results_file.flush()
            results.append(row)
            print(json.dumps(row, ensure_ascii=False))

    summary = {
        "num_questions": len(results),
        "num_success": sum(1 for row in results if row["success"]),
        "results_path": str(results_path),
        "run_dir": str(run_dir),
    }
    write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
