from __future__ import annotations

import argparse
import json
import re
import string
import time
import unicodedata
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .cli import build_arg_parser, _build_pipeline
from .utils import timestamp, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_ROOT = Path(os.environ.get("OPERA_EVAL_ROOT", PROJECT_ROOT / "eval_sets" / "sample"))
DEFAULT_DATASETS = {
    "hotpotqa": str(DEFAULT_EVAL_ROOT / "hotpotqa_500.jsonl"),
    "2wiki": str(DEFAULT_EVAL_ROOT / "2wiki_500.jsonl"),
    "musique": str(DEFAULT_EVAL_ROOT / "musique_500.jsonl"),
}


def normalize_answer(text: str) -> str:
    def strip_accents(s: str) -> str:
        normalized = unicodedata.normalize("NFKD", s)
        return "".join(ch for ch in normalized if not unicodedata.combining(ch))

    def remove_articles(s: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def white_space_fix(s: str) -> str:
        return " ".join(s.split())

    def remove_punc(s: str) -> str:
        return "".join(ch for ch in s if ch not in set(string.punctuation))

    normalized = strip_accents(str(text).lower())
    normalized = remove_punc(normalized)
    normalized = re.sub(r"\bunited states(?: of america)?\b", "america", normalized)
    normalized = re.sub(r"\bu s\b", "america", normalized)
    return white_space_fix(remove_articles(normalized))


DEMONYM_CANONICAL = {
    "america": "america",
    "american": "america",
    "usa": "america",
    "us": "america",
    "brazil": "brazil",
    "brazilian": "brazil",
    "brasileiro": "brazil",
    "canada": "canada",
    "canadian": "canada",
    "england": "england",
    "english": "england",
    "france": "france",
    "french": "france",
    "netherlands": "netherlands",
    "netherland": "netherlands",
    "holland": "netherlands",
    "dutch": "netherlands",
    "germany": "germany",
    "german": "germany",
    "india": "india",
    "indian": "india",
    "italy": "italy",
    "italian": "italy",
    "japan": "japan",
    "japanese": "japan",
    "mexico": "mexico",
    "mexican": "mexico",
    "pakistan": "pakistan",
    "pakistani": "pakistan",
    "spain": "spain",
    "spanish": "spain",
}


def _singularize_token(token: str) -> str:
    if len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("es") and len(token) > 4 and not token.endswith(("ses", "xes", "zes")):
        return token[:-1]
    if token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _canonical_token(token: str) -> str:
    token = _singularize_token(token)
    return DEMONYM_CANONICAL.get(token, token)


def _canonical_tokens(text: str) -> List[str]:
    return [_canonical_token(token) for token in normalize_answer(text).split()]


def _acronym(tokens: List[str]) -> str:
    stop = {"of", "and", "the", "for", "in", "on", "at", "to"}
    letters = [token[0] for token in tokens if token and token not in stop]
    return "".join(letters)


def exact_match_score(prediction: str, ground_truth: str) -> float:
    pred = normalize_answer(prediction)
    gold = normalize_answer(ground_truth)
    if pred == gold:
        return 1.0
    if _canonical_tokens(prediction) == _canonical_tokens(ground_truth):
        return 1.0
    pred_tokens = pred.split()
    gold_tokens = gold.split()
    if _acronym(pred_tokens) == "".join(gold_tokens) or _acronym(gold_tokens) == "".join(pred_tokens):
        if min(len(pred_tokens), len(gold_tokens)) == 1 and max(len(pred_tokens), len(gold_tokens)) >= 2:
            return 1.0
    return float(_strict_loose_match(pred, gold))


def _strict_loose_match(pred: str, gold: str) -> bool:
    pred_tokens = _canonical_tokens(pred)
    gold_tokens = _canonical_tokens(gold)
    if not pred_tokens or not gold_tokens:
        return False

    shorter, longer = (pred_tokens, gold_tokens) if len(pred_tokens) <= len(gold_tokens) else (gold_tokens, pred_tokens)
    start = _contiguous_subsequence_start(shorter, longer)
    if start < 0:
        return False

    if len(shorter) >= 2:
        if sum(len(token) for token in shorter) < 5:
            return False
        return True

    token = shorter[0]
    if any(ch.isdigit() for ch in token):
        return True

    # Conservative one-token allowance for abbreviated named entities such as
    # "Gatwick" vs "Gatwick Airport"; avoid broad adjectives like "American".
    typed_suffixes = {
        "airport",
        "castle",
        "county",
        "district",
        "province",
        "river",
        "university",
        "comic",
        "comics",
        "film",
        "city",
    }
    geo_suffixes = {
        "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
        "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho",
        "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana",
        "maine", "maryland", "massachusetts", "michigan", "minnesota",
        "mississippi", "missouri", "montana", "nebraska", "nevada",
        "hampshire", "jersey", "mexico", "york", "carolina", "dakota",
        "ohio", "oklahoma", "oregon", "pennsylvania", "tennessee", "texas",
        "utah", "vermont", "virginia", "washington", "wisconsin", "wyoming",
        "canada", "kenya", "italy", "france", "germany", "spain", "mexico",
        "pakistan", "india", "australia", "england", "scotland", "ireland",
        "america",
    }
    return (
        len(longer) == 2
        and start == 0
        and longer[-1] in typed_suffixes | geo_suffixes
        and len(token) >= 4
    )


def _is_contiguous_subsequence(shorter: List[str], longer: List[str]) -> bool:
    return _contiguous_subsequence_start(shorter, longer) >= 0


def _contiguous_subsequence_start(shorter: List[str], longer: List[str]) -> int:
    if len(shorter) > len(longer):
        return -1
    width = len(shorter)
    for idx in range(len(longer) - width + 1):
        if longer[idx : idx + width] == shorter:
            return idx
    return -1


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = _canonical_tokens(prediction)
    gold_tokens = _canonical_tokens(ground_truth)
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common: Dict[str, int] = {}
    for token in gold_tokens:
        common[token] = common.get(token, 0) + 1
    num_same = 0
    for token in pred_tokens:
        if common.get(token, 0) > 0:
            num_same += 1
            common[token] -= 1
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def metric_max_over_ground_truths(prediction: str, ground_truths: Sequence[str]) -> Tuple[float, float]:
    if not ground_truths:
        return 0.0, 0.0
    scores = []
    for gold in ground_truths:
        em = exact_match_score(prediction, gold)
        f1 = 1.0 if em == 1.0 else f1_score(prediction, gold)
        scores.append((em, f1))
    em = max(item[0] for item in scores)
    f1 = max(item[1] for item in scores)
    return em, f1


def read_jsonl(path: Path, *, limit: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows


def extract_gold_answers(row: Dict[str, Any], *, answer_field: str, aliases_field: str) -> List[str]:
    answers: List[str] = []
    raw_answer = row.get(answer_field)
    if isinstance(raw_answer, list):
        answers.extend(str(item) for item in raw_answer if str(item).strip())
    elif raw_answer is not None and str(raw_answer).strip():
        answers.append(str(raw_answer))

    aliases = row.get(aliases_field) or []
    if isinstance(aliases, list):
        answers.extend(str(item) for item in aliases if str(item).strip())
    elif str(aliases).strip():
        answers.append(str(aliases))

    deduped: List[str] = []
    seen = set()
    for answer in answers:
        key = normalize_answer(answer)
        if key and key not in seen:
            seen.add(key)
            deduped.append(answer)
    return deduped


def parse_dataset_specs(specs: Iterable[str]) -> Dict[str, Path]:
    specs = list(specs)
    datasets = {} if specs else {name: Path(path) for name, path in DEFAULT_DATASETS.items()}
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"Invalid --dataset value: {spec}. Use name=/path/file.jsonl")
        name, path = spec.split("=", 1)
        datasets[name.strip()] = Path(path.strip())
    return datasets


def build_eval_parser() -> argparse.ArgumentParser:
    pipeline_parser = build_arg_parser()
    parser = argparse.ArgumentParser(
        description="Evaluate OPERA on QA datasets and compute EM/F1.",
        parents=[pipeline_parser],
        conflict_handler="resolve",
    )
    parser.set_defaults(question=None, input_jsonl=None)
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Dataset in name=/path/file.jsonl form. If omitted, defaults to HotpotQA, 2Wiki, and MuSiQue.",
    )
    parser.add_argument("--eval-limit", type=int, default=100)
    parser.add_argument("--answer-field", type=str, default="answer")
    parser.add_argument("--aliases-field", type=str, default="answer_aliases")
    parser.add_argument("--eval-output-dir", type=str, default=str(PROJECT_ROOT / "eval_runs"))
    parser.add_argument(
        "--dataset-aware-retriever",
        action="store_true",
        help="When using the HTTP retriever, send the dataset name with each search request.",
    )
    return parser


def evaluate_dataset(
    *,
    dataset_name: str,
    dataset_path: Path,
    rows: List[Dict[str, Any]],
    pipeline,
    args: argparse.Namespace,
    run_dir: Path,
) -> Dict[str, Any]:
    dataset_dir = run_dir / dataset_name
    traces_dir = dataset_dir / "traces"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    traces_dir.mkdir(parents=True, exist_ok=True)
    results_path = dataset_dir / "results.jsonl"

    result_rows: List[Dict[str, Any]] = []
    start_dataset = time.perf_counter()
    with results_path.open("w", encoding="utf-8") as results_file:
        for idx, row in enumerate(rows, start=1):
            qid = str(row.get(args.id_field) or row.get("id") or f"{dataset_name}_{idx}")
            question = str(row.get(args.question_field) or row.get("question") or "")
            gold_answers = extract_gold_answers(
                row,
                answer_field=args.answer_field,
                aliases_field=args.aliases_field,
            )
            safe_id = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in qid)
            trace_path = traces_dir / f"{safe_id}.json"
            t0 = time.perf_counter()
            error = None
            try:
                trace = pipeline.run(question, question_id=qid)
                write_json(trace_path, trace)
                prediction = str(trace.get("final_answer") or "")
                stats = trace.get("stats", {})
                success = bool(trace.get("success"))
            except Exception as exc:
                prediction = ""
                stats = {}
                success = False
                error = str(exc)
                write_json(trace_path, {"id": qid, "question": question, "error": error})
            latency = time.perf_counter() - t0
            em, f1 = metric_max_over_ground_truths(prediction, gold_answers)
            if em > f1 + 1e-12:
                raise AssertionError(
                    f"Metric invariant violated for {qid}: EM={em}, F1={f1}, prediction={prediction!r}"
                )
            result = {
                "dataset": dataset_name,
                "index": idx,
                "id": qid,
                "question": question,
                "prediction": prediction,
                "gold_answers": gold_answers,
                "gold_answer_count": len(gold_answers),
                "exact_match": em,
                "f1": f1,
                "success": success,
                "latency_sec": latency,
                "trace_path": str(trace_path),
                "stats": stats,
                "error": error,
            }
            results_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            results_file.flush()
            result_rows.append(result)
            print(json.dumps({
                "dataset": dataset_name,
                "idx": idx,
                "id": qid,
                "em": em,
                "f1": round(f1, 4),
                "gold_answer_count": len(gold_answers),
                "success": success,
                "latency_sec": round(latency, 2),
                "answer": prediction[:160],
                "error": error,
            }, ensure_ascii=False), flush=True)

    total_time = time.perf_counter() - start_dataset
    n = len(result_rows)
    summary = {
        "dataset": dataset_name,
        "dataset_path": str(dataset_path),
        "num_questions": n,
        "exact_match": sum(row["exact_match"] for row in result_rows) / n if n else 0.0,
        "f1": sum(row["f1"] for row in result_rows) / n if n else 0.0,
        "success_rate": sum(1 for row in result_rows if row["success"]) / n if n else 0.0,
        "error_rate": sum(1 for row in result_rows if row["error"]) / n if n else 0.0,
        "total_time_sec": total_time,
        "avg_latency_sec": sum(row["latency_sec"] for row in result_rows) / n if n else 0.0,
        "avg_retrieval_calls": sum(row["stats"].get("retrieval_calls", 0) for row in result_rows) / n if n else 0.0,
        "avg_rewrite_calls": sum(row["stats"].get("rewrite_calls", 0) for row in result_rows) / n if n else 0.0,
        "avg_analysis_calls": sum(row["stats"].get("analysis_calls", 0) for row in result_rows) / n if n else 0.0,
        "results_path": str(results_path),
        "traces_dir": str(traces_dir),
    }
    write_json(dataset_dir / "summary.json", summary)
    return summary


def main() -> None:
    args = build_eval_parser().parse_args()
    args.output_dir = args.eval_output_dir
    datasets = parse_dataset_specs(args.dataset)
    run_dir = Path(args.eval_output_dir) / timestamp()
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "args.json", vars(args))

    for name, path in datasets.items():
        if not path.exists():
            raise FileNotFoundError(f"{name}: {path}")

    pipeline = _build_pipeline(args)
    summaries: List[Dict[str, Any]] = []
    for dataset_name, dataset_path in datasets.items():
        if args.dataset_aware_retriever and hasattr(pipeline.retriever, "set_dataset"):
            pipeline.retriever.set_dataset(dataset_name)
        rows = read_jsonl(dataset_path, limit=args.eval_limit)
        summaries.append(
            evaluate_dataset(
                dataset_name=dataset_name,
                dataset_path=dataset_path,
                rows=rows,
                pipeline=pipeline,
                args=args,
                run_dir=run_dir,
            )
        )

    overall = {
        "run_dir": str(run_dir),
        "num_datasets": len(summaries),
        "num_questions": sum(item["num_questions"] for item in summaries),
        "macro_exact_match": sum(item["exact_match"] for item in summaries) / len(summaries) if summaries else 0.0,
        "macro_f1": sum(item["f1"] for item in summaries) / len(summaries) if summaries else 0.0,
        "macro_success_rate": sum(item["success_rate"] for item in summaries) / len(summaries) if summaries else 0.0,
        "datasets": summaries,
    }
    write_json(run_dir / "overall_summary.json", overall)
    print(json.dumps(overall, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
