#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from opera.evaluate import metric_max_over_ground_truths


def normalize(text: str) -> str:
    table = str.maketrans({ch: " " for ch in string.punctuation})
    text = str(text or "").lower().translate(table)
    return " ".join(text.split())


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def trace_documents_text(trace_path: Path) -> str:
    if not trace_path.exists():
        return ""
    try:
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
    except Exception:
        return ""

    chunks: List[str] = []
    for step in trace.get("steps", []) or []:
        for attempt in step.get("attempts", []) or []:
            for doc in attempt.get("documents", []) or []:
                chunks.append(str(doc.get("title") or ""))
                chunks.append(str(doc.get("content") or ""))
    return normalize(" ".join(chunks))


def musique_hop(row: Dict[str, Any]) -> str:
    qid = str(row.get("id") or "")
    match = re.search(r"(\d)hop", qid)
    if match:
        return f"{match.group(1)}hop"
    return "unknown"


def summarize_dataset(dataset_dir: Path) -> Dict[str, Any]:
    results_path = dataset_dir / "results.jsonl"
    rows = list(iter_jsonl(results_path)) if results_path.exists() else []
    n = len(rows)
    if not n:
        return {"dataset": dataset_dir.name, "num_questions": 0}

    counters = Counter()
    hop_stats: Dict[str, Counter] = defaultdict(Counter)
    examples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for row in rows:
        pred = str(row.get("prediction") or "")
        em, f1 = metric_max_over_ground_truths(pred, row.get("gold_answers") or [])
        success = bool(row.get("success"))
        not_found = "not found" in pred.lower()

        golds = [normalize(item) for item in row.get("gold_answers", []) if normalize(item)]
        docs_text = trace_documents_text(Path(row.get("trace_path") or ""))
        gold_in_docs = any(gold in docs_text for gold in golds)

        counters["em_sum"] += em
        counters["f1_sum"] += f1
        counters["success"] += int(success)
        counters["not_found"] += int(not_found)
        counters["partial_f1"] += int(em == 0 and f1 > 0)
        counters["success_wrong"] += int(success and em == 0)
        counters["failed_right"] += int((not success) and em == 1)
        counters["wrong_gold_in_docs"] += int(em == 0 and gold_in_docs)
        counters["notfound_gold_in_docs"] += int(em == 0 and not_found and gold_in_docs)

        if dataset_dir.name == "musique":
            hop = musique_hop(row)
            hop_stats[hop]["n"] += 1
            hop_stats[hop]["em_sum"] += em
            hop_stats[hop]["f1_sum"] += f1
            hop_stats[hop]["success"] += int(success)

        for key, flag in {
            "not_found": not_found,
            "success_wrong": success and em == 0,
            "failed_right": (not success) and em == 1,
            "partial_f1": em == 0 and f1 > 0,
            "wrong_gold_in_docs": em == 0 and gold_in_docs,
        }.items():
            if flag and len(examples[key]) < 5:
                examples[key].append(
                    {
                        "index": row.get("index"),
                        "id": row.get("id"),
                        "question": row.get("question"),
                        "prediction": pred,
                        "gold_answers": row.get("gold_answers"),
                        "f1": f1,
                        "success": success,
                        "trace_path": row.get("trace_path"),
                    }
                )

    hop_summary = {}
    for hop, stat in sorted(hop_stats.items()):
        hop_n = stat["n"]
        hop_summary[hop] = {
            "num_questions": hop_n,
            "exact_match": stat["em_sum"] / hop_n if hop_n else 0.0,
            "f1": stat["f1_sum"] / hop_n if hop_n else 0.0,
            "success_rate": stat["success"] / hop_n if hop_n else 0.0,
        }

    return {
        "dataset": dataset_dir.name,
        "num_questions": n,
        "exact_match": counters["em_sum"] / n,
        "f1": counters["f1_sum"] / n,
        "success_rate": counters["success"] / n,
        "not_found_rate": counters["not_found"] / n,
        "partial_f1_rate": counters["partial_f1"] / n,
        "success_wrong_rate": counters["success_wrong"] / n,
        "failed_right_rate": counters["failed_right"] / n,
        "wrong_gold_in_docs_rate": counters["wrong_gold_in_docs"] / n,
        "notfound_gold_in_docs_rate": counters["notfound_gold_in_docs"] / n,
        "counts": dict(counters),
        "hop_summary": hop_summary,
        "examples": dict(examples),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze OPERA eval logs using the current scorer.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    datasets = []
    for child in sorted(args.run_dir.iterdir()):
        if child.is_dir() and (child / "results.jsonl").exists():
            datasets.append(summarize_dataset(child))

    report = {"run_dir": str(args.run_dir), "datasets": datasets}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
