#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from build_opera_index import (  # noqa: E402
    DEFAULT_2WIKI_DEV,
    DEFAULT_2WIKI_TRAIN,
    DEFAULT_HOTPOT_DEV,
    DEFAULT_MIXED_INDEX,
    DEFAULT_MIXED_META,
    DEFAULT_MUSIQUE_ARROW_DIR,
    DEFAULT_MUSIQUE_META,
    compute_plan,
)
from index_build_utils import (  # noqa: E402
    doc_key,
    normalize_space,
    parse_splits,
    read_jsonl,
    support_sentence_keys,
)


DEFAULT_MUSIQUE_DEV = Path(os.environ["OPERA_MUSIQUE_DEV_JSONL"]) if os.environ.get("OPERA_MUSIQUE_DEV_JSONL") else None

DocKey = Tuple[str, str]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_raw_map_json(path: Path) -> Dict[str, Dict[str, Any]]:
    return {str(row.get("_id") or row.get("id")): row for row in read_json(path)}


def build_raw_map_jsonl(path: Path) -> Dict[str, Dict[str, Any]]:
    return {str(row.get("id")): row for row in read_jsonl(path)}


def musique_support_paragraph_keys(row: Dict[str, Any]) -> Set[DocKey]:
    paragraphs = row.get("paragraphs") or []
    by_idx = {
        int(paragraph.get("idx")): paragraph
        for paragraph in paragraphs
        if isinstance(paragraph, dict) and paragraph.get("idx") is not None
    }
    support_indices: Set[int] = set()
    for paragraph in paragraphs:
        if isinstance(paragraph, dict) and paragraph.get("is_supporting") is True and paragraph.get("idx") is not None:
            support_indices.add(int(paragraph["idx"]))
    for step in row.get("question_decomposition") or []:
        support_idx = (step or {}).get("paragraph_support_idx")
        if support_idx is not None:
            support_indices.add(int(support_idx))

    keys: Set[DocKey] = set()
    for support_idx in support_indices:
        paragraph = by_idx.get(support_idx)
        if not paragraph:
            continue
        title = paragraph.get("title")
        text = paragraph.get("paragraph_text")
        if title and text:
            keys.add(doc_key(title, text))
    return keys


def build_planned_doc_sets(plan: Dict[str, Any]) -> tuple[Set[DocKey], Dict[str, List[str]]]:
    exact_keys: Set[DocKey] = set()
    by_title: Dict[str, List[str]] = defaultdict(list)

    excluded = plan["excluded_old_indices"]
    for idx, doc in enumerate(plan["mixed_docs"]):
        if idx in excluded:
            continue
        key = doc_key(doc.get("title"), doc.get("content"))
        exact_keys.add(key)
        by_title[key[0]].append(key[1])

    for doc in plan["new_musique_docs"]:
        key = doc_key(doc.get("title"), doc.get("content"))
        exact_keys.add(key)
        by_title[key[0]].append(key[1])

    return exact_keys, by_title


def key_present(key: DocKey, exact_keys: Set[DocKey], by_title: Dict[str, List[str]]) -> bool:
    if key in exact_keys:
        return True
    title, content = key
    content = normalize_space(content)
    for candidate in by_title.get(title, []):
        candidate = normalize_space(candidate)
        if content and (content in candidate or candidate in content):
            return True
    return False


def iter_eval_gold_keys(
    eval_set_paths: Sequence[Path],
    *,
    hotpot_rows: Dict[str, Dict[str, Any]],
    wiki_rows: Dict[str, Dict[str, Any]],
    musique_rows: Dict[str, Dict[str, Any]],
) -> Iterable[tuple[str, str, DocKey]]:
    for eval_path in eval_set_paths:
        for item in read_jsonl(eval_path):
            dataset = str(item.get("source_dataset") or item.get("dataset") or "")
            original_id = str(item.get("original_id") or item.get("id") or "")
            qid = str(item.get("id") or original_id)
            if dataset == "hotpotqa":
                row = hotpot_rows.get(original_id)
                if row:
                    for key in support_sentence_keys(row):
                        yield "hotpotqa", qid, key
            elif dataset in {"2wikimultihopqa", "2wiki"}:
                row = wiki_rows.get(original_id)
                if row:
                    for key in support_sentence_keys(row):
                        yield "2wiki", qid, key
            elif dataset == "musique":
                row = musique_rows.get(original_id)
                if row:
                    for key in musique_support_paragraph_keys(row):
                        yield "musique", qid, key


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify support-document coverage for an OPERA index rebuild.")
    parser.add_argument("--mixed-index", type=Path, default=DEFAULT_MIXED_INDEX, required=DEFAULT_MIXED_INDEX is None)
    parser.add_argument("--mixed-meta", type=Path, default=DEFAULT_MIXED_META, required=DEFAULT_MIXED_META is None)
    parser.add_argument("--musique-meta", type=Path, default=DEFAULT_MUSIQUE_META, required=DEFAULT_MUSIQUE_META is None)
    parser.add_argument("--musique-arrow-dir", type=Path, default=DEFAULT_MUSIQUE_ARROW_DIR, required=DEFAULT_MUSIQUE_ARROW_DIR is None)
    parser.add_argument("--wiki-train-json", type=Path, default=DEFAULT_2WIKI_TRAIN, required=DEFAULT_2WIKI_TRAIN is None)
    parser.add_argument("--hotpot-dev-json", type=Path, default=DEFAULT_HOTPOT_DEV, required=DEFAULT_HOTPOT_DEV is None)
    parser.add_argument("--2wiki-dev-json", type=Path, default=DEFAULT_2WIKI_DEV, required=DEFAULT_2WIKI_DEV is None)
    parser.add_argument("--musique-dev-jsonl", type=Path, default=DEFAULT_MUSIQUE_DEV, required=DEFAULT_MUSIQUE_DEV is None)
    parser.add_argument("--eval-set", type=Path, action="append", required=True)
    parser.add_argument("--target-total", type=int, default=None)
    parser.add_argument("--old-musique-splits", type=parse_splits, default=parse_splits("dev,train"))
    parser.add_argument("--new-musique-splits", type=parse_splits, default=parse_splits("dev,train"))
    parser.add_argument("--window-radius", type=int, default=1)
    parser.add_argument("--max-windows-per-paragraph", type=int, default=0)
    parser.add_argument("--min-window-chars", type=int, default=35)
    parser.add_argument("--no-protect-full-dev-support", action="store_true")
    parser.add_argument("--max-missing-examples", type=int, default=20)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    args.protect_eval_set = args.eval_set
    args.output_dir = Path("/tmp")
    args.output_name = "coverage_only"
    args.reconstruct_batch_size = 8192
    args.embed_batch_size = 96
    args.bge_model = ""
    args.bge_fp32 = False

    plan = compute_plan(args)
    exact_keys, by_title = build_planned_doc_sets(plan)

    hotpot_rows = build_raw_map_json(args.hotpot_dev_json)
    wiki_rows = build_raw_map_json(args.__dict__["2wiki_dev_json"])
    musique_rows = build_raw_map_jsonl(args.musique_dev_jsonl)

    total_by_dataset: Dict[str, int] = defaultdict(int)
    missing_by_dataset: Dict[str, int] = defaultdict(int)
    missing_examples: List[Dict[str, str]] = []

    for dataset, qid, key in iter_eval_gold_keys(
        args.eval_set,
        hotpot_rows=hotpot_rows,
        wiki_rows=wiki_rows,
        musique_rows=musique_rows,
    ):
        total_by_dataset[dataset] += 1
        if not key_present(key, exact_keys, by_title):
            missing_by_dataset[dataset] += 1
            if len(missing_examples) < args.max_missing_examples:
                missing_examples.append(
                    {
                        "dataset": dataset,
                        "id": qid,
                        "title": key[0],
                        "content": key[1][:300],
                    }
                )

    datasets = sorted(set(total_by_dataset) | set(missing_by_dataset))
    summary = {
        "planned_corpus": plan["stats"],
        "eval_sets": [str(path) for path in args.eval_set],
        "gold_docs_total": sum(total_by_dataset.values()),
        "gold_docs_missing": sum(missing_by_dataset.values()),
        "coverage_by_dataset": {
            dataset: {
                "gold_docs_total": total_by_dataset[dataset],
                "gold_docs_missing": missing_by_dataset[dataset],
                "coverage": (
                    1.0 - missing_by_dataset[dataset] / total_by_dataset[dataset]
                    if total_by_dataset[dataset]
                    else 1.0
                ),
            }
            for dataset in datasets
        },
        "missing_examples": missing_examples,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if summary["gold_docs_missing"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
