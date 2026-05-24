#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

from index_build_utils import (
    DEFAULT_2WIKI_DEV,
    DEFAULT_2WIKI_TRAIN,
    DEFAULT_BGE_LOCAL,
    DEFAULT_HOTPOT_DEV,
    DEFAULT_MIXED_INDEX,
    DEFAULT_MIXED_META,
    DEFAULT_MUSIQUE_ARROW_DIR,
    DEFAULT_MUSIQUE_META,
    DEFAULT_OUTPUT_DIR,
    DocKey,
    add_old_vectors,
    all_dev_support_sentence_keys,
    doc_key,
    extract_2wiki_train_sentence_keys,
    extract_old_musique_sentence_keys,
    load_pickle,
    normalize_space,
    parse_splits,
    protected_eval_sentence_keys,
    transform_retained_mixed_doc,
    unwrap_documents,
)


DEFAULT_OUTPUT_NAME = "OPERA-index"


def split_sentences(text: str) -> List[str]:
    text = normalize_space(text)
    if not text:
        return []
    raw_sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences: List[str] = []
    for sentence in raw_sentences:
        sentence = normalize_space(sentence)
        if not sentence:
            continue
        if len(sentence) < 25 and sentences:
            sentences[-1] = normalize_space(f"{sentences[-1]} {sentence}")
            continue
        sentences.append(sentence)
    return sentences


def embedding_text(doc: Dict[str, Any]) -> str:
    title = normalize_space(doc.get("title"))
    content = normalize_space(doc.get("content") or doc.get("text") or doc.get("paragraph_text"))
    return normalize_space(f"{title}. {content}") if title else content


def transform_musique_paragraph(raw: Dict[str, Any], idx: int) -> Dict[str, Any]:
    title = normalize_space(raw.get("title"))
    content = normalize_space(raw.get("content") or raw.get("text") or raw.get("paragraph_text"))
    source = normalize_space(raw.get("source"))
    doc_hash = normalize_space(raw.get("doc_hash") or f"musique_{idx}")
    return {
        "id": f"musique_para_{source}_{doc_hash}",
        "content": content,
        "title": title,
        "metadata": {
            "title": title,
            "dataset": "musique",
            "source": source,
            "chunk_type": "paragraph",
            "doc_hash": doc_hash,
        },
    }


def transform_musique_window(
    raw: Dict[str, Any],
    idx: int,
    *,
    window_id: int,
    sentence_start: int,
    sentence_end: int,
    content: str,
) -> Dict[str, Any]:
    title = normalize_space(raw.get("title"))
    source = normalize_space(raw.get("source"))
    doc_hash = normalize_space(raw.get("doc_hash") or f"musique_{idx}")
    return {
        "id": f"musique_win_{source}_{doc_hash}_{window_id}",
        "content": content,
        "title": title,
        "metadata": {
            "title": title,
            "dataset": "musique",
            "source": source,
            "chunk_type": "sentence_window",
            "parent_doc_hash": doc_hash,
            "window_id": window_id,
            "sentence_start": sentence_start,
            "sentence_end": sentence_end,
        },
    }


def select_musique_retrieval_docs(
    docs: List[Dict[str, Any]],
    splits: Sequence[str],
    *,
    window_radius: int,
    max_windows_per_paragraph: int,
    min_window_chars: int,
) -> List[Dict[str, Any]]:
    wanted = set(splits)
    new_docs: List[Dict[str, Any]] = []
    seen: Set[DocKey] = set()

    for idx, raw in enumerate(docs):
        source = normalize_space(raw.get("source"))
        if source not in wanted:
            continue
        title = normalize_space(raw.get("title"))
        content = normalize_space(raw.get("content") or raw.get("text") or raw.get("paragraph_text"))
        if not title or not content:
            continue

        paragraph_key = doc_key(title, content)
        if paragraph_key not in seen:
            seen.add(paragraph_key)
            new_docs.append(transform_musique_paragraph(raw, idx))

        sentences = split_sentences(content)
        if not sentences:
            continue
        window_limit = len(sentences) if max_windows_per_paragraph <= 0 else min(len(sentences), max_windows_per_paragraph)
        for window_id, sent_idx in enumerate(range(window_limit)):
            start = max(0, sent_idx - window_radius)
            end = min(len(sentences), sent_idx + window_radius + 1)
            window_text = normalize_space(" ".join(sentences[start:end]))
            if len(window_text) < min_window_chars:
                continue
            key = doc_key(title, window_text)
            if key in seen:
                continue
            seen.add(key)
            new_docs.append(
                transform_musique_window(
                    raw,
                    idx,
                    window_id=window_id,
                    sentence_start=start,
                    sentence_end=end,
                    content=window_text,
                )
            )

    return new_docs


def low_information_drop_key(doc: Dict[str, Any], idx: int) -> Tuple[int, int, int, str]:
    content = normalize_space(doc.get("content") or doc.get("contents") or doc.get("text"))
    title = normalize_space(doc.get("title") or (doc.get("metadata") or {}).get("title"))
    length = len(content)
    alpha_count = sum(ch.isalpha() for ch in content)
    token_count = len(content.split())
    starts_fragment = bool(content[:1] and content[:1].islower())
    lacks_terminal = bool(content and content[-1] not in ".!?")

    if length < 40:
        bucket = 0
    elif length < 80:
        bucket = 1
    elif length < 120:
        bucket = 2
    elif token_count < 12 or alpha_count < 40 or starts_fragment or lacks_terminal:
        bucket = 3
    else:
        bucket = 4
    return bucket, length, token_count, f"{title}\t{idx}"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a fixed-size mixed corpus where MuSiQue uses paragraph plus sentence-window chunks "
            "and excess capacity is taken from low-information unprotected 2Wiki train sentences."
        )
    )
    parser.add_argument("--mixed-index", type=Path, default=DEFAULT_MIXED_INDEX, required=DEFAULT_MIXED_INDEX is None)
    parser.add_argument("--mixed-meta", type=Path, default=DEFAULT_MIXED_META, required=DEFAULT_MIXED_META is None)
    parser.add_argument("--musique-meta", type=Path, default=DEFAULT_MUSIQUE_META, required=DEFAULT_MUSIQUE_META is None)
    parser.add_argument("--musique-arrow-dir", type=Path, default=DEFAULT_MUSIQUE_ARROW_DIR, required=DEFAULT_MUSIQUE_ARROW_DIR is None)
    parser.add_argument("--wiki-train-json", type=Path, default=DEFAULT_2WIKI_TRAIN, required=DEFAULT_2WIKI_TRAIN is None)
    parser.add_argument("--hotpot-dev-json", type=Path, default=DEFAULT_HOTPOT_DEV, required=DEFAULT_HOTPOT_DEV is None)
    parser.add_argument("--2wiki-dev-json", type=Path, default=DEFAULT_2WIKI_DEV, required=DEFAULT_2WIKI_DEV is None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-name", type=str, default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--target-total", type=int, default=None)
    parser.add_argument("--old-musique-splits", type=parse_splits, default=parse_splits("dev,train"))
    parser.add_argument("--new-musique-splits", type=parse_splits, default=parse_splits("dev,train"))
    parser.add_argument("--window-radius", type=int, default=1)
    parser.add_argument(
        "--max-windows-per-paragraph",
        type=int,
        default=0,
        help="0 means keep every sentence-centered window.",
    )
    parser.add_argument("--min-window-chars", type=int, default=35)
    parser.add_argument(
        "--protect-eval-set",
        type=Path,
        action="append",
        default=[],
        help="Evaluation JSONL whose HotpotQA/2Wiki support sentences must not be dropped while balancing.",
    )
    parser.add_argument(
        "--no-protect-full-dev-support",
        action="store_true",
        help="Disable default protection of all HotpotQA/2Wiki dev support sentences.",
    )
    parser.add_argument("--reconstruct-batch-size", type=int, default=8192)
    parser.add_argument("--embed-batch-size", type=int, default=96)
    parser.add_argument("--bge-model", type=str, default=DEFAULT_BGE_LOCAL)
    parser.add_argument("--bge-fp32", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def compute_plan(args: argparse.Namespace) -> Dict[str, Any]:
    mixed_docs = unwrap_documents(load_pickle(args.mixed_meta))
    musique_docs = unwrap_documents(load_pickle(args.musique_meta))
    target_total = args.target_total or len(mixed_docs)

    old_musique_keys = extract_old_musique_sentence_keys(args.musique_arrow_dir, args.old_musique_splits)
    old_musique_indices = [
        idx
        for idx, doc in enumerate(mixed_docs)
        if doc_key(doc.get("title"), doc.get("content")) in old_musique_keys
    ]
    old_musique_set = set(old_musique_indices)

    new_musique_docs = select_musique_retrieval_docs(
        musique_docs,
        args.new_musique_splits,
        window_radius=max(0, args.window_radius),
        max_windows_per_paragraph=args.max_windows_per_paragraph,
        min_window_chars=args.min_window_chars,
    )

    retained_after_replace = len(mixed_docs) - len(old_musique_indices)
    total_before_balancing = retained_after_replace + len(new_musique_docs)
    drop_needed = total_before_balancing - target_total
    if drop_needed < 0:
        raise ValueError(
            f"Replacement would create only {total_before_balancing} docs, below target {target_total}."
        )

    wiki_train_keys = extract_2wiki_train_sentence_keys(args.wiki_train_json) if drop_needed else set()
    protected_keys: Set[DocKey] = set()
    if not args.no_protect_full_dev_support:
        protected_keys.update(
            all_dev_support_sentence_keys(
                hotpot_dev=args.hotpot_dev_json,
                wiki_dev=args.__dict__["2wiki_dev_json"],
            )
        )
    protected_keys.update(
        protected_eval_sentence_keys(
            args.protect_eval_set,
            hotpot_dev=args.hotpot_dev_json,
            wiki_dev=args.__dict__["2wiki_dev_json"],
        )
    )

    wiki_train_indices = [
        idx
        for idx, doc in enumerate(mixed_docs)
        if idx not in old_musique_set and doc_key(doc.get("title"), doc.get("content")) in wiki_train_keys
    ]
    protected_wiki_train_indices = [
        idx
        for idx in wiki_train_indices
        if doc_key(mixed_docs[idx].get("title"), mixed_docs[idx].get("content")) in protected_keys
    ]
    droppable_wiki_train_indices = [
        idx
        for idx in wiki_train_indices
        if doc_key(mixed_docs[idx].get("title"), mixed_docs[idx].get("content")) not in protected_keys
    ]
    if drop_needed > len(droppable_wiki_train_indices):
        raise ValueError(
            f"Need to drop {drop_needed} docs, but only found {len(droppable_wiki_train_indices)} "
            "unprotected 2Wiki train docs."
        )

    ranked_droppable = sorted(
        droppable_wiki_train_indices,
        key=lambda idx: low_information_drop_key(mixed_docs[idx], idx),
    )
    dropped_wiki_train_indices = set(ranked_droppable[:drop_needed])
    dropped_lengths = [
        len(normalize_space(mixed_docs[idx].get("content") or mixed_docs[idx].get("contents") or mixed_docs[idx].get("text")))
        for idx in dropped_wiki_train_indices
    ]
    excluded_old_indices = old_musique_set | dropped_wiki_train_indices
    final_total = len(mixed_docs) - len(excluded_old_indices) + len(new_musique_docs)

    chunk_counts: Dict[str, int] = {}
    for doc in new_musique_docs:
        chunk_type = str((doc.get("metadata") or {}).get("chunk_type") or "unknown")
        chunk_counts[chunk_type] = chunk_counts.get(chunk_type, 0) + 1

    return {
        "mixed_docs": mixed_docs,
        "new_musique_docs": new_musique_docs,
        "old_musique_indices": old_musique_indices,
        "dropped_wiki_train_indices": dropped_wiki_train_indices,
        "excluded_old_indices": excluded_old_indices,
        "stats": {
            "old_mixed_total": len(mixed_docs),
            "target_total": target_total,
            "old_musique_sentence_keys": len(old_musique_keys),
            "old_musique_docs_removed": len(old_musique_indices),
            "new_musique_docs": len(new_musique_docs),
            "new_musique_chunk_counts": chunk_counts,
            "total_before_balancing": total_before_balancing,
            "drop_needed_from_2wiki_train": drop_needed,
            "available_2wiki_train_docs": len(wiki_train_indices),
            "protected_eval_gold_sentence_keys": len(protected_keys),
            "protect_full_dev_support": not args.no_protect_full_dev_support,
            "protected_2wiki_train_docs": len(protected_wiki_train_indices),
            "droppable_2wiki_train_docs": len(droppable_wiki_train_indices),
            "dropped_2wiki_train_docs": len(dropped_wiki_train_indices),
            "dropped_2wiki_train_avg_chars": (
                sum(dropped_lengths) / len(dropped_lengths) if dropped_lengths else 0.0
            ),
            "dropped_2wiki_train_lt80": sum(length < 80 for length in dropped_lengths),
            "dropped_2wiki_train_lt120": sum(length < 120 for length in dropped_lengths),
            "final_total": final_total,
            "new_musique_splits": list(args.new_musique_splits),
            "old_musique_splits": list(args.old_musique_splits),
            "window_radius": args.window_radius,
            "max_windows_per_paragraph": args.max_windows_per_paragraph,
            "min_window_chars": args.min_window_chars,
            "drop_strategy": "lowest_information_unprotected_2wiki_train_first",
        },
    }


def add_musique_vectors_by_encoding(
    *,
    faiss: Any,
    output_index: Any,
    new_musique_docs: List[Dict[str, Any]],
    output_metadata: List[Dict[str, Any]],
    bge_model: str,
    use_fp16: bool,
    batch_size: int,
) -> None:
    import numpy as np
    from FlagEmbedding import BGEM3FlagModel

    model = BGEM3FlagModel(bge_model if Path(bge_model).exists() else "BAAI/bge-m3", use_fp16=use_fp16)
    for start in range(0, len(new_musique_docs), batch_size):
        batch = new_musique_docs[start : start + batch_size]
        texts = [embedding_text(doc) for doc in batch]
        try:
            encoded = model.encode(
                texts,
                batch_size=batch_size,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
        except TypeError:
            encoded = model.encode(texts, batch_size=batch_size)
        vectors = encoded["dense_vecs"] if isinstance(encoded, dict) else encoded
        matrix = np.asarray(vectors, dtype="float32")
        faiss.normalize_L2(matrix)
        output_index.add(matrix)
        output_metadata.extend(batch)
        print(
            json.dumps(
                {
                    "event": "encoded_musique_retrieval_vectors",
                    "processed": start + len(batch),
                    "output_index_size": int(output_index.ntotal),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )


def build_index(args: argparse.Namespace, plan: Dict[str, Any]) -> None:
    import faiss

    args.output_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.output_dir / f"{args.output_name}.index"
    meta_path = args.output_dir / f"{args.output_name}.meta"
    info_path = args.output_dir / f"{args.output_name}_info.json"

    old_index = faiss.read_index(str(args.mixed_index))
    dim = old_index.d
    output_index = faiss.IndexFlatIP(dim)
    output_metadata: List[Dict[str, Any]] = []

    add_old_vectors(
        faiss=faiss,
        output_index=output_index,
        old_index=old_index,
        mixed_docs=plan["mixed_docs"],
        excluded_old_indices=plan["excluded_old_indices"],
        output_metadata=output_metadata,
        batch_size=args.reconstruct_batch_size,
    )

    add_musique_vectors_by_encoding(
        faiss=faiss,
        output_index=output_index,
        new_musique_docs=plan["new_musique_docs"],
        output_metadata=output_metadata,
        bge_model=args.bge_model,
        use_fp16=not args.bge_fp32,
        batch_size=args.embed_batch_size,
    )

    expected = plan["stats"]["final_total"]
    if output_index.ntotal != expected or len(output_metadata) != expected:
        raise RuntimeError(
            f"final count mismatch: index={output_index.ntotal}, meta={len(output_metadata)}, expected={expected}"
        )

    faiss.write_index(output_index, str(index_path))
    with meta_path.open("wb") as f:
        pickle.dump(output_metadata, f, protocol=pickle.HIGHEST_PROTOCOL)

    info = dict(plan["stats"])
    info.update(
        {
            "output_index": str(index_path),
            "output_meta": str(meta_path),
            "index_type": "IndexFlatIP",
            "embedding_dim": dim,
            "musique_embedding_mode": "title_prefixed_dual_granularity_reencoded",
            "notes": [
                "Old MuSiQue support-sentence chunks are removed.",
                "MuSiQue replacement uses all selected split paragraphs plus sentence windows.",
                "MuSiQue is_supporting, answers, and decomposition metadata are intentionally not written.",
                "Balancing drops low-information unprotected 2Wiki train sentences first.",
                "All HotpotQA/2Wiki dev support sentences remain protected by default.",
            ],
        }
    )
    with info_path.open("w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(json.dumps({"event": "done", **info}, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    args = build_arg_parser().parse_args()
    plan = compute_plan(args)
    print(json.dumps({"event": "plan", **plan["stats"]}, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return
    build_index(args, plan)


if __name__ == "__main__":
    main()
