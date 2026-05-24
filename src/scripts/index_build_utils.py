#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value else None


DEFAULT_MIXED_INDEX = env_path("OPERA_BASE_INDEX")
DEFAULT_MIXED_META = env_path("OPERA_BASE_METADATA")
DEFAULT_MUSIQUE_INDEX = env_path("OPERA_MUSIQUE_INDEX")
DEFAULT_MUSIQUE_META = env_path("OPERA_MUSIQUE_METADATA")
DEFAULT_MUSIQUE_ARROW_DIR = env_path("OPERA_MUSIQUE_ARROW_DIR")
DEFAULT_2WIKI_TRAIN = env_path("OPERA_2WIKI_TRAIN_JSON")
DEFAULT_HOTPOT_DEV = env_path("OPERA_HOTPOT_DEV_JSON")
DEFAULT_2WIKI_DEV = env_path("OPERA_2WIKI_DEV_JSON")
DEFAULT_OUTPUT_DIR = Path(os.environ.get("OPERA_INDEX_OUTPUT_DIR", str(PROJECT_ROOT / "indexes")))
DEFAULT_OUTPUT_NAME = "OPERA-index-paragraph"
DEFAULT_BGE_LOCAL = os.environ.get("OPERA_BGE_MODEL", "BAAI/bge-m3")

DocKey = Tuple[str, str]


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def doc_key(title: Any, content: Any) -> DocKey:
    return normalize_space(str(title or "")), normalize_space(str(content or ""))


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def unwrap_documents(metadata: Any) -> List[Dict[str, Any]]:
    if isinstance(metadata, dict) and "documents" in metadata:
        metadata = metadata["documents"]
    if not isinstance(metadata, list):
        raise TypeError(f"Unsupported metadata type: {type(metadata)!r}")
    return metadata


def extract_old_musique_sentence_keys(arrow_dir: Path, splits: Sequence[str]) -> Set[DocKey]:
    import pyarrow.ipc as ipc

    keys: Set[DocKey] = set()
    for split in splits:
        path = arrow_dir / f"flash_rag_datasets-{split}.arrow"
        if not path.exists():
            raise FileNotFoundError(f"MuSiQue arrow split not found: {path}")
        with path.open("rb") as f:
            reader = ipc.open_stream(f)
            while True:
                try:
                    batch = reader.read_next_batch().to_pydict()
                except StopIteration:
                    break
                for row_idx in range(len(batch["id"])):
                    metadata = batch["metadata"][row_idx] or {}
                    for step in metadata.get("question_decomposition") or []:
                        paragraph = (step or {}).get("support_paragraph") or {}
                        title = paragraph.get("title")
                        text = paragraph.get("paragraph_text")
                        if not title or not text or not str(text).strip():
                            continue
                        for sentence in str(text).split(". "):
                            sentence = sentence.strip()
                            if not sentence:
                                continue
                            if not sentence.endswith("."):
                                sentence += "."
                            keys.add(doc_key(title, sentence))
    return keys


def extract_2wiki_train_sentence_keys(path: Path) -> Set[DocKey]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    keys: Set[DocKey] = set()
    for item in data:
        for title, sentences in item["context"]:
            for sentence in sentences:
                sentence = normalize_space(sentence)
                if sentence:
                    keys.add(doc_key(title, sentence))
    return keys


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_raw_map(path: Path) -> Dict[str, Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    return {str(row.get("_id") or row.get("id")): row for row in rows}


def support_sentence_keys(row: Dict[str, Any]) -> Set[DocKey]:
    title_to_sentences: Dict[str, List[str]] = {
        str(title): [normalize_space(sentence) for sentence in sentences]
        for title, sentences in row.get("context", [])
    }
    keys: Set[DocKey] = set()
    for fact in row.get("supporting_facts") or []:
        if not isinstance(fact, list) or len(fact) < 2:
            continue
        title = str(fact[0])
        try:
            sent_idx = int(fact[1])
        except (TypeError, ValueError):
            continue
        sentences = title_to_sentences.get(title)
        if sentences is None or not (0 <= sent_idx < len(sentences)):
            continue
        sentence = sentences[sent_idx]
        if sentence:
            keys.add(doc_key(title, sentence))
    return keys


def protected_eval_sentence_keys(
    eval_set_paths: Sequence[Path],
    *,
    hotpot_dev: Path,
    wiki_dev: Path,
) -> Set[DocKey]:
    if not eval_set_paths:
        return set()

    hotpot_rows = build_raw_map(hotpot_dev)
    wiki_rows = build_raw_map(wiki_dev)
    protected: Set[DocKey] = set()
    for eval_path in eval_set_paths:
        for item in read_jsonl(eval_path):
            dataset = str(item.get("source_dataset") or item.get("dataset") or "")
            original_id = str(item.get("original_id") or item.get("id") or "")
            if dataset == "hotpotqa":
                row = hotpot_rows.get(original_id)
            elif dataset in {"2wikimultihopqa", "2wiki"}:
                row = wiki_rows.get(original_id)
            else:
                row = None
            if row:
                protected.update(support_sentence_keys(row))
    return protected


def all_dev_support_sentence_keys(*, hotpot_dev: Path, wiki_dev: Path) -> Set[DocKey]:
    protected: Set[DocKey] = set()
    for row in build_raw_map(hotpot_dev).values():
        protected.update(support_sentence_keys(row))
    for row in build_raw_map(wiki_dev).values():
        protected.update(support_sentence_keys(row))
    return protected


def transform_musique_doc(raw: Dict[str, Any], idx: int, *, title_prefix_content: bool) -> Dict[str, Any]:
    title = normalize_space(raw.get("title"))
    content = normalize_space(raw.get("content") or raw.get("text") or raw.get("paragraph_text"))
    source = normalize_space(raw.get("source"))
    doc_hash = normalize_space(raw.get("doc_hash") or f"musique_{idx}")
    if title_prefix_content and title and not content.lower().startswith(title.lower()):
        prompt_content = f"{title}. {content}"
    else:
        prompt_content = content
    return {
        "id": f"musique_para_{source}_{doc_hash}",
        "content": prompt_content,
        "title": title,
        "metadata": {
            "title": title,
            "dataset": "musique",
            "source": source,
            "chunk_type": "paragraph",
            "doc_hash": doc_hash,
        },
    }


def transform_retained_mixed_doc(raw: Dict[str, Any], idx: int) -> Dict[str, Any]:
    title = normalize_space(raw.get("title") or (raw.get("metadata") or {}).get("title"))
    content = normalize_space(raw.get("content") or raw.get("contents") or raw.get("text"))
    metadata = dict(raw.get("metadata") or {})
    metadata.setdefault("title", title)
    metadata.setdefault("corpus_stage", "mixed_retained")
    return {
        "id": str(raw.get("id") or f"doc_{idx}"),
        "content": content,
        "title": title,
        "metadata": metadata,
    }


def parse_splits(raw: str) -> List[str]:
    splits = [item.strip() for item in raw.split(",") if item.strip()]
    if not splits:
        raise argparse.ArgumentTypeError("split list must not be empty")
    return splits


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a 1.78M-scale mixed FAISS index by replacing old MuSiQue "
            "support-sentence chunks with MuSiQue paragraph chunks."
        )
    )
    parser.add_argument("--mixed-index", type=Path, default=DEFAULT_MIXED_INDEX, required=DEFAULT_MIXED_INDEX is None)
    parser.add_argument("--mixed-meta", type=Path, default=DEFAULT_MIXED_META, required=DEFAULT_MIXED_META is None)
    parser.add_argument("--musique-index", type=Path, default=DEFAULT_MUSIQUE_INDEX, required=DEFAULT_MUSIQUE_INDEX is None)
    parser.add_argument("--musique-meta", type=Path, default=DEFAULT_MUSIQUE_META, required=DEFAULT_MUSIQUE_META is None)
    parser.add_argument("--musique-arrow-dir", type=Path, default=DEFAULT_MUSIQUE_ARROW_DIR, required=DEFAULT_MUSIQUE_ARROW_DIR is None)
    parser.add_argument("--wiki-train-json", type=Path, default=DEFAULT_2WIKI_TRAIN, required=DEFAULT_2WIKI_TRAIN is None)
    parser.add_argument("--hotpot-dev-json", type=Path, default=DEFAULT_HOTPOT_DEV, required=DEFAULT_HOTPOT_DEV is None)
    parser.add_argument("--2wiki-dev-json", type=Path, default=DEFAULT_2WIKI_DEV, required=DEFAULT_2WIKI_DEV is None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-name", type=str, default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--target-total", type=int, default=None)
    parser.add_argument("--drop-seed", type=int, default=42)
    parser.add_argument("--old-musique-splits", type=parse_splits, default=parse_splits("dev,train"))
    parser.add_argument("--new-musique-splits", type=parse_splits, default=parse_splits("dev,train"))
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
        help="Disable the default protection of all HotpotQA/2Wiki dev support sentences while balancing.",
    )
    parser.add_argument("--reconstruct-batch-size", type=int, default=8192)
    parser.add_argument("--embed-batch-size", type=int, default=64)
    parser.add_argument("--bge-model", type=str, default=DEFAULT_BGE_LOCAL)
    parser.add_argument("--bge-fp32", action="store_true")
    parser.add_argument(
        "--reencode-musique-title-prefix",
        action="store_true",
        help="Encode MuSiQue paragraph vectors as '<title>. <paragraph>' instead of reusing the existing paragraph-only vectors.",
    )
    parser.add_argument(
        "--title-prefix-content",
        action="store_true",
        help="Also store '<title>. <paragraph>' as content shown to the reader. By default only the embedding text is title-prefixed.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only compute replacement/drop statistics; do not build the FAISS index.",
    )
    return parser


def select_musique_docs(
    docs: List[Dict[str, Any]],
    splits: Sequence[str],
    *,
    title_prefix_content: bool,
) -> tuple[List[int], List[Dict[str, Any]]]:
    wanted = set(splits)
    indices: List[int] = []
    transformed: List[Dict[str, Any]] = []
    seen: Set[DocKey] = set()
    for idx, doc in enumerate(docs):
        source = normalize_space(doc.get("source"))
        if source not in wanted:
            continue
        title = normalize_space(doc.get("title"))
        content = normalize_space(doc.get("content") or doc.get("text") or doc.get("paragraph_text"))
        key = doc_key(title, content)
        if not title or not content or key in seen:
            continue
        seen.add(key)
        indices.append(idx)
        transformed.append(transform_musique_doc(doc, idx, title_prefix_content=title_prefix_content))
    return indices, transformed


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

    musique_indices, new_musique_docs = select_musique_docs(
        musique_docs,
        args.new_musique_splits,
        title_prefix_content=args.title_prefix_content,
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
    protected_keys.update(protected_eval_sentence_keys(
        args.protect_eval_set,
        hotpot_dev=args.hotpot_dev_json,
        wiki_dev=args.__dict__["2wiki_dev_json"],
    ))
    old_musique_set = set(old_musique_indices)
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

    rng = random.Random(args.drop_seed)
    dropped_wiki_train_indices = set(rng.sample(droppable_wiki_train_indices, drop_needed)) if drop_needed else set()
    excluded_old_indices = old_musique_set | dropped_wiki_train_indices
    final_total = len(mixed_docs) - len(excluded_old_indices) + len(new_musique_docs)

    return {
        "mixed_docs": mixed_docs,
        "musique_source_docs": musique_docs,
        "musique_source_indices": musique_indices,
        "new_musique_docs": new_musique_docs,
        "old_musique_indices": old_musique_indices,
        "dropped_wiki_train_indices": dropped_wiki_train_indices,
        "excluded_old_indices": excluded_old_indices,
        "stats": {
            "old_mixed_total": len(mixed_docs),
            "target_total": target_total,
            "old_musique_sentence_keys": len(old_musique_keys),
            "old_musique_docs_removed": len(old_musique_indices),
            "new_musique_paragraph_docs": len(new_musique_docs),
            "total_before_balancing": total_before_balancing,
            "drop_needed_from_2wiki_train": drop_needed,
            "available_2wiki_train_docs": len(wiki_train_indices),
            "protected_eval_gold_sentence_keys": len(protected_keys),
            "protect_full_dev_support": not args.no_protect_full_dev_support,
            "protected_2wiki_train_docs": len(protected_wiki_train_indices),
            "droppable_2wiki_train_docs": len(droppable_wiki_train_indices),
            "dropped_2wiki_train_docs": len(dropped_wiki_train_indices),
            "final_total": final_total,
            "new_musique_splits": list(args.new_musique_splits),
            "old_musique_splits": list(args.old_musique_splits),
            "drop_seed": args.drop_seed,
            "reencode_musique_title_prefix": bool(args.reencode_musique_title_prefix),
            "title_prefix_content": bool(args.title_prefix_content),
        },
    }


def add_old_vectors(
    *,
    faiss: Any,
    output_index: Any,
    old_index: Any,
    mixed_docs: List[Dict[str, Any]],
    excluded_old_indices: Set[int],
    output_metadata: List[Dict[str, Any]],
    batch_size: int,
) -> None:
    total = len(mixed_docs)
    for start in range(0, total, batch_size):
        count = min(batch_size, total - start)
        vectors = old_index.reconstruct_n(start, count)
        keep_offsets = [
            offset
            for offset in range(count)
            if start + offset not in excluded_old_indices
        ]
        if keep_offsets:
            selected = vectors[keep_offsets].astype("float32", copy=False)
            faiss.normalize_L2(selected)
            output_index.add(selected)
            output_metadata.extend(
                transform_retained_mixed_doc(mixed_docs[start + offset], start + offset)
                for offset in keep_offsets
            )
        print(
            json.dumps(
                {
                    "event": "added_old_vectors",
                    "processed": start + count,
                    "output_index_size": int(output_index.ntotal),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )


def add_musique_vectors_from_existing(
    *,
    faiss: Any,
    output_index: Any,
    musique_index: Any,
    source_indices: List[int],
    new_musique_docs: List[Dict[str, Any]],
    output_metadata: List[Dict[str, Any]],
    batch_size: int,
) -> None:
    for start in range(0, len(source_indices), batch_size):
        indices = source_indices[start : start + batch_size]
        vectors = [musique_index.reconstruct(int(idx)) for idx in indices]
        import numpy as np

        matrix = np.asarray(vectors, dtype="float32")
        faiss.normalize_L2(matrix)
        output_index.add(matrix)
        output_metadata.extend(new_musique_docs[start : start + len(indices)])
        print(
            json.dumps(
                {
                    "event": "added_musique_existing_vectors",
                    "processed": start + len(indices),
                    "output_index_size": int(output_index.ntotal),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )


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
        texts = [
            normalize_space(f"{doc.get('title', '')}. {doc.get('content', '')}")
            for doc in batch
        ]
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
                    "event": "encoded_musique_title_prefix_vectors",
                    "processed": start + len(batch),
                    "output_index_size": int(output_index.ntotal),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )


def build_index(args: argparse.Namespace, plan: Dict[str, Any]) -> None:
    import faiss

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / f"{args.output_name}.index"
    meta_path = output_dir / f"{args.output_name}.meta"
    info_path = output_dir / f"{args.output_name}_info.json"

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

    if args.reencode_musique_title_prefix:
        add_musique_vectors_by_encoding(
            faiss=faiss,
            output_index=output_index,
            new_musique_docs=plan["new_musique_docs"],
            output_metadata=output_metadata,
            bge_model=args.bge_model,
            use_fp16=not args.bge_fp32,
            batch_size=args.embed_batch_size,
        )
    else:
        musique_index = faiss.read_index(str(args.musique_index))
        if musique_index.d != dim:
            raise ValueError(f"dimension mismatch: mixed={dim}, musique={musique_index.d}")
        add_musique_vectors_from_existing(
            faiss=faiss,
            output_index=output_index,
            musique_index=musique_index,
            source_indices=plan["musique_source_indices"],
            new_musique_docs=plan["new_musique_docs"],
            output_metadata=output_metadata,
            batch_size=args.reconstruct_batch_size,
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
            "musique_embedding_mode": (
                "title_prefix_reencoded"
                if args.reencode_musique_title_prefix
                else "existing_musique_paragraph_vectors"
            ),
            "notes": [
                "Old MuSiQue support-sentence chunks are removed.",
                "MuSiQue replacement uses all selected split paragraphs, not is_supporting labels.",
                "MuSiQue is_supporting metadata is intentionally not written to the merged metadata.",
                "2Wiki train documents are downsampled by fixed seed only to keep the target corpus size unchanged.",
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
