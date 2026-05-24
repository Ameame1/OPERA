#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def clean_aliases(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def convert_hotpot(row: Dict[str, Any]) -> Dict[str, Any]:
    original_id = str(row.get("_id") or row.get("id"))
    return {
        "id": f"hotpotqa_{original_id}",
        "question": str(row.get("question", "")),
        "answer": str(row.get("answer", "")),
        "answer_aliases": [],
        "source_dataset": "hotpotqa",
        "original_id": original_id,
        "question_type": row.get("type"),
        "difficulty": row.get("level"),
    }


def convert_2wiki(row: Dict[str, Any]) -> Dict[str, Any]:
    original_id = str(row.get("_id") or row.get("id"))
    return {
        "id": f"2wiki_{original_id}",
        "question": str(row.get("question", "")),
        "answer": str(row.get("answer", "")),
        "answer_aliases": [],
        "source_dataset": "2wikimultihopqa",
        "original_id": original_id,
        "question_type": row.get("type"),
        "difficulty": "unknown",
    }


def musique_hop(row: Dict[str, Any]) -> str:
    match = re.match(r"([234])hop", str(row.get("id", "")))
    if not match:
        raise ValueError(f"Cannot infer MuSiQue hop from id: {row.get('id')}")
    return match.group(1)


def convert_musique(row: Dict[str, Any]) -> Dict[str, Any]:
    hop = musique_hop(row)
    return {
        "id": str(row.get("id")),
        "question": str(row.get("question", "")),
        "answer": str(row.get("answer", "")),
        "answer_aliases": clean_aliases(row.get("answer_aliases")),
        "source_dataset": "musique",
        "original_id": str(row.get("id")),
        "question_type": f"{hop}hop",
        "difficulty": "unknown",
        "hop_count": int(hop),
        "answerable": row.get("answerable"),
    }


def sample_rows(rows: List[Dict[str, Any]], *, n: int, rng: random.Random, name: str) -> List[Dict[str, Any]]:
    if len(rows) < n:
        raise ValueError(f"{name}: need {n} rows but only found {len(rows)}")
    return rng.sample(rows, n)


def default_output_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "eval_sets" / "sample"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build reproducible OPERA evaluation splits from official HotpotQA, "
            "2WikiMultiHopQA, and MuSiQue dev files."
        )
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--hotpot-dev", type=Path, required=True)
    parser.add_argument("--2wiki-dev", type=Path, required=True)
    parser.add_argument("--musique-dev", type=Path, required=True)
    parser.add_argument("--n-per-dataset", type=int, default=500)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = args.output_dir or default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    hotpot_raw = read_json(args.hotpot_dev)
    wiki_raw = read_json(args.__dict__["2wiki_dev"])
    musique_raw = read_jsonl(args.musique_dev)

    hotpot_rows = [
        convert_hotpot(row)
        for row in sample_rows(hotpot_raw, n=args.n_per_dataset, rng=rng, name="hotpotqa")
    ]
    wiki_rows = [
        convert_2wiki(row)
        for row in sample_rows(wiki_raw, n=args.n_per_dataset, rng=rng, name="2wiki")
    ]

    musique_selected = sample_rows(musique_raw, n=args.n_per_dataset, rng=rng, name="musique")
    musique_rows = [convert_musique(row) for row in musique_selected]

    outputs = {
        "hotpotqa": output_dir / f"hotpotqa_{args.n_per_dataset}.jsonl",
        "2wiki": output_dir / f"2wiki_{args.n_per_dataset}.jsonl",
        "musique": output_dir / f"musique_{args.n_per_dataset}.jsonl",
    }
    counts = {
        "hotpotqa": write_jsonl(outputs["hotpotqa"], hotpot_rows),
        "2wiki": write_jsonl(outputs["2wiki"], wiki_rows),
        "musique": write_jsonl(outputs["musique"], musique_rows),
    }
    manifest = {
        "seed": args.seed,
        "description": "Seeded evaluation split sampled from official dev sets.",
        "sources": {
            "hotpotqa": str(args.hotpot_dev),
            "2wiki": str(args.__dict__["2wiki_dev"]),
            "musique": str(args.musique_dev),
        },
        "files": {name: str(path.name) for name, path in outputs.items()},
        "counts": counts,
        "musique_hop_counts": dict(Counter(row["hop_count"] for row in musique_rows)),
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(json.dumps({**manifest, "output_dir": str(output_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
