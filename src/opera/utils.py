from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional


def force_gpu0(gpu_id: int = 0) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def extract_json(text: str) -> Any:
    """Extract the first valid JSON object or array from an LLM response."""
    if not text:
        raise ValueError("empty response")

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    starts = sorted([m.start() for m in re.finditer(r"[\[{]", cleaned)])
    candidates: List[tuple[int, int, Any]] = []
    last_error: Optional[Exception] = None
    for start in starts:
        try:
            value, _ = decoder.raw_decode(cleaned[start:])
            if isinstance(value, dict):
                rank = 0
            elif isinstance(value, list) and any(isinstance(item, (dict, list)) for item in value):
                rank = 0
            elif isinstance(value, list):
                rank = 1
            else:
                rank = 2
            candidates.append((rank, start, value))
        except json.JSONDecodeError as exc:
            last_error = exc
    if candidates:
        candidates.sort(key=lambda item: (item[0], item[1]))
        return candidates[0][2]
    raise ValueError(f"no valid JSON found: {last_error}")


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
