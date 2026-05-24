from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Document:
    doc_id: str
    title: str
    content: str
    score: float
    rank: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, max_chars: Optional[int] = None) -> Dict[str, Any]:
        data = asdict(self)
        if max_chars and len(data["content"]) > max_chars:
            data["content"] = data["content"][:max_chars] + "..."
        return data


@dataclass
class PlanStep:
    step_id: int
    subgoal: str
    dependencies: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AnalysisResult:
    status: str
    answer: str = ""
    analysis: str = ""
    confidence: Optional[float] = None
    raw_response: str = ""

    def is_sufficient(self) -> bool:
        return self.status.lower().strip() == "yes" and bool(self.answer.strip())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RewriteResult:
    rewritten_query: str
    strategy: str = ""
    keywords: List[str] = field(default_factory=list)
    raw_response: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
