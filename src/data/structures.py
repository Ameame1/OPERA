"""
Core data structures for OPERA-MAPGRPO system
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from enum import Enum
import json
from datetime import datetime


class InformationStatus(Enum):
    """Status of information sufficiency"""
    SUFFICIENT = "yes"
    INSUFFICIENT = "no"


class ExecutionStatus(Enum):
    """Status of execution steps"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PlanStep:
    """Single step in a strategic plan with placeholder support"""
    step_id: int
    sub_question: str
    goal: str
    dependencies: List[int] = field(default_factory=list)
    expected_info_type: str = "entity"
    placeholders: Dict[str, int] = field(default_factory=dict)  # Maps placeholder text to dependency step
    result: Optional[str] = None
    confidence: float = 0.0
    
    def has_dependencies(self) -> bool:
        """Check if this step has dependencies"""
        return len(self.dependencies) > 0
    
    def fill_placeholders(self, previous_results: Dict[int, str]) -> str:
        """Fill placeholders with results from previous steps"""
        filled_question = self.sub_question
        for placeholder, dep_id in self.placeholders.items():
            if dep_id in previous_results:
                filled_question = filled_question.replace(
                    f"[{placeholder} from step {dep_id}]",
                    previous_results[dep_id]
                )
        return filled_question
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "step_id": self.step_id,
            "sub_question": self.sub_question,
            "goal": self.goal,
            "dependencies": self.dependencies,
            "expected_info_type": self.expected_info_type,
            "placeholders": self.placeholders,
            "result": self.result,
            "confidence": self.confidence
        }


@dataclass
class StrategicPlan:
    """Complete strategic plan for a complex question"""
    original_question: str
    sub_questions: List[PlanStep]
    reasoning: str = ""
    plan_version: int = 1
    confidence_score: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def get_step(self, step_id: int) -> Optional[PlanStep]:
        """Get a specific step by ID"""
        for step in self.sub_questions:
            if step.step_id == step_id:
                return step
        return None
    
    def get_executable_steps(self, completed_steps: List[int]) -> List[PlanStep]:
        """Get steps that can be executed (all dependencies satisfied)"""
        executable = []
        for step in self.sub_questions:
            if step.step_id not in completed_steps:
                if all(dep in completed_steps for dep in step.dependencies):
                    executable.append(step)
        return executable
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "original_question": self.original_question,
            "sub_questions": [step.to_dict() for step in self.sub_questions],
            "reasoning": self.reasoning,
            "plan_version": self.plan_version,
            "confidence_score": self.confidence_score,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategicPlan':
        """Create from dictionary representation"""
        steps = [
            PlanStep(
                step_id=s["step_id"],
                sub_question=s["sub_question"],
                goal=s["goal"],
                dependencies=s.get("dependencies", []),
                expected_info_type=s.get("expected_info_type", "entity"),
                placeholders=s.get("placeholders", {}),
                result=s.get("result"),
                confidence=s.get("confidence", 0.0)
            )
            for s in data["sub_questions"]
        ]
        
        return cls(
            original_question=data["original_question"],
            sub_questions=steps,
            reasoning=data.get("reasoning", ""),
            plan_version=data.get("plan_version", 1),
            confidence_score=data.get("confidence_score", 0.0),
            timestamp=data.get("timestamp", datetime.now().isoformat())
        )


@dataclass
class Document:
    """Retrieved document with metadata"""
    doc_id: str
    content: str
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "doc_id": self.doc_id,
            "content": self.content,
            "score": self.score,
            "metadata": self.metadata
        }


@dataclass
class RetrievalParameters:
    """Parameters for adaptive retrieval"""
    query: str
    top_k: int = 5
    search_type: str = "dense"
    filters: Dict[str, Any] = field(default_factory=dict)
    rerank: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "query": self.query,
            "top_k": self.top_k,
            "search_type": self.search_type,
            "filters": self.filters,
            "rerank": self.rerank
        }


@dataclass
class AnalysisResult:
    """Result from Analysis-Answer Agent"""
    status: InformationStatus
    answer: Optional[str] = None
    analysis: str = ""
    confidence: float = 0.0
    supporting_docs: List[str] = field(default_factory=list)
    
    def is_sufficient(self) -> bool:
        """Check if information is sufficient"""
        return self.status == InformationStatus.SUFFICIENT
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "status": self.status.value,
            "answer": self.answer,
            "analysis": self.analysis,
            "confidence": self.confidence,
            "supporting_docs": self.supporting_docs
        }


@dataclass
class RewriteResult:
    """Result from Rewrite Agent"""
    rewritten_query: str
    strategy: str = "keyword_expansion"
    keywords: List[str] = field(default_factory=list)
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "rewritten_query": self.rewritten_query,
            "strategy": self.strategy,
            "keywords": self.keywords,
            "confidence": self.confidence
        }


@dataclass
class ExecutionTrace:
    """Single execution trace in the system"""
    step_id: int
    sub_question: str
    filled_question: str
    retrieval_params: RetrievalParameters
    retrieved_docs: List[Document]
    analysis_result: Optional[AnalysisResult] = None
    rewrite_result: Optional[RewriteResult] = None
    status: ExecutionStatus = ExecutionStatus.PENDING
    attempts: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "step_id": self.step_id,
            "sub_question": self.sub_question,
            "filled_question": self.filled_question,
            "retrieval_params": self.retrieval_params.to_dict(),
            "retrieved_docs": [doc.to_dict() for doc in self.retrieved_docs],
            "analysis_result": self.analysis_result.to_dict() if self.analysis_result else None,
            "rewrite_result": self.rewrite_result.to_dict() if self.rewrite_result else None,
            "status": self.status.value,
            "attempts": self.attempts,
            "timestamp": self.timestamp
        }


@dataclass
class ReasoningState:
    """Complete reasoning state for the OPERA system"""
    question: str
    strategic_plan: Optional[StrategicPlan] = None
    current_step_index: int = 0
    collected_facts: Dict[int, str] = field(default_factory=dict)
    execution_traces: List[ExecutionTrace] = field(default_factory=list)
    failure_context: List[str] = field(default_factory=list)
    final_answer: Optional[str] = None
    confidence_score: float = 0.0
    
    def add_fact(self, step_id: int, fact: str, source_doc: str):
        """Add a collected fact with metadata"""
        self.collected_facts[step_id] = {
            "content": fact,
            "source_document": source_doc,
            "timestamp": datetime.now().isoformat()
        }
    
    def add_execution_trace(self, trace: ExecutionTrace):
        """Add an execution trace"""
        self.execution_traces.append(trace)
    
    def add_failure(self, context: str):
        """Add a failure context"""
        self.failure_context.append(f"[{datetime.now().isoformat()}] {context}")
    
    def get_completed_steps(self) -> List[int]:
        """Get list of completed step IDs"""
        return [
            trace.step_id 
            for trace in self.execution_traces 
            if trace.status == ExecutionStatus.COMPLETED
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "question": self.question,
            "strategic_plan": self.strategic_plan.to_dict() if self.strategic_plan else None,
            "current_step_index": self.current_step_index,
            "collected_facts": self.collected_facts,
            "execution_traces": [trace.to_dict() for trace in self.execution_traces],
            "failure_context": self.failure_context,
            "final_answer": self.final_answer,
            "confidence_score": self.confidence_score
        }


@dataclass
class EvaluationMetrics:
    """Evaluation metrics for system performance"""
    exact_match: float = 0.0
    f1_score: float = 0.0
    accuracy: float = 0.0
    retrieval_ndcg: float = 0.0
    planning_quality: float = 0.0
    execution_efficiency: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "exact_match": self.exact_match,
            "f1_score": self.f1_score,
            "accuracy": self.accuracy,
            "retrieval_ndcg": self.retrieval_ndcg,
            "planning_quality": self.planning_quality,
            "execution_efficiency": self.execution_efficiency
        }