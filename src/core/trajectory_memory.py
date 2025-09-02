"""
Trajectory Memory Component - Records and manages reasoning trajectories
Enhances interpretability by providing clear rationale for each action
"""

from typing import Dict, Any, List, Optional
import json
import logging
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from ..data.structures import ReasoningState, StrategicPlan

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryEvent:
    """Single event in the reasoning trajectory"""
    timestamp: str
    event_type: str
    component: str
    details: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


class TrajectoryMemoryComponent:
    """
    Records complete reasoning trajectories for interpretability
    Provides clear rationale for each action taken by agents
    """
    
    def __init__(self, save_dir: Optional[str] = None):
        """
        Initialize trajectory memory
        
        Args:
            save_dir: Directory to save trajectory logs
        """
        self.save_dir = Path(save_dir) if save_dir else None
        self.current_trajectory: List[TrajectoryEvent] = []
        self.trajectory_id = self._generate_trajectory_id()
        
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def _generate_trajectory_id(self) -> str:
        """Generate unique trajectory ID"""
        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    
    def record_event(
        self,
        event_type: str,
        component: str,
        details: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Record a single event in the trajectory
        
        Args:
            event_type: Type of event (e.g., 'planning', 'retrieval', 'analysis')
            component: Component that generated the event
            details: Event details
            metadata: Optional metadata
        """
        event = TrajectoryEvent(
            timestamp=datetime.now().isoformat(),
            event_type=event_type,
            component=component,
            details=details,
            metadata=metadata or {}
        )
        
        self.current_trajectory.append(event)
        logger.debug(f"Recorded {event_type} event from {component}")
    
    def record_planning_decision(self, question: str, plan: StrategicPlan):
        """
        Record planning phase decision
        
        Args:
            question: Original question
            plan: Generated strategic plan
        """
        self.record_event(
            event_type="planning",
            component="PlanAgent",
            details={
                "question": question,
                "plan": plan.to_dict(),
                "reasoning": plan.reasoning,
                "num_steps": len(plan.sub_questions),
                "confidence": plan.confidence_score
            },
            metadata={
                "rationale": "Strategic decomposition to identify information needs",
                "decision_factors": [
                    "Question complexity",
                    "Information dependencies",
                    "Expected answer type"
                ]
            }
        )
    
    def record_retrieval_request(self, step_id: int, query: str, parameters: Dict[str, Any]):
        """
        Record retrieval request
        
        Args:
            step_id: Step ID
            query: Retrieval query
            parameters: Retrieval parameters
        """
        self.record_event(
            event_type="retrieval_request",
            component="RetrievalCoordinator",
            details={
                "step_id": step_id,
                "query": query,
                "parameters": parameters
            },
            metadata={
                "rationale": "Fetching relevant documents for information extraction"
            }
        )
    
    def record_retrieval_response(self, step_id: int, num_docs: int, top_scores: List[float]):
        """
        Record retrieval response
        
        Args:
            step_id: Step ID
            num_docs: Number of documents retrieved
            top_scores: Top document scores
        """
        self.record_event(
            event_type="retrieval_response",
            component="RetrievalCoordinator",
            details={
                "step_id": step_id,
                "num_documents": num_docs,
                "top_scores": top_scores
            },
            metadata={
                "quality_indicators": {
                    "high_confidence": top_scores[0] > 0.8 if top_scores else False,
                    "score_gap": top_scores[0] - top_scores[-1] if len(top_scores) > 1 else 0
                }
            }
        )
    
    def record_analysis_decision(
        self,
        step_id: int,
        sub_question: str,
        status: str,
        answer: Optional[str],
        analysis: str,
        supporting_docs: List[str]
    ):
        """
        Record analysis decision from Analysis-Answer Agent
        
        Args:
            step_id: Step ID
            sub_question: Sub-question being analyzed
            status: Information sufficiency status
            answer: Extracted answer (if sufficient)
            analysis: Analysis reasoning
            supporting_docs: Supporting document IDs
        """
        self.record_event(
            event_type="analysis_decision",
            component="AnalysisAnswerAgent",
            details={
                "step_id": step_id,
                "sub_question": sub_question,
                "status": status,
                "answer": answer,
                "analysis": analysis,
                "supporting_docs": supporting_docs
            },
            metadata={
                "rationale": f"Information {'sufficient' if status == 'yes' else 'insufficient'} for answering",
                "decision_basis": "Document content analysis and relevance assessment"
            }
        )
    
    def record_rewrite_attempt(
        self,
        step_id: int,
        original_query: str,
        rewritten_query: str,
        strategy: str
    ):
        """
        Record query rewrite attempt
        
        Args:
            step_id: Step ID
            original_query: Original query
            rewritten_query: Rewritten query
            strategy: Rewrite strategy used
        """
        self.record_event(
            event_type="query_rewrite",
            component="RewriteAgent",
            details={
                "step_id": step_id,
                "original_query": original_query,
                "rewritten_query": rewritten_query,
                "strategy": strategy
            },
            metadata={
                "rationale": "Query reformulation to improve retrieval effectiveness",
                "strategy_explanation": {
                    "keyword_expansion": "Adding related terms to broaden search",
                    "entity_focus": "Emphasizing specific entities in query",
                    "temporal_focus": "Adding time-related constraints",
                    "relation_focus": "Highlighting relationships between entities"
                }.get(strategy, "Custom strategy")
            }
        )
    
    def record_execution_success(self, step_id: int, answer: str, supporting_docs: List[str]):
        """
        Record successful step execution
        
        Args:
            step_id: Step ID
            answer: Final answer for the step
            supporting_docs: Documents that supported the answer
        """
        self.record_event(
            event_type="execution_success",
            component="Orchestrator",
            details={
                "step_id": step_id,
                "answer": answer,
                "supporting_docs": supporting_docs
            },
            metadata={
                "outcome": "success",
                "rationale": "Successfully extracted required information"
            }
        )
    
    def record_execution_failure(self, step_id: int, reason: str):
        """
        Record failed step execution
        
        Args:
            step_id: Step ID
            reason: Failure reason
        """
        self.record_event(
            event_type="execution_failure",
            component="Orchestrator",
            details={
                "step_id": step_id,
                "reason": reason
            },
            metadata={
                "outcome": "failure",
                "rationale": "Unable to find required information despite retries"
            }
        )
    
    def record_final_answer(self, answer: str, confidence: float, reasoning_summary: str):
        """
        Record final answer formulation
        
        Args:
            answer: Final answer
            confidence: Confidence score
            reasoning_summary: Summary of reasoning process
        """
        self.record_event(
            event_type="final_answer",
            component="Orchestrator",
            details={
                "answer": answer,
                "confidence": confidence,
                "reasoning_summary": reasoning_summary
            },
            metadata={
                "rationale": "Synthesized final answer from collected information",
                "quality_factors": [
                    "Completeness of information",
                    "Consistency of facts",
                    "Confidence in sources"
                ]
            }
        )
    
    def record_complete_trajectory(self, state: ReasoningState):
        """
        Record complete trajectory from final state
        
        Args:
            state: Final reasoning state
        """
        self.record_event(
            event_type="trajectory_complete",
            component="System",
            details={
                "question": state.question,
                "final_answer": state.final_answer,
                "confidence": state.confidence_score,
                "total_steps": len(state.strategic_plan.sub_questions) if state.strategic_plan else 0,
                "completed_steps": len(state.get_completed_steps()),
                "failures": len(state.failure_context)
            },
            metadata={
                "summary": "Complete reasoning trajectory recorded"
            }
        )
        
        # Save trajectory if save_dir is configured
        if self.save_dir:
            self._save_trajectory()
    
    def get_complete_trajectory(self) -> List[Dict[str, Any]]:
        """
        Get complete trajectory as list of dictionaries
        
        Returns:
            List of trajectory events
        """
        return [
            {
                "timestamp": event.timestamp,
                "event_type": event.event_type,
                "component": event.component,
                "details": event.details,
                "metadata": event.metadata
            }
            for event in self.current_trajectory
        ]
    
    def get_trajectory_summary(self) -> Dict[str, Any]:
        """
        Get summary of current trajectory
        
        Returns:
            Trajectory summary
        """
        event_counts = {}
        for event in self.current_trajectory:
            event_counts[event.event_type] = event_counts.get(event.event_type, 0) + 1
        
        return {
            "trajectory_id": self.trajectory_id,
            "total_events": len(self.current_trajectory),
            "event_counts": event_counts,
            "start_time": self.current_trajectory[0].timestamp if self.current_trajectory else None,
            "end_time": self.current_trajectory[-1].timestamp if self.current_trajectory else None
        }
    
    def _save_trajectory(self):
        """Save trajectory to disk"""
        if not self.save_dir:
            return
        
        trajectory_file = self.save_dir / f"trajectory_{self.trajectory_id}.json"
        
        try:
            trajectory_data = {
                "trajectory_id": self.trajectory_id,
                "events": self.get_complete_trajectory(),
                "summary": self.get_trajectory_summary()
            }
            
            with open(trajectory_file, 'w') as f:
                json.dump(trajectory_data, f, indent=2)
            
            logger.info(f"Saved trajectory to {trajectory_file}")
        except Exception as e:
            logger.error(f"Failed to save trajectory: {str(e)}")
    
    def clear_trajectory(self):
        """Clear current trajectory and start fresh"""
        self.current_trajectory = []
        self.trajectory_id = self._generate_trajectory_id()
        logger.info("Cleared trajectory memory")