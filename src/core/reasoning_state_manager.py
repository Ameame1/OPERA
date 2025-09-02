"""
Reasoning State Manager - Manages the state throughout the OPERA pipeline
"""

from typing import Dict, Any, Optional, List
import json
import logging
from datetime import datetime
from pathlib import Path
from ..data.structures import ReasoningState, StrategicPlan, ExecutionTrace

logger = logging.getLogger(__name__)


class ReasoningStateManager:
    """
    Manages reasoning state throughout the OPERA system execution
    Provides persistence and state tracking capabilities
    """
    
    def __init__(self, checkpoint_dir: Optional[str] = None):
        """
        Initialize state manager
        
        Args:
            checkpoint_dir: Directory for saving state checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.current_state: Optional[ReasoningState] = None
        self.state_history: List[ReasoningState] = []
        
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def create_state(self, question: str) -> ReasoningState:
        """
        Create a new reasoning state
        
        Args:
            question: Initial question
            
        Returns:
            New ReasoningState instance
        """
        state = ReasoningState(question=question)
        self.current_state = state
        self.state_history.append(state)
        
        logger.info(f"Created new reasoning state for question: {question[:50]}...")
        return state
    
    def update_state(self, state: ReasoningState) -> ReasoningState:
        """
        Update the current state
        
        Args:
            state: Updated state
            
        Returns:
            Updated state
        """
        self.current_state = state
        
        # Save checkpoint if enabled
        if self.checkpoint_dir:
            self._save_checkpoint(state)
        
        return state
    
    def get_current_state(self) -> Optional[ReasoningState]:
        """
        Get the current reasoning state
        
        Returns:
            Current state or None
        """
        return self.current_state
    
    def add_plan(self, plan: StrategicPlan) -> ReasoningState:
        """
        Add strategic plan to current state
        
        Args:
            plan: Strategic plan
            
        Returns:
            Updated state
        """
        if not self.current_state:
            raise ValueError("No current state exists")
        
        self.current_state.strategic_plan = plan
        logger.info(f"Added strategic plan with {len(plan.sub_questions)} steps")
        
        return self.update_state(self.current_state)
    
    def add_execution_trace(self, trace: ExecutionTrace) -> ReasoningState:
        """
        Add execution trace to current state
        
        Args:
            trace: Execution trace
            
        Returns:
            Updated state
        """
        if not self.current_state:
            raise ValueError("No current state exists")
        
        self.current_state.add_execution_trace(trace)
        logger.info(f"Added execution trace for step {trace.step_id}")
        
        return self.update_state(self.current_state)
    
    def add_collected_fact(self, step_id: int, fact: str, source_doc: str) -> ReasoningState:
        """
        Add a collected fact to current state
        
        Args:
            step_id: Step ID
            fact: Collected fact
            source_doc: Source document ID
            
        Returns:
            Updated state
        """
        if not self.current_state:
            raise ValueError("No current state exists")
        
        self.current_state.add_fact(step_id, fact, source_doc)
        logger.info(f"Added fact for step {step_id}: {fact[:50]}...")
        
        return self.update_state(self.current_state)
    
    def add_failure_context(self, context: str) -> ReasoningState:
        """
        Add failure context to current state
        
        Args:
            context: Failure context description
            
        Returns:
            Updated state
        """
        if not self.current_state:
            raise ValueError("No current state exists")
        
        self.current_state.add_failure(context)
        logger.warning(f"Added failure context: {context}")
        
        return self.update_state(self.current_state)
    
    def set_final_answer(self, answer: str, confidence: float) -> ReasoningState:
        """
        Set final answer in current state
        
        Args:
            answer: Final answer
            confidence: Confidence score
            
        Returns:
            Updated state
        """
        if not self.current_state:
            raise ValueError("No current state exists")
        
        self.current_state.final_answer = answer
        self.current_state.confidence_score = confidence
        logger.info(f"Set final answer with confidence {confidence}: {answer[:50]}...")
        
        return self.update_state(self.current_state)
    
    def get_completed_steps(self) -> List[int]:
        """
        Get list of completed step IDs
        
        Returns:
            List of completed step IDs
        """
        if not self.current_state:
            return []
        
        return self.current_state.get_completed_steps()
    
    def get_executable_steps(self) -> List[Any]:
        """
        Get steps that can be executed
        
        Returns:
            List of executable steps
        """
        if not self.current_state or not self.current_state.strategic_plan:
            return []
        
        completed = self.get_completed_steps()
        return self.current_state.strategic_plan.get_executable_steps(completed)
    
    def _save_checkpoint(self, state: ReasoningState):
        """
        Save state checkpoint to disk
        
        Args:
            state: State to save
        """
        if not self.checkpoint_dir:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_file = self.checkpoint_dir / f"state_checkpoint_{timestamp}.json"
        
        try:
            with open(checkpoint_file, 'w') as f:
                json.dump(state.to_dict(), f, indent=2)
            logger.debug(f"Saved checkpoint to {checkpoint_file}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {str(e)}")
    
    def load_checkpoint(self, checkpoint_file: str) -> ReasoningState:
        """
        Load state from checkpoint
        
        Args:
            checkpoint_file: Path to checkpoint file
            
        Returns:
            Loaded state
        """
        with open(checkpoint_file, 'r') as f:
            state_dict = json.load(f)
        
        # Reconstruct state from dict
        # This is a simplified version - full implementation would properly
        # reconstruct all nested objects
        state = ReasoningState(question=state_dict['question'])
        
        if state_dict.get('strategic_plan'):
            state.strategic_plan = StrategicPlan.from_dict(state_dict['strategic_plan'])
        
        state.current_step_index = state_dict.get('current_step_index', 0)
        state.collected_facts = state_dict.get('collected_facts', {})
        state.failure_context = state_dict.get('failure_context', [])
        state.final_answer = state_dict.get('final_answer')
        state.confidence_score = state_dict.get('confidence_score', 0.0)
        
        self.current_state = state
        logger.info(f"Loaded checkpoint from {checkpoint_file}")
        
        return state
    
    def get_state_summary(self) -> Dict[str, Any]:
        """
        Get summary of current state
        
        Returns:
            State summary dictionary
        """
        if not self.current_state:
            return {"status": "No active state"}
        
        state = self.current_state
        completed_steps = self.get_completed_steps()
        
        summary = {
            "question": state.question,
            "has_plan": state.strategic_plan is not None,
            "total_steps": len(state.strategic_plan.sub_questions) if state.strategic_plan else 0,
            "completed_steps": len(completed_steps),
            "collected_facts": len(state.collected_facts),
            "failures": len(state.failure_context),
            "has_final_answer": state.final_answer is not None,
            "confidence": state.confidence_score
        }
        
        return summary