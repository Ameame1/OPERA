"""
Orchestrator - Central coordinator for OPERA-MAPGRPO workflow
Manages the interaction between GPM and REM modules
"""

from typing import Dict, Any, Optional, List
import logging
from ..data.structures import (
    ReasoningState, StrategicPlan, ExecutionTrace, 
    ExecutionStatus, InformationStatus, Document,
    RetrievalParameters
)
from ..agents import PlanAgent, AnalysisAnswerAgent, RewriteAgent
from .reasoning_state_manager import ReasoningStateManager
from .trajectory_memory import TrajectoryMemoryComponent
from ..utils.query_variant_generator import QueryVariantGenerator
from ..utils.placeholder_filler import SmartPlaceholderFiller

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Central orchestrator for OPERA system
    Coordinates the workflow between all components
    """
    
    def __init__(
        self,
        plan_agent: PlanAgent,
        analysis_answer_agent: AnalysisAnswerAgent,
        rewrite_agent: RewriteAgent,
        retrieval_coordinator: Any,  # Will be implemented separately
        state_manager: ReasoningStateManager,
        trajectory_memory: TrajectoryMemoryComponent,
        max_retries: int = 3
    ):
        """
        Initialize orchestrator with all components
        
        Args:
            plan_agent: Plan Agent for strategic decomposition
            analysis_answer_agent: Analysis-Answer Agent for execution
            rewrite_agent: Rewrite Agent for query reformulation
            retrieval_coordinator: Retrieval system interface
            state_manager: State management component
            trajectory_memory: Trajectory recording component
            max_retries: Maximum retries per sub-question
        """
        self.plan_agent = plan_agent
        self.analysis_answer_agent = analysis_answer_agent
        self.rewrite_agent = rewrite_agent
        self.retrieval_coordinator = retrieval_coordinator
        self.state_manager = state_manager
        self.trajectory_memory = trajectory_memory
        self.max_retries = max_retries
        
        # Framework optimization components
        self.query_generator = QueryVariantGenerator()
        self.placeholder_filler = SmartPlaceholderFiller()
        
    def process_question(self, question: str) -> Dict[str, Any]:
        """
        Process a complex question through the OPERA pipeline
        
        Args:
            question: Complex question to answer
            
        Returns:
            Dictionary with final answer and execution details
        """
        logger.info(f"Processing question: {question}")
        
        # Initialize reasoning state
        state = ReasoningState(question=question)
        
        try:
            # Phase 1: Strategic Planning (GPM)
            state = self._planning_phase(state)
            if not state.strategic_plan:
                return self._create_error_response("Failed to create strategic plan")
            
            # Phase 2: Tactical Execution (REM)
            state = self._execution_phase(state)
            
            # Phase 3: Answer Formulation
            state = self._formulation_phase(state)
            
            # Record final trajectory
            self.trajectory_memory.record_complete_trajectory(state)
            
            return self._create_success_response(state)
            
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            return self._create_error_response(str(e))
    
    def _planning_phase(self, state: ReasoningState) -> ReasoningState:
        """
        Execute planning phase using Plan Agent
        
        Args:
            state: Current reasoning state
            
        Returns:
            Updated state with strategic plan
        """
        logger.info("Executing planning phase...")
        
        # Generate strategic plan
        plan_result = self.plan_agent.process({'question': state.question})
        state.strategic_plan = plan_result['plan']
        
        # Record planning decision
        self.trajectory_memory.record_planning_decision(
            question=state.question,
            plan=state.strategic_plan
        )
        
        logger.info(f"Generated plan with {len(state.strategic_plan.sub_questions)} steps")
        return state
    
    def _execution_phase(self, state: ReasoningState) -> ReasoningState:
        """
        Execute tactical phase using Analysis-Answer and Rewrite Agents
        
        Args:
            state: Current reasoning state
            
        Returns:
            Updated state with execution results
        """
        logger.info("Executing tactical phase...")
        
        # Get executable steps
        while True:
            executable_steps = state.strategic_plan.get_executable_steps(
                state.get_completed_steps()
            )
            
            if not executable_steps:
                break  # All steps completed
            
            # Execute each available step
            for step in executable_steps:
                state = self._execute_single_step(state, step)
        
        return state
    
    def _execute_single_step(self, state: ReasoningState, step: Any) -> ReasoningState:
        """
        Execute a single step with retry logic
        
        Args:
            state: Current reasoning state
            step: Step to execute
            
        Returns:
            Updated state
        """
        logger.info(f"Executing step {step.step_id}: {step.sub_question}")
        
        # Fill placeholders using smart filler
        filled_question = self.placeholder_filler.fill_placeholders(
            template=step.sub_question,
            collected_facts=state.collected_facts
        )
        
        # Initialize execution trace
        trace = ExecutionTrace(
            step_id=step.step_id,
            sub_question=step.sub_question,
            filled_question=filled_question,
            retrieval_params=RetrievalParameters(query=filled_question),
            retrieved_docs=[],
            status=ExecutionStatus.IN_PROGRESS
        )
        
        # Retry loop
        for attempt in range(self.max_retries):
            trace.attempts = attempt + 1
            
            # Retrieve documents
            if attempt == 0:
                # Initial retrieval
                retrieval_params = RetrievalParameters(
                    query=filled_question,
                    top_k=5
                )
            else:
                # Use rewritten query
                retrieval_params = RetrievalParameters(
                    query=trace.rewrite_result.rewritten_query,
                    top_k=5
                )
            
            trace.retrieval_params = retrieval_params
            documents = self._retrieve_documents(retrieval_params)
            trace.retrieved_docs = documents
            
            # Analyze documents
            analysis_result = self.analysis_answer_agent.process({
                'sub_question': filled_question,
                'documents': documents
            })['result']
            
            trace.analysis_result = analysis_result
            
            # Check if sufficient
            if analysis_result.is_sufficient():
                # Success! Record answer
                state.collected_facts[step.step_id] = analysis_result.answer
                trace.status = ExecutionStatus.COMPLETED
                state.add_execution_trace(trace)
                
                # Record success in trajectory memory
                self.trajectory_memory.record_execution_success(
                    step_id=step.step_id,
                    answer=analysis_result.answer,
                    supporting_docs=analysis_result.supporting_docs
                )
                
                logger.info(f"Step {step.step_id} completed: {analysis_result.answer}")
                break
            else:
                # Insufficient information - rewrite query
                if attempt < self.max_retries - 1:
                    # Create document preview for rewrite agent
                    docs_preview = self._create_docs_preview(trace.retrieved_docs[:3])  # Top 3 docs
                    
                    rewrite_result = self.rewrite_agent.process({
                        'missing_info': analysis_result.analysis,
                        'original_query': filled_question,
                        'docs_preview': docs_preview
                    })['result']
                    
                    trace.rewrite_result = rewrite_result
                    
                    # Record rewrite attempt
                    self.trajectory_memory.record_rewrite_attempt(
                        step_id=step.step_id,
                        original_query=filled_question,
                        rewritten_query=rewrite_result.rewritten_query,
                        strategy=rewrite_result.strategy
                    )
                    
                    logger.info(f"Rewriting query: {rewrite_result.rewritten_query}")
        
        # If we exit the loop without success
        if trace.status != ExecutionStatus.COMPLETED:
            trace.status = ExecutionStatus.FAILED
            state.add_failure(f"Failed to answer step {step.step_id} after {self.max_retries} attempts")
            state.add_execution_trace(trace)
            
            # Record failure
            self.trajectory_memory.record_execution_failure(
                step_id=step.step_id,
                reason="Maximum retries exceeded"
            )
        
        return state
    
    def _create_docs_preview(self, documents: List[Document]) -> str:
        """
        Create a preview of documents for the Rewrite Agent
        
        Args:
            documents: List of documents to preview
            
        Returns:
            Formatted document preview string
        """
        if not documents:
            return "No documents retrieved"
        
        preview_parts = []
        for i, doc in enumerate(documents):
            # Truncate content for preview
            content_preview = doc.content[:150] + "..." if len(doc.content) > 150 else doc.content
            preview_parts.append(f"[Doc{i+1}] {content_preview}")
        
        return "\n".join(preview_parts)
    
    def _retrieve_documents(self, params: RetrievalParameters) -> List[Document]:
        """
        Retrieve documents using the retrieval coordinator with multi-query strategy
        
        Args:
            params: Retrieval parameters
            
        Returns:
            List of retrieved documents
        """
        # Generate query variants for better coverage
        query_variants = self.query_generator.generate_variants(params.query)
        logger.info(f"Generated {len(query_variants)} query variants: {query_variants}")
        
        # Retrieve documents for each variant
        all_documents = []
        seen_doc_ids = set()
        
        for variant_query in query_variants:
            variant_params = RetrievalParameters(
                query=variant_query,
                top_k=params.top_k,
                search_type=params.search_type,
                filters=params.filters,
                rerank=params.rerank
            )
            
            if hasattr(self.retrieval_coordinator, 'retrieve'):
                docs = self.retrieval_coordinator.retrieve(variant_params)
            else:
                # Placeholder implementation
                docs = [
                    Document(
                        doc_id=f"doc_{variant_query[:10]}_{i}",
                        content=f"Placeholder document {i} for query: {variant_query}",
                        score=0.9 - i * 0.1
                    )
                    for i in range(params.top_k)
                ]
            
            # Merge results, avoiding duplicates
            for doc in docs:
                if doc.doc_id not in seen_doc_ids:
                    seen_doc_ids.add(doc.doc_id)
                    all_documents.append(doc)
                else:
                    # If document already seen, boost its score
                    for existing_doc in all_documents:
                        if existing_doc.doc_id == doc.doc_id:
                            existing_doc.score = max(existing_doc.score, doc.score * 1.2)
                            break
        
        # Sort by score and return top documents
        all_documents.sort(key=lambda x: x.score, reverse=True)
        return all_documents[:params.top_k * 2]  # Return more documents for better coverage
    
    def _formulation_phase(self, state: ReasoningState) -> ReasoningState:
        """
        Formulate final answer from collected facts using intelligent synthesis
        
        Args:
            state: Current reasoning state
            
        Returns:
            Updated state with final answer
        """
        logger.info("Executing answer formulation phase...")
        
        # Collect all answers in order
        if state.strategic_plan:
            answers = []
            for step in state.strategic_plan.sub_questions:
                if step.step_id in state.collected_facts:
                    answers.append(state.collected_facts[step.step_id])
            
            # Use the last answer as final answer (original design)
            if answers:
                state.final_answer = answers[-1]
                # Set confidence based on completion rate
                total_steps = len(state.strategic_plan.sub_questions)
                successful_steps = len(answers)
                state.confidence_score = successful_steps / total_steps if total_steps > 0 else 0.2
            else:
                state.final_answer = "Unable to find answer"
                state.confidence_score = 0.0
        else:
            state.final_answer = "Unable to find answer"
            state.confidence_score = 0.0
        
        return state
    
    def _create_success_response(self, state: ReasoningState) -> Dict[str, Any]:
        """Create success response"""
        return {
            'status': 'success',
            'answer': state.final_answer,
            'confidence': state.confidence_score,
            'reasoning_state': state.to_dict(),
            'trajectory': self.trajectory_memory.get_complete_trajectory()
        }
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create error response"""
        return {
            'status': 'error',
            'error': error_message,
            'answer': None,
            'confidence': 0.0
        }