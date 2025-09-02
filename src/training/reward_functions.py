"""
Reward functions for MAPGRPO training
Implements role-specific reward functions for each agent
"""

from typing import Dict, Any, List, Optional
import numpy as np
import logging
from abc import ABC, abstractmethod
import re

logger = logging.getLogger(__name__)


class BaseRewardFunction(ABC):
    """Base class for reward functions"""
    
    @abstractmethod
    def compute_reward(
        self,
        candidate: Dict[str, Any],
        dependency_models: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Compute reward for a candidate
        
        Args:
            candidate: Candidate output
            dependency_models: Previously trained models
            
        Returns:
            Reward score
        """
        pass


class PlanAgentRewardFunction(BaseRewardFunction):
    """
    Reward function for Plan Agent
    Evaluates strategic decomposition quality
    """
    
    def __init__(
        self,
        logic_weight: float = 0.25,
        execution_weight: float = 0.25,
        accuracy_weight: float = 0.30,
        efficiency_weight: float = 0.10,
        placeholder_weight: float = 0.10
    ):
        """
        Initialize Plan Agent reward function with 5 components
        
        Args:
            logic_weight: Weight for logical coherence
            execution_weight: Weight for execution feasibility
            accuracy_weight: Weight for answer accuracy
            efficiency_weight: Weight for plan efficiency
            placeholder_weight: Weight for placeholder usage correctness
        """
        self.logic_weight = logic_weight
        self.execution_weight = execution_weight
        self.accuracy_weight = accuracy_weight
        self.efficiency_weight = efficiency_weight
        self.placeholder_weight = placeholder_weight
        
        # Weights should already sum to 1, but normalize just in case
        total = logic_weight + execution_weight + accuracy_weight + efficiency_weight + placeholder_weight
        if abs(total - 1.0) > 0.01:  # Allow small floating point errors
            self.logic_weight /= total
            self.execution_weight /= total
            self.accuracy_weight /= total
            self.efficiency_weight /= total
            self.placeholder_weight /= total
    
    def compute_reward(
        self,
        candidate: Dict[str, Any],
        dependency_models: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Compute reward for plan generation
        
        Args:
            candidate: Candidate plan
            dependency_models: Not used for Plan Agent
            
        Returns:
            Reward score
        """
        plan = candidate.get("output", {})
        question = candidate.get("input", {}).get("question", "")
        ground_truth = candidate.get("ground_truth", {})
        execution_result = candidate.get("execution_result", {})
        
        # Compute all 5 component scores
        logic_score = self._evaluate_logic(plan, question)
        execution_score = self._evaluate_execution_potential(plan)
        accuracy_score = self._evaluate_answer_accuracy(execution_result, ground_truth)
        efficiency_score = self._evaluate_efficiency(plan)
        placeholder_score = self._evaluate_placeholder_usage(plan)
        
        # Weighted combination using 5 components
        total_reward = (
            self.logic_weight * logic_score +
            self.execution_weight * execution_score +
            self.accuracy_weight * accuracy_score +
            self.efficiency_weight * efficiency_score +
            self.placeholder_weight * placeholder_score
        )
        
        return total_reward
    
    def _evaluate_logic(self, plan: Dict[str, Any], question: str) -> float:
        """
        Evaluate logical decomposition quality
        
        Args:
            plan: Generated plan
            question: Original question
            
        Returns:
            Logic score (0-1)
        """
        sub_questions = plan.get("sub_questions", [])
        
        if not sub_questions:
            return 0.0
        
        score = 1.0
        
        # Check if decomposition is atomic
        for sq in sub_questions:
            q_text = sq.get("sub_question", "")
            # Penalize compound questions (containing "and")
            if " and " in q_text.lower():
                score *= 0.8
        
        # Check logical flow
        for i, sq in enumerate(sub_questions):
            deps = sq.get("dependencies", [])
            # Dependencies should reference earlier steps
            if any(d >= i for d in deps):
                score *= 0.7
        
        # Check coverage (all key entities from question should appear)
        question_entities = self._extract_entities(question)
        plan_entities = set()
        for sq in sub_questions:
            plan_entities.update(self._extract_entities(sq.get("sub_question", "")))
        
        coverage = len(question_entities.intersection(plan_entities)) / max(len(question_entities), 1)
        score *= coverage
        
        return max(0.0, min(1.0, score))
    
    def _evaluate_answer_accuracy(self, execution_result: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
        """
        Evaluate answer accuracy using EM score
        
        Args:
            execution_result: Execution result containing final answer
            ground_truth: Ground truth containing correct answer
            
        Returns:
            Accuracy score (0-1)
        """
        if not execution_result or not ground_truth:
            return 0.0
            
        final_answer = execution_result.get("final_answer", "")
        correct_answer = ground_truth.get("answer", "")
        
        # Use exact match after normalization
        if self._normalize_answer(final_answer) == self._normalize_answer(correct_answer):
            return 1.0
        else:
            return 0.0
    
    def _evaluate_efficiency(self, plan: Dict[str, Any]) -> float:
        """
        Evaluate plan efficiency based on number of steps
        
        Args:
            plan: Generated plan
            
        Returns:
            Efficiency score (0-1)
        """
        sub_questions = plan.get("sub_questions", [])
        num_steps = len(sub_questions)
        
        if num_steps == 0:
            return 0.0
        elif num_steps <= 1:
            return 0.3  # Too simple
        elif num_steps <= 3:
            return 1.0  # Optimal
        elif num_steps <= 5:
            return 0.8  # Acceptable
        else:
            return 0.5  # Too complex
    
    def _evaluate_placeholder_usage(self, plan: Dict[str, Any]) -> float:
        """
        Evaluate correctness of placeholder usage
        
        Args:
            plan: Generated plan
            
        Returns:
            Placeholder score (0-1)
        """
        sub_questions = plan.get("sub_questions", [])
        
        if not sub_questions:
            return 0.0
        
        total_score = 0.0
        total_checks = 0
        
        # Check each sub-question
        for sq in sub_questions:
            q_text = sq.get("sub_question", "")
            dependencies = sq.get("dependencies", [])
            step_id = sq.get("step_id", 0)
            
            # Find placeholders in text
            placeholders = re.findall(r'\[([^\]]+) from step (\d+)\]', q_text)
            
            # Check placeholder correctness
            for entity_type, ref_step in placeholders:
                total_checks += 1
                ref_step_num = int(ref_step)
                
                # Check if reference is valid (to earlier step)
                if ref_step_num < step_id:
                    total_score += 0.5
                    
                    # Check if dependency is declared
                    if ref_step_num in dependencies:
                        total_score += 0.5
            
            # Check if all dependencies have placeholders
            for dep in dependencies:
                if f"from step {dep}]" not in q_text:
                    total_checks += 1
                    # Missing placeholder for declared dependency
                    total_score += 0.0
        
        if total_checks == 0:
            # No placeholders needed - that's fine
            return 1.0
        
        return total_score / total_checks
    
    def _evaluate_structure(self, plan: Dict[str, Any]) -> float:
        """
        Evaluate structural correctness
        
        Args:
            plan: Generated plan
            
        Returns:
            Structure score (0-1)
        """
        sub_questions = plan.get("sub_questions", [])
        
        if not sub_questions:
            return 0.0
        
        score = 1.0
        
        # Check required fields
        for sq in sub_questions:
            required_fields = ["step_id", "sub_question", "dependencies"]
            for field in required_fields:
                if field not in sq:
                    score *= 0.8
        
        # Check placeholder syntax
        for sq in sub_questions:
            q_text = sq.get("sub_question", "")
            placeholders = re.findall(r'\[([^\]]+) from step (\d+)\]', q_text)
            
            for info_type, step_num in placeholders:
                # Verify step reference is valid
                if int(step_num) >= sq.get("step_id", 999):
                    score *= 0.7
                # Verify dependency is declared
                if int(step_num) - 1 not in sq.get("dependencies", []):
                    score *= 0.9
        
        # Reasonable number of steps (2-4 typically)
        num_steps = len(sub_questions)
        if num_steps < 2:
            score *= 0.7
        elif num_steps > 5:
            score *= 0.8
        
        return max(0.0, min(1.0, score))
    
    def _evaluate_execution_potential(self, plan: Dict[str, Any]) -> float:
        """
        Evaluate potential for successful execution
        
        Args:
            plan: Generated plan
            
        Returns:
            Execution score (0-1)
        """
        sub_questions = plan.get("sub_questions", [])
        
        if not sub_questions:
            return 0.0
        
        score = 1.0
        
        # Check if questions are answerable
        for sq in sub_questions:
            q_text = sq.get("sub_question", "")
            
            # Questions should be specific
            if len(q_text.split()) < 3:
                score *= 0.8
            
            # Questions should have clear intent
            question_words = ["what", "who", "when", "where", "which", "how"]
            if not any(word in q_text.lower() for word in question_words):
                score *= 0.9
        
        # Check dependency chain feasibility
        completed = set()
        executable_count = 0
        
        for sq in sub_questions:
            deps = sq.get("dependencies", [])
            if all(d in completed for d in deps):
                executable_count += 1
                completed.add(sq.get("step_id", -1) - 1)  # Convert to 0-index
        
        feasibility = executable_count / max(len(sub_questions), 1)
        score *= feasibility
        
        return max(0.0, min(1.0, score))
    
    def _extract_entities(self, text: str) -> set:
        """Simple entity extraction (proper nouns)"""
        # This is a simplified version - real implementation would use NER
        words = text.split()
        entities = {word for word in words if word and word[0].isupper()}
        return entities


class AnalysisAnswerRewardFunction(BaseRewardFunction):
    """
    Reward function for Analysis-Answer Agent
    Evaluates information extraction and sufficiency judgment
    """
    
    def __init__(
        self,
        judgment_weight: float = 0.25,
        answer_weight: float = 0.65,
        format_weight: float = 0.10
    ):
        """
        Initialize Analysis-Answer reward function
        
        Args:
            judgment_weight: Weight for sufficiency judgment
            answer_weight: Weight for answer accuracy (EM)
            format_weight: Weight for output format
        """
        self.judgment_weight = judgment_weight
        self.answer_weight = answer_weight
        self.format_weight = format_weight
    
    def compute_reward(
        self,
        candidate: Dict[str, Any],
        dependency_models: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Compute reward for analysis-answer generation
        
        Args:
            candidate: Candidate output
            dependency_models: Not used directly
            
        Returns:
            Reward score
        """
        output = candidate.get("output", {})
        ground_truth = candidate.get("ground_truth", {})
        
        # For YES scenarios
        if ground_truth.get("status") == "yes":
            judgment_score = 1.0 if output.get("status") == "yes" else 0.0
            answer_score = self._compute_em_score(
                output.get("answer", ""),
                ground_truth.get("answer", "")
            )
            format_score = self._evaluate_format(output)
            
            total_reward = (
                self.judgment_weight * judgment_score +
                self.answer_weight * answer_score +
                self.format_weight * format_score
            )
        
        # For NO scenarios  
        else:
            # Adjust weights for NO scenarios
            judgment_score = 1.0 if output.get("status") == "no" else 0.0
            format_score = self._evaluate_format(output)
            
            total_reward = (
                0.90 * judgment_score +
                0.10 * format_score
            )
        
        return total_reward
    
    def _compute_em_score(self, predicted: str, ground_truth: str) -> float:
        """
        Compute Exact Match score
        
        Args:
            predicted: Predicted answer
            ground_truth: Ground truth answer
            
        Returns:
            EM score (0 or 1)
        """
        # Normalize answers
        pred_norm = self._normalize_answer(predicted)
        gt_norm = self._normalize_answer(ground_truth)
        
        return 1.0 if pred_norm == gt_norm else 0.0
    
    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison"""
        if not answer:
            return ""
        
        # Convert to lowercase
        answer = answer.lower()
        
        # Remove articles
        for article in ["the", "a", "an"]:
            answer = re.sub(f"\\b{article}\\b", "", answer)
        
        # Remove punctuation
        answer = re.sub(r"[^\w\s]", "", answer)
        
        # Remove extra whitespace
        answer = " ".join(answer.split())
        
        return answer.strip()
    
    def _evaluate_format(self, output: Dict[str, Any]) -> float:
        """Evaluate output format compliance"""
        required_fields = ["status", "analysis"]
        score = 1.0
        
        for field in required_fields:
            if field not in output:
                score *= 0.7
        
        # Check status value
        if output.get("status") not in ["yes", "no"]:
            score *= 0.5
        
        # Check answer field consistency
        if output.get("status") == "yes" and not output.get("answer"):
            score *= 0.8
        elif output.get("status") == "no" and output.get("answer"):
            score *= 0.9
        
        return score


class RewriteAgentRewardFunction(BaseRewardFunction):
    """
    Reward function for Rewrite Agent
    Evaluates query reformulation effectiveness
    """
    
    def __init__(
        self,
        retrieval_weight: float = 0.90,
        format_weight: float = 0.10
    ):
        """
        Initialize Rewrite Agent reward function
        
        Args:
            retrieval_weight: Weight for retrieval effectiveness
            format_weight: Weight for output format
        """
        self.retrieval_weight = retrieval_weight
        self.format_weight = format_weight
    
    def compute_reward(
        self,
        candidate: Dict[str, Any],
        dependency_models: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Compute reward for query rewriting
        
        Args:
            candidate: Candidate output
            dependency_models: May include retriever for evaluation
            
        Returns:
            Reward score
        """
        output = candidate.get("output", {})
        original_query = candidate.get("input", {}).get("original_query", "")
        
        # Compute retrieval effectiveness
        retrieval_score = self._evaluate_retrieval_improvement(
            original_query,
            output.get("rewritten_query", ""),
            candidate.get("retrieved_docs", []),
            candidate.get("golden_docs", [])
        )
        
        # Compute format score
        format_score = self._evaluate_format(output)
        
        # Weighted combination
        total_reward = (
            self.retrieval_weight * retrieval_score +
            self.format_weight * format_score
        )
        
        return total_reward
    
    def _evaluate_retrieval_improvement(
        self,
        original_query: str,
        rewritten_query: str,
        retrieved_docs: List[str],
        golden_docs: List[str]
    ) -> float:
        """
        Evaluate retrieval improvement using NDCG
        
        Args:
            original_query: Original query
            rewritten_query: Rewritten query
            retrieved_docs: Retrieved document IDs
            golden_docs: Golden document IDs
            
        Returns:
            Retrieval score (0-1)
        """
        if not golden_docs:
            return 0.5  # No ground truth available
        
        # Compute NDCG@k
        k = len(retrieved_docs)
        dcg = 0.0
        
        for i, doc_id in enumerate(retrieved_docs[:k]):
            if doc_id in golden_docs:
                # Relevance is 1 for golden docs, 0 otherwise
                dcg += 1.0 / np.log2(i + 2)  # i+2 because positions start at 1
        
        # Ideal DCG (all golden docs at top positions)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(golden_docs))))
        
        # NDCG
        ndcg = dcg / idcg if idcg > 0 else 0.0
        
        # Use square root for smoother reward gradient
        return np.sqrt(ndcg)
    
    def _evaluate_format(self, output: Dict[str, Any]) -> float:
        """Evaluate output format compliance"""
        required_fields = ["rewritten_query"]
        score = 1.0
        
        for field in required_fields:
            if field not in output:
                score *= 0.5
        
        # Check if rewritten query is non-empty
        if not output.get("rewritten_query", "").strip():
            score *= 0.3
        
        # Check strategy field
        valid_strategies = ["keyword_expansion", "entity_focus", "temporal_focus", "relation_focus"]
        if output.get("strategy") not in valid_strategies:
            score *= 0.9
        
        return score