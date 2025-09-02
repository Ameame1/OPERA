"""
Plan Agent GRPO Trainer
Implements specialized training for the strategic planning agent
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, List, Any, Optional
import numpy as np
import json
import logging
from pathlib import Path

from .mapgrpo_base import BaseAgentTrainer, MAPGRPOConfig
from .reward_functions import PlanAgentRewardFunction
from ..agents import PlanAgent, AnalysisAnswerAgent
from ..data.structures import Document
from ..retrieval import RetrievalCoordinator

logger = logging.getLogger(__name__)


class PlanAgentDataset(Dataset):
    """Dataset for Plan Agent training"""
    
    def __init__(self, data: List[Dict[str, Any]]):
        """
        Initialize dataset
        
        Args:
            data: List of training examples, each containing:
                - question: Original question
                - candidates: List of plan candidates with scores
                - golden_plan: Expert-generated plan
        """
        self.data = data
        self.group_size = 12  # Default group size
        
        # Group data by questions
        self.grouped_data = self._group_by_question()
    
    def _group_by_question(self) -> List[Dict[str, Any]]:
        """Group candidates by question"""
        grouped = []
        
        for item in self.data:
            question = item["question"]
            candidates = item["candidates"]
            golden_plan = item["golden_plan"]
            
            # Ensure we have exactly group_size candidates
            if len(candidates) < self.group_size:
                # Pad with variations of the golden plan
                while len(candidates) < self.group_size:
                    candidates.append(golden_plan)
            elif len(candidates) > self.group_size:
                # Take top candidates by score
                candidates = sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)
                candidates = candidates[:self.group_size]
            
            grouped.append({
                "question": question,
                "candidates": candidates,
                "golden_plan": golden_plan,
                "ground_truth": item.get("ground_truth", {})
            })
        
        return grouped
    
    def __len__(self):
        return len(self.grouped_data)
    
    def __getitem__(self, idx):
        return self.grouped_data[idx]


class PlanAgentTrainer(BaseAgentTrainer):
    """
    Specialized trainer for Plan Agent
    Handles strategic decomposition training with end-to-end execution scoring
    """
    
    def __init__(
        self,
        config: MAPGRPOConfig,
        analysis_agent: Optional[AnalysisAnswerAgent] = None,
        retrieval_coordinator: Optional[RetrievalCoordinator] = None
    ):
        """
        Initialize Plan Agent trainer
        
        Args:
            config: Training configuration
            analysis_agent: Pre-trained Analysis-Answer agent for execution scoring
            retrieval_coordinator: Retrieval system for document fetching
        """
        super().__init__(config.plan_model_name, config)
        
        # Components for end-to-end execution
        self.analysis_agent = analysis_agent
        self.retrieval_coordinator = retrieval_coordinator
        
        # Initialize reward function with 5-component system
        self.reward_function = PlanAgentRewardFunction(
            logic_weight=config.plan_reward_weights["logical_coherence"],
            execution_weight=config.plan_reward_weights["execution_feasibility"],
            accuracy_weight=config.plan_reward_weights["answer_accuracy"],
            efficiency_weight=config.plan_reward_weights["efficiency"],
            placeholder_weight=config.plan_reward_weights["placeholder_usage"]
        )
        
        # Plan agent instance for generation
        self.plan_agent = PlanAgent(model_name=config.plan_model_name)
    
    def prepare_training_data(self, data: List[Dict]) -> Dataset:
        """
        Prepare Plan Agent training data
        
        Args:
            data: Raw training data
            
        Returns:
            PlanAgentDataset instance
        """
        return PlanAgentDataset(data)
    
    def compute_rewards(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Compute rewards for plan candidates
        Implements end-to-end execution scoring
        
        Args:
            batch: Batch containing questions and candidate plans
            
        Returns:
            Tensor of rewards for each candidate
        """
        rewards = []
        
        # Process each group
        for group in batch["groups"]:
            question = group["question"]
            candidates = group["candidates"]
            ground_truth = group.get("ground_truth", {})
            
            group_rewards = []
            
            for candidate in candidates:
                # End-to-end execution scoring (if components available)
                execution_result = {}
                if self.analysis_agent and self.retrieval_coordinator:
                    execution_result = self._execute_plan(
                        question, candidate, ground_truth
                    )
                
                # Compute comprehensive reward using all 5 components
                total_reward = self.reward_function.compute_reward({
                    "input": {"question": question},
                    "output": candidate,
                    "ground_truth": ground_truth,
                    "execution_result": execution_result
                })
                
                group_rewards.append(total_reward)
            
            rewards.extend(group_rewards)
        
        return torch.tensor(rewards, dtype=torch.float32)
    
    def _execute_plan(
        self,
        question: str,
        plan: Dict[str, Any],
        ground_truth: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute plan end-to-end and return results
        
        Args:
            question: Original question
            plan: Candidate plan
            ground_truth: Ground truth information
            
        Returns:
            Execution result dictionary
        """
        try:
            sub_questions = plan.get("sub_questions", [])
            if not sub_questions:
                return {"success_rate": 0.0, "completion_rate": 0.0, "final_answer": ""}
            
            collected_facts = {}
            execution_success = 0
            completed_steps = 0
            total_steps = len(sub_questions)
            final_answer = ""
            
            # Execute each step
            for step in sub_questions:
                step_id = step.get("step_id", 0)
                sub_question = step.get("sub_question", "")
                
                # Fill placeholders
                filled_question = self._fill_placeholders(
                    sub_question, collected_facts
                )
                
                # Retrieve documents
                docs = []
                if self.retrieval_coordinator:
                    retrieval_params = {
                        "query": filled_question,
                        "top_k": 5
                    }
                    docs = self.retrieval_coordinator.retrieve(retrieval_params)
                
                # Analyze with Analysis-Answer agent
                if self.analysis_agent and docs:
                    analysis_result = self.analysis_agent.analyze({
                        "sub_question": filled_question,
                        "documents": docs
                    })
                    
                    if analysis_result.get("status") == "yes":
                        execution_success += 1
                        completed_steps += 1
                        # Store the answer as collected fact
                        answer = analysis_result.get("answer", "")
                        collected_facts[step_id] = answer
                        # Update final answer
                        if step_id == len(sub_questions):  # Last step
                            final_answer = answer
                    else:
                        # Step failed but we count it as attempted
                        completed_steps += 1
            
            # Return execution result
            return {
                "success_rate": execution_success / max(total_steps, 1),
                "completion_rate": completed_steps / max(total_steps, 1),
                "final_answer": final_answer,
                "collected_facts": collected_facts
            }
            
        except Exception as e:
            logger.warning(f"Error in plan execution: {e}")
            return {"success_rate": 0.0, "completion_rate": 0.0, "final_answer": ""}
    
    def _fill_placeholders(
        self,
        template: str,
        collected_facts: Dict[int, str]
    ) -> str:
        """Fill placeholders in sub-question"""
        import re
        
        # Pattern: [entity from step X]
        pattern = r'\[([^\]]+) from step (\d+)\]'
        
        def replace_func(match):
            info_type = match.group(1)
            step_num = int(match.group(2))
            
            if step_num in collected_facts:
                return collected_facts[step_num]
            else:
                return match.group(0)  # Keep original if not found
        
        return re.sub(pattern, replace_func, template)
    
    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison"""
        if not answer:
            return ""
        
        answer = answer.lower().strip()
        # Remove articles
        for article in ["the", "a", "an"]:
            answer = answer.replace(f" {article} ", " ")
        # Remove punctuation
        import string
        answer = answer.translate(str.maketrans("", "", string.punctuation))
        # Remove extra spaces
        answer = " ".join(answer.split())
        
        return answer
    
    def generate_candidates(
        self,
        questions: List[str],
        num_candidates: int = 12
    ) -> List[List[Dict[str, Any]]]:
        """
        Generate plan candidates for questions
        
        Args:
            questions: List of questions
            num_candidates: Number of candidates per question
            
        Returns:
            List of candidate lists
        """
        all_candidates = []
        
        for question in questions:
            candidates = []
            
            # Generate with different temperatures
            temperatures = [0.3, 0.5, 0.7, 0.9]
            samples_per_temp = num_candidates // len(temperatures)
            
            for temp in temperatures:
                for _ in range(samples_per_temp):
                    try:
                        plan = self.plan_agent.decompose(question, temperature=temp)
                        candidates.append(plan)
                    except Exception as e:
                        logger.warning(f"Failed to generate candidate: {e}")
                        # Add dummy candidate
                        candidates.append({
                            "sub_questions": [
                                {
                                    "step_id": 1,
                                    "sub_question": question,
                                    "dependencies": []
                                }
                            ]
                        })
            
            # Ensure we have exactly num_candidates
            while len(candidates) < num_candidates:
                candidates.append(candidates[-1])  # Duplicate last
            
            all_candidates.append(candidates[:num_candidates])
        
        return all_candidates
    
    def _collate_fn(self, examples: List[Dict]) -> Dict[str, Any]:
        """
        Custom collate function for Plan Agent
        
        Args:
            examples: List of examples from dataset
            
        Returns:
            Batched data
        """
        # Group all candidates together
        all_texts = []
        groups = []
        
        for example in examples:
            question = example["question"]
            candidates = example["candidates"]
            
            # Create prompts for each candidate
            group_texts = []
            for candidate in candidates:
                prompt = self.plan_agent.build_prompt(question)
                response = json.dumps(candidate, ensure_ascii=False)
                full_text = f"{prompt}\n{response}"
                
                all_texts.append(full_text)
                group_texts.append(full_text)
            
            groups.append({
                "question": question,
                "candidates": candidates,
                "ground_truth": example.get("ground_truth", {}),
                "texts": group_texts
            })
        
        # Tokenize all texts together
        encoded = self.tokenizer(
            all_texts,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "groups": groups
        }
    
    def save_checkpoint(self, step: int):
        """Save Plan Agent checkpoint"""
        save_path = Path(self.config.output_dir) / "plan_agent" / f"checkpoint-{step}"
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        # Save training state
        state = {
            "step": step,
            "config": self.config.__dict__,
            "metrics": self.metrics
        }
        
        with open(save_path / "trainer_state.json", "w") as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Plan Agent checkpoint saved to {save_path}")