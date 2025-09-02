"""
Analysis-Answer Agent GRPO Trainer
Implements specialized training for information extraction and sufficiency judgment
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, List, Any, Optional
import numpy as np
import json
import logging
from pathlib import Path

from .mapgrpo_base import BaseAgentTrainer, MAPGRPOConfig
from .reward_functions import AnalysisAnswerRewardFunction
from ..agents import AnalysisAnswerAgent
from ..data.structures import Document

logger = logging.getLogger(__name__)


class AnalysisAgentDataset(Dataset):
    """Dataset for Analysis-Answer Agent training"""
    
    def __init__(self, data: List[Dict[str, Any]], group_size: int = 12):
        """
        Initialize dataset
        
        Args:
            data: List of training examples, each containing:
                - sub_question: Sub-question to answer
                - documents: Retrieved documents
                - scenario: "yes" or "no" scenario
                - ground_truth: Expected output
            group_size: Number of candidates per group
        """
        self.data = data
        self.group_size = group_size
        
        # Separate YES and NO scenarios for balanced training
        self.yes_scenarios = [d for d in data if d["scenario"] == "yes"]
        self.no_scenarios = [d for d in data if d["scenario"] == "no"]
        
        logger.info(f"Analysis dataset: {len(self.yes_scenarios)} YES, {len(self.no_scenarios)} NO scenarios")
    
    def __len__(self):
        # Use the larger set size to ensure all data is used
        return max(len(self.yes_scenarios), len(self.no_scenarios))
    
    def __getitem__(self, idx):
        # Alternate between YES and NO scenarios for balance
        if idx % 2 == 0 and idx // 2 < len(self.yes_scenarios):
            return self.yes_scenarios[idx // 2]
        elif idx // 2 < len(self.no_scenarios):
            return self.no_scenarios[idx // 2]
        else:
            # Wrap around if one set is exhausted
            if len(self.yes_scenarios) > len(self.no_scenarios):
                return self.yes_scenarios[idx % len(self.yes_scenarios)]
            else:
                return self.no_scenarios[idx % len(self.no_scenarios)]


class AnalysisAgentTrainer(BaseAgentTrainer):
    """
    Specialized trainer for Analysis-Answer Agent
    Handles information extraction and sufficiency judgment training
    """
    
    def __init__(self, config: MAPGRPOConfig):
        """
        Initialize Analysis-Answer Agent trainer
        
        Args:
            config: Training configuration
        """
        super().__init__(config.analysis_model_name, config)
        
        # Initialize reward function
        self.reward_function = AnalysisAnswerRewardFunction(
            judgment_weight=config.analysis_reward_weights["yes"]["status"],
            answer_weight=config.analysis_reward_weights["yes"]["answer"],
            format_weight=config.analysis_reward_weights["yes"]["format"]
        )
        
        # Analysis agent instance for generation
        self.analysis_agent = AnalysisAnswerAgent(model_name=config.analysis_model_name)
    
    def prepare_training_data(self, data: List[Dict]) -> Dataset:
        """
        Prepare Analysis-Answer Agent training data
        
        Args:
            data: Raw training data
            
        Returns:
            AnalysisAgentDataset instance
        """
        return AnalysisAgentDataset(data, self.config.group_size)
    
    def compute_rewards(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Compute rewards for analysis candidates
        
        Args:
            batch: Batch containing sub-questions, documents, and candidates
            
        Returns:
            Tensor of rewards for each candidate
        """
        rewards = []
        
        # Process each example
        for i in range(len(batch["sub_questions"])):
            sub_question = batch["sub_questions"][i]
            documents = batch["documents"][i]
            ground_truth = batch["ground_truths"][i]
            candidates = batch["candidates"][i]
            
            # Compute reward for each candidate
            for candidate in candidates:
                reward = self.reward_function.compute_reward({
                    "output": candidate,
                    "ground_truth": ground_truth
                })
                rewards.append(reward)
        
        return torch.tensor(rewards, dtype=torch.float32)
    
    def generate_candidates(
        self,
        sub_questions: List[str],
        documents_list: List[List[Document]],
        num_candidates: int = 12
    ) -> List[List[Dict[str, Any]]]:
        """
        Generate analysis candidates for sub-questions
        
        Args:
            sub_questions: List of sub-questions
            documents_list: List of document lists for each sub-question
            num_candidates: Number of candidates per question
            
        Returns:
            List of candidate lists
        """
        all_candidates = []
        
        for sub_question, documents in zip(sub_questions, documents_list):
            candidates = []
            
            # Generate with different temperatures and prompting strategies
            temperatures = [0.1, 0.3, 0.5, 0.7]
            samples_per_temp = num_candidates // len(temperatures)
            
            for temp in temperatures:
                for sample_idx in range(samples_per_temp):
                    try:
                        # Vary prompting slightly for diversity
                        if sample_idx % 2 == 0:
                            # Standard prompting
                            result = self.analysis_agent.analyze(
                                sub_question,
                                documents,
                                temperature=temp
                            )
                        else:
                            # Slightly modified prompting (e.g., emphasizing accuracy)
                            result = self.analysis_agent.analyze(
                                sub_question,
                                documents,
                                temperature=temp,
                                emphasis="accuracy"
                            )
                        
                        candidates.append(result)
                        
                    except Exception as e:
                        logger.warning(f"Failed to generate candidate: {e}")
                        # Add dummy candidate
                        candidates.append({
                            "status": "no",
                            "answer": "",
                            "analysis": "Unable to determine from the provided documents."
                        })
            
            # Ensure we have exactly num_candidates
            while len(candidates) < num_candidates:
                candidates.append(candidates[-1])  # Duplicate last
            
            all_candidates.append(candidates[:num_candidates])
        
        return all_candidates
    
    def _create_training_groups(
        self,
        examples: List[Dict]
    ) -> List[Dict[str, Any]]:
        """
        Create training groups with generated candidates
        
        Args:
            examples: Raw examples from dataset
            
        Returns:
            List of training groups
        """
        groups = []
        
        # Batch examples for efficient generation
        batch_size = 4
        for i in range(0, len(examples), batch_size):
            batch = examples[i:i + batch_size]
            
            sub_questions = [ex["sub_question"] for ex in batch]
            documents_list = [ex["documents"] for ex in batch]
            
            # Generate candidates for this batch
            candidates_list = self.generate_candidates(
                sub_questions,
                documents_list,
                self.config.group_size
            )
            
            # Create groups
            for ex, candidates in zip(batch, candidates_list):
                groups.append({
                    "sub_question": ex["sub_question"],
                    "documents": ex["documents"],
                    "candidates": candidates,
                    "ground_truth": ex["ground_truth"],
                    "scenario": ex["scenario"]
                })
        
        return groups
    
    def _collate_fn(self, examples: List[Dict]) -> Dict[str, Any]:
        """
        Custom collate function for Analysis-Answer Agent
        
        Args:
            examples: List of examples from dataset
            
        Returns:
            Batched data
        """
        # Create training groups with candidates
        groups = self._create_training_groups(examples)
        
        # Prepare for tokenization
        all_texts = []
        all_sub_questions = []
        all_documents = []
        all_candidates = []
        all_ground_truths = []
        
        for group in groups:
            sub_question = group["sub_question"]
            documents = group["documents"]
            candidates = group["candidates"]
            ground_truth = group["ground_truth"]
            
            # Create prompts for each candidate
            for candidate in candidates:
                # Build prompt
                prompt = self.analysis_agent.get_prompt({
                    "sub_question": sub_question,
                    "documents": documents
                })
                
                # Create response
                response = json.dumps(candidate, ensure_ascii=False)
                
                # Full text
                full_text = f"{prompt}\n{response}"
                all_texts.append(full_text)
            
            # Store group information
            all_sub_questions.extend([sub_question] * len(candidates))
            all_documents.extend([documents] * len(candidates))
            all_candidates.extend(candidates)
            all_ground_truths.extend([ground_truth] * len(candidates))
        
        # Tokenize all texts
        encoded = self.tokenizer(
            all_texts,
            padding=True,
            truncation=True,
            max_length=2048,  # Longer for documents
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "sub_questions": all_sub_questions,
            "documents": all_documents,
            "candidates": all_candidates,
            "ground_truths": all_ground_truths
        }
    
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Single training step with detailed metrics
        
        Args:
            batch: Training batch
            
        Returns:
            Dictionary of metrics
        """
        # Compute loss
        loss, outputs = self.compute_loss(batch, return_outputs=True)
        
        # Extract metrics
        rewards = outputs["rewards"]
        advantages = outputs["advantages"]
        
        # Separate metrics for YES/NO scenarios
        yes_mask = torch.tensor([
            gt.get("status") == "yes" for gt in batch["ground_truths"]
        ], device=rewards.device)
        
        no_mask = ~yes_mask
        
        metrics = {
            "loss": loss.item(),
            "avg_reward": rewards.mean().item(),
            "avg_advantage": advantages.mean().item(),
            "yes_reward": rewards[yes_mask].mean().item() if yes_mask.any() else 0.0,
            "no_reward": rewards[no_mask].mean().item() if no_mask.any() else 0.0,
            "yes_count": yes_mask.sum().item(),
            "no_count": no_mask.sum().item()
        }
        
        return metrics
    
    def evaluate(
        self,
        eval_dataset: Dataset,
        num_samples: int = 100
    ) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            eval_dataset: Evaluation dataset
            num_samples: Number of samples to evaluate
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        
        total_correct_yes = 0
        total_correct_no = 0
        total_yes = 0
        total_no = 0
        total_em_score = 0
        em_count = 0
        
        with torch.no_grad():
            for i in range(min(num_samples, len(eval_dataset))):
                example = eval_dataset[i]
                
                # Generate prediction
                result = self.analysis_agent.analyze(
                    example["sub_question"],
                    example["documents"],
                    temperature=0.1  # Low temperature for evaluation
                )
                
                ground_truth = example["ground_truth"]
                
                # Evaluate judgment accuracy
                if ground_truth["status"] == "yes":
                    total_yes += 1
                    if result["status"] == "yes":
                        total_correct_yes += 1
                        
                        # Evaluate EM score for YES scenarios
                        pred_answer = self._normalize_answer(result.get("answer", ""))
                        true_answer = self._normalize_answer(ground_truth.get("answer", ""))
                        
                        if pred_answer == true_answer:
                            total_em_score += 1
                        em_count += 1
                else:
                    total_no += 1
                    if result["status"] == "no":
                        total_correct_no += 1
        
        metrics = {
            "yes_accuracy": total_correct_yes / max(total_yes, 1),
            "no_accuracy": total_correct_no / max(total_no, 1),
            "overall_accuracy": (total_correct_yes + total_correct_no) / max(total_yes + total_no, 1),
            "em_score": total_em_score / max(em_count, 1),
            "yes_samples": total_yes,
            "no_samples": total_no
        }
        
        self.model.train()
        return metrics
    
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
    
    def save_checkpoint(self, step: int):
        """Save Analysis-Answer Agent checkpoint"""
        save_path = Path(self.config.output_dir) / "analysis_agent" / f"checkpoint-{step}"
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
        
        logger.info(f"Analysis-Answer Agent checkpoint saved to {save_path}")