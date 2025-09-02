"""
Rewrite Agent GRPO Trainer
Implements specialized training for query reformulation
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, List, Any, Optional
import numpy as np
import json
import logging
from pathlib import Path

from .mapgrpo_base import BaseAgentTrainer, MAPGRPOConfig
from .reward_functions import RewriteAgentRewardFunction
from ..agents import RewriteAgent
from ..retrieval import RetrievalCoordinator

logger = logging.getLogger(__name__)


class RewriteAgentDataset(Dataset):
    """Dataset for Rewrite Agent training"""
    
    def __init__(self, data: List[Dict[str, Any]], group_size: int = 12):
        """
        Initialize dataset
        
        Args:
            data: List of training examples, each containing:
                - original_query: Failed query
                - failure_info: Reason for failure from Analysis Agent
                - current_docs: Documents retrieved with original query
                - golden_docs: Target documents that should be retrieved
            group_size: Number of candidates per group
        """
        self.data = data
        self.group_size = group_size
        
        logger.info(f"Rewrite dataset: {len(data)} failure scenarios")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class RewriteAgentTrainer(BaseAgentTrainer):
    """
    Specialized trainer for Rewrite Agent
    Handles query reformulation training with retrieval feedback
    """
    
    def __init__(
        self,
        config: MAPGRPOConfig,
        retrieval_coordinator: Optional[RetrievalCoordinator] = None
    ):
        """
        Initialize Rewrite Agent trainer
        
        Args:
            config: Training configuration
            retrieval_coordinator: Retrieval system for evaluating rewrites
        """
        # Rewrite agent uses smaller model
        super().__init__(config.rewrite_model_name, config)
        
        self.retrieval_coordinator = retrieval_coordinator
        
        # Initialize reward function
        self.reward_function = RewriteAgentRewardFunction(
            retrieval_weight=config.rewrite_reward_weights["ndcg"],
            format_weight=config.rewrite_reward_weights["format"]
        )
        
        # Rewrite agent instance for generation
        self.rewrite_agent = RewriteAgent(model_name=config.rewrite_model_name)
    
    def prepare_training_data(self, data: List[Dict]) -> Dataset:
        """
        Prepare Rewrite Agent training data
        
        Args:
            data: Raw training data (failure cases)
            
        Returns:
            RewriteAgentDataset instance
        """
        return RewriteAgentDataset(data, self.config.group_size)
    
    def compute_rewards(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Compute rewards for rewrite candidates
        Includes retrieval evaluation if retriever is available
        
        Args:
            batch: Batch containing queries and rewrite candidates
            
        Returns:
            Tensor of rewards for each candidate
        """
        rewards = []
        
        # Process each example
        for i in range(len(batch["original_queries"])):
            original_query = batch["original_queries"][i]
            failure_info = batch["failure_infos"][i]
            current_docs = batch["current_docs"][i]
            golden_docs = batch["golden_docs"][i]
            candidates = batch["candidates"][i]
            
            # Evaluate each rewrite candidate
            for candidate in candidates:
                # Static format evaluation
                candidate_dict = {
                    "input": {
                        "original_query": original_query,
                        "failure_info": failure_info
                    },
                    "output": candidate,
                    "golden_docs": golden_docs
                }
                
                # If retriever available, evaluate retrieval quality
                if self.retrieval_coordinator:
                    rewritten_query = candidate.get("rewritten_query", "")
                    if rewritten_query:
                        # Retrieve with rewritten query
                        retrieval_params = {
                            "query": rewritten_query,
                            "top_k": 10
                        }
                        retrieved_docs = self.retrieval_coordinator.retrieve(retrieval_params)
                        
                        # Extract document IDs
                        retrieved_ids = [doc.doc_id for doc in retrieved_docs]
                        candidate_dict["retrieved_docs"] = retrieved_ids
                
                # Compute reward
                reward = self.reward_function.compute_reward(candidate_dict)
                rewards.append(reward)
        
        return torch.tensor(rewards, dtype=torch.float32)
    
    def generate_candidates(
        self,
        original_queries: List[str],
        failure_infos: List[str],
        current_docs_list: List[List[Dict]],
        num_candidates: int = 12
    ) -> List[List[Dict[str, Any]]]:
        """
        Generate rewrite candidates for failed queries
        
        Args:
            original_queries: List of failed queries
            failure_infos: List of failure reasons
            current_docs_list: List of current document lists
            num_candidates: Number of candidates per query
            
        Returns:
            List of candidate lists
        """
        all_candidates = []
        
        for orig_query, failure_info, current_docs in zip(
            original_queries, failure_infos, current_docs_list
        ):
            candidates = []
            
            # Generate with different strategies
            strategies = [
                ("keyword_expansion", 0.3),
                ("entity_focus", 0.5),
                ("temporal_focus", 0.5),
                ("relation_focus", 0.7)
            ]
            
            samples_per_strategy = num_candidates // len(strategies)
            
            for strategy, temperature in strategies:
                for _ in range(samples_per_strategy):
                    try:
                        result = self.rewrite_agent.rewrite(
                            sub_goal=orig_query,
                            current_docs=current_docs,
                            failure_info=failure_info,
                            temperature=temperature,
                            strategy_hint=strategy
                        )
                        candidates.append(result)
                        
                    except Exception as e:
                        logger.warning(f"Failed to generate candidate: {e}")
                        # Add dummy candidate
                        candidates.append({
                            "rewritten_query": orig_query + " information data",
                            "strategy": strategy,
                            "keywords": orig_query.split()[:5]
                        })
            
            # Add some variations without strategy hints
            while len(candidates) < num_candidates:
                try:
                    result = self.rewrite_agent.rewrite(
                        sub_goal=orig_query,
                        current_docs=current_docs,
                        failure_info=failure_info,
                        temperature=np.random.uniform(0.3, 0.9)
                    )
                    candidates.append(result)
                except:
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
        
        # Process in batches
        batch_size = 4
        for i in range(0, len(examples), batch_size):
            batch = examples[i:i + batch_size]
            
            original_queries = [ex["original_query"] for ex in batch]
            failure_infos = [ex["failure_info"] for ex in batch]
            current_docs_list = [ex["current_docs"] for ex in batch]
            
            # Generate candidates
            candidates_list = self.generate_candidates(
                original_queries,
                failure_infos,
                current_docs_list,
                self.config.group_size
            )
            
            # Create groups
            for ex, candidates in zip(batch, candidates_list):
                groups.append({
                    "original_query": ex["original_query"],
                    "failure_info": ex["failure_info"],
                    "current_docs": ex["current_docs"],
                    "golden_docs": ex.get("golden_docs", []),
                    "candidates": candidates
                })
        
        return groups
    
    def _collate_fn(self, examples: List[Dict]) -> Dict[str, Any]:
        """
        Custom collate function for Rewrite Agent
        
        Args:
            examples: List of examples from dataset
            
        Returns:
            Batched data
        """
        # Create training groups with candidates
        groups = self._create_training_groups(examples)
        
        # Prepare for tokenization
        all_texts = []
        all_original_queries = []
        all_failure_infos = []
        all_current_docs = []
        all_golden_docs = []
        all_candidates = []
        
        for group in groups:
            original_query = group["original_query"]
            failure_info = group["failure_info"]
            current_docs = group["current_docs"]
            golden_docs = group["golden_docs"]
            candidates = group["candidates"]
            
            # Create prompts for each candidate
            for candidate in candidates:
                # Build prompt
                prompt = self.rewrite_agent.get_prompt({
                    "original_query": original_query,
                    "failure_info": failure_info,
                    "docs_preview": self._format_docs_preview(current_docs)
                })
                
                # Create response
                response = json.dumps(candidate, ensure_ascii=False)
                
                # Full text
                full_text = f"{prompt}\n{response}"
                all_texts.append(full_text)
            
            # Store group information
            all_original_queries.extend([original_query] * len(candidates))
            all_failure_infos.extend([failure_info] * len(candidates))
            all_current_docs.extend([current_docs] * len(candidates))
            all_golden_docs.extend([golden_docs] * len(candidates))
            all_candidates.extend(candidates)
        
        # Tokenize all texts
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
            "original_queries": all_original_queries,
            "failure_infos": all_failure_infos,
            "current_docs": all_current_docs,
            "golden_docs": all_golden_docs,
            "candidates": all_candidates
        }
    
    def _format_docs_preview(self, docs: List[Dict]) -> str:
        """Format document preview for prompt"""
        if not docs:
            return "No documents available"
        
        previews = []
        for i, doc in enumerate(docs[:3]):  # Max 3 docs
            title = doc.get("title", f"Doc {i+1}")
            content = doc.get("content", "")[:100]
            previews.append(f"[{i+1}] {title}: {content}...")
        
        return "\n".join(previews)
    
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
        
        total_ndcg = 0.0
        total_improvement = 0.0
        valid_samples = 0
        
        with torch.no_grad():
            for i in range(min(num_samples, len(eval_dataset))):
                example = eval_dataset[i]
                
                # Generate rewrite
                result = self.rewrite_agent.rewrite(
                    sub_goal=example["original_query"],
                    current_docs=example["current_docs"],
                    failure_info=example["failure_info"],
                    temperature=0.1  # Low temperature for evaluation
                )
                
                if self.retrieval_coordinator and result.get("rewritten_query"):
                    # Evaluate retrieval improvement
                    orig_params = {
                        "query": example["original_query"],
                        "top_k": 10
                    }
                    orig_docs = self.retrieval_coordinator.retrieve(orig_params)
                    orig_ids = [doc.doc_id for doc in orig_docs]
                    
                    rewrite_params = {
                        "query": result["rewritten_query"],
                        "top_k": 10
                    }
                    rewrite_docs = self.retrieval_coordinator.retrieve(rewrite_params)
                    rewrite_ids = [doc.doc_id for doc in rewrite_docs]
                    
                    golden_docs = example.get("golden_docs", [])
                    
                    if golden_docs:
                        # Compute NDCG
                        orig_ndcg = self._compute_ndcg(orig_ids, golden_docs)
                        rewrite_ndcg = self._compute_ndcg(rewrite_ids, golden_docs)
                        
                        total_ndcg += rewrite_ndcg
                        total_improvement += (rewrite_ndcg - orig_ndcg)
                        valid_samples += 1
        
        metrics = {
            "avg_ndcg": total_ndcg / max(valid_samples, 1),
            "avg_improvement": total_improvement / max(valid_samples, 1),
            "valid_samples": valid_samples
        }
        
        self.model.train()
        return metrics
    
    def _compute_ndcg(self, retrieved_ids: List[str], golden_ids: List[str]) -> float:
        """Compute NDCG@k score"""
        if not golden_ids:
            return 0.0
        
        k = len(retrieved_ids)
        dcg = 0.0
        
        for i, doc_id in enumerate(retrieved_ids[:k]):
            if doc_id in golden_ids:
                dcg += 1.0 / np.log2(i + 2)
        
        # Ideal DCG
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(golden_ids))))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def save_checkpoint(self, step: int):
        """Save Rewrite Agent checkpoint"""
        save_path = Path(self.config.output_dir) / "rewrite_agent" / f"checkpoint-{step}"
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
        
        logger.info(f"Rewrite Agent checkpoint saved to {save_path}")