"""
MAPGRPO Base Components - Core infrastructure for Multi-Agent Progressive GRPO
Borrowing best practices from TRL while maintaining OPERA's unique requirements
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments
)
from accelerate import Accelerator
import wandb

logger = logging.getLogger(__name__)


@dataclass
class MAPGRPOConfig:
    """Configuration for MAPGRPO training"""
    # Model configurations
    plan_model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    analysis_model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    rewrite_model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    
    # Training hyperparameters
    learning_rate: float = 1e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    
    # GRPO specific parameters
    group_size: int = 12  # Number of candidates per question
    beta: float = 0.10  # KL penalty coefficient
    gamma: float = 0.99  # Discount factor
    clip_range: float = 0.2  # PPO-style clipping
    
    # Stage-specific epochs
    plan_epochs: int = 3
    analysis_epochs: int = 2
    rewrite_epochs: int = 2
    
    # Reward weights
    plan_reward_weights: Dict[str, float] = None
    analysis_reward_weights: Dict[str, float] = None
    rewrite_reward_weights: Dict[str, float] = None
    
    # Output and logging
    output_dir: str = "checkpoints/mapgrpo"
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 100
    
    # Hardware settings
    fp16: bool = True
    gradient_checkpointing: bool = True
    
    def __post_init__(self):
        """Initialize default reward weights if not provided"""
        if self.plan_reward_weights is None:
            self.plan_reward_weights = {
                "logical_coherence": 0.25,
                "execution_feasibility": 0.25,
                "answer_accuracy": 0.30,
                "efficiency": 0.10,
                "placeholder_usage": 0.10
            }
        
        if self.analysis_reward_weights is None:
            self.analysis_reward_weights = {
                "yes": {"status": 0.25, "answer": 0.65, "format": 0.10},
                "no": {"status": 0.90, "format": 0.10}
            }
        
        if self.rewrite_reward_weights is None:
            self.rewrite_reward_weights = {
                "ndcg": 0.90,
                "format": 0.10
            }


class GroupAdvantageCalculator:
    """
    Computes group-based advantages following GRPO algorithm
    Inspired by TRL's implementation but adapted for OPERA
    """
    
    def __init__(self, group_size: int = 12):
        self.group_size = group_size
    
    def compute_advantages(
        self, 
        rewards: torch.Tensor,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Compute advantages using group-based normalization
        
        Args:
            rewards: Tensor of shape (batch_size * group_size,)
            normalize: Whether to normalize advantages
            
        Returns:
            advantages: Tensor of same shape as rewards
        """
        # Reshape to group format
        batch_size = rewards.size(0) // self.group_size
        grouped_rewards = rewards.view(batch_size, self.group_size)
        
        # Compute group statistics
        mean_rewards = grouped_rewards.mean(dim=1, keepdim=True)
        std_rewards = grouped_rewards.std(dim=1, keepdim=True)
        
        # Normalize within each group
        advantages = (grouped_rewards - mean_rewards) / (std_rewards + 1e-8)
        
        # Flatten back
        advantages = advantages.view(-1)
        
        if normalize:
            # Additional normalization across all advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages
    
    def compute_rewards_with_kl_penalty(
        self,
        rewards: torch.Tensor,
        log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        beta: float = 0.1
    ) -> torch.Tensor:
        """
        Apply KL penalty to rewards
        
        Args:
            rewards: Base rewards
            log_probs: Log probabilities from current policy
            ref_log_probs: Log probabilities from reference policy
            beta: KL penalty coefficient
            
        Returns:
            Adjusted rewards with KL penalty
        """
        kl_divergence = log_probs - ref_log_probs
        adjusted_rewards = rewards - beta * kl_divergence
        
        return adjusted_rewards


class BaseAgentTrainer(ABC):
    """
    Base class for agent-specific trainers
    Provides common functionality while allowing customization
    """
    
    def __init__(
        self,
        model_name: str,
        config: MAPGRPOConfig,
        tokenizer: Optional[PreTrainedTokenizer] = None
    ):
        self.config = config
        self.model_name = model_name
        
        # Initialize model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if config.fp16 else torch.float32,
            device_map="auto"
        )
        
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer = tokenizer
        
        # Reference model for KL penalty (frozen copy)
        self.ref_model = None
        if config.beta > 0:
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if config.fp16 else torch.float32,
                device_map="auto"
            )
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate
        )
        
        # Advantage calculator
        self.advantage_calculator = GroupAdvantageCalculator(config.group_size)
        
        # Metrics tracking
        self.metrics = {}
    
    @abstractmethod
    def compute_rewards(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Compute rewards for a batch of samples
        Must be implemented by each agent-specific trainer
        """
        pass
    
    @abstractmethod
    def prepare_training_data(self, data: List[Dict]) -> Dataset:
        """
        Prepare agent-specific training data
        Must be implemented by each agent-specific trainer
        """
        pass
    
    def compute_loss(
        self,
        batch: Dict[str, Any],
        return_outputs: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Compute GRPO loss for the batch
        """
        # Get model outputs
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        
        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        logits = outputs.logits
        
        # Compute log probabilities
        log_probs = self._compute_log_probs(logits, input_ids)
        
        # Get reference log probs if using KL penalty
        ref_log_probs = None
        if self.ref_model is not None:
            with torch.no_grad():
                ref_outputs = self.ref_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True
                )
                ref_logits = ref_outputs.logits
                ref_log_probs = self._compute_log_probs(ref_logits, input_ids)
        
        # Compute rewards
        rewards = self.compute_rewards(batch)
        
        # Apply KL penalty if needed
        if ref_log_probs is not None:
            rewards = self.advantage_calculator.compute_rewards_with_kl_penalty(
                rewards, log_probs, ref_log_probs, self.config.beta
            )
        
        # Compute advantages
        advantages = self.advantage_calculator.compute_advantages(rewards)
        
        # GRPO loss: -advantages * log_probs
        loss = -(advantages * log_probs).mean()
        
        # Add entropy bonus for exploration
        entropy = -(torch.exp(log_probs) * log_probs).sum(dim=-1).mean()
        loss = loss - 0.01 * entropy  # Small entropy coefficient
        
        if return_outputs:
            outputs = {
                "loss": loss,
                "rewards": rewards,
                "advantages": advantages,
                "log_probs": log_probs,
                "ref_log_probs": ref_log_probs
            }
            return loss, outputs
        
        return loss
    
    def _compute_log_probs(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute log probabilities for the generated tokens
        """
        # Shift for autoregressive
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        
        # Compute log probs
        log_probs = F.log_softmax(shift_logits, dim=-1)
        
        # Gather log probs for actual tokens
        gathered_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        return gathered_log_probs.sum(dim=-1)
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        num_epochs: Optional[int] = None
    ):
        """
        Main training loop
        """
        if num_epochs is None:
            num_epochs = self.config.num_epochs
        
        # Create dataloader
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn
        )
        
        # Training loop
        global_step = 0
        for epoch in range(num_epochs):
            self.model.train()
            epoch_losses = []
            
            for batch in train_dataloader:
                # Move batch to device
                batch = {k: v.to(self.model.device) if torch.is_tensor(v) else v 
                         for k, v in batch.items()}
                
                # Compute loss
                loss = self.compute_loss(batch)
                
                # Backward pass
                loss.backward()
                
                # Gradient accumulation
                if (global_step + 1) % self.config.gradient_accumulation_steps == 0:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        max_norm=1.0
                    )
                    
                    # Optimizer step
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                epoch_losses.append(loss.item())
                global_step += 1
                
                # Logging
                if global_step % self.config.logging_steps == 0:
                    avg_loss = np.mean(epoch_losses[-self.config.logging_steps:])
                    logger.info(f"Step {global_step}, Loss: {avg_loss:.4f}")
                    
                    if wandb.run is not None:
                        wandb.log({
                            f"{self.model_name}/loss": avg_loss,
                            f"{self.model_name}/epoch": epoch,
                            "global_step": global_step
                        })
                
                # Save checkpoint
                if global_step % self.config.save_steps == 0:
                    self.save_checkpoint(global_step)
            
            # Epoch summary
            epoch_loss = np.mean(epoch_losses)
            logger.info(f"Epoch {epoch} completed. Average loss: {epoch_loss:.4f}")
    
    def save_checkpoint(self, step: int):
        """Save model checkpoint"""
        save_path = f"{self.config.output_dir}/{self.model_name}/checkpoint-{step}"
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        logger.info(f"Checkpoint saved to {save_path}")
    
    def _collate_fn(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate function for dataloader
        Should be overridden for agent-specific needs
        """
        # Default implementation - tokenize and pad
        texts = [ex["text"] for ex in examples]
        
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": encoded["input_ids"].clone()
        }