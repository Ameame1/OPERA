"""
Base Agent interface for OPERA-MAPGRPO
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Base class for all OPERA agents"""
    
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        max_length: int = 2048,
        temperature: float = 0.1
    ):
        """
        Initialize base agent
        
        Args:
            model_name: Name or path of the language model
            device: Device to run the model on
            max_length: Maximum sequence length
            temperature: Generation temperature
        """
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        self.max_length = max_length
        self.temperature = temperature
        
        # Initialize model and tokenizer
        self._init_model()
        
    def _init_model(self):
        """Initialize the language model and tokenizer"""
        logger.info(f"Loading model {self.model_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            pad_token='<|endoftext|>'
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device if self.device == "cuda" else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        self.model.eval()
        logger.info(f"Model loaded successfully on {self.device}")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate text using the language model
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Generation temperature (overrides default)
            system_prompt: Optional system prompt
            
        Returns:
            Generated text
        """
        if temperature is None:
            temperature = self.temperature
            
        # Construct messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        return response.strip()
    
    @abstractmethod
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process inputs and return outputs
        
        Args:
            inputs: Input dictionary
            
        Returns:
            Output dictionary
        """
        pass
    
    @abstractmethod
    def get_prompt(self, inputs: Dict[str, Any]) -> str:
        """
        Construct prompt for the specific agent
        
        Args:
            inputs: Input dictionary
            
        Returns:
            Formatted prompt string
        """
        pass
    
    def validate_inputs(self, inputs: Dict[str, Any], required_keys: List[str]):
        """
        Validate that required keys are present in inputs
        
        Args:
            inputs: Input dictionary
            required_keys: List of required keys
            
        Raises:
            ValueError: If required keys are missing
        """
        missing_keys = [key for key in required_keys if key not in inputs]
        if missing_keys:
            raise ValueError(f"Missing required input keys: {missing_keys}")
    
    def parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        Parse JSON from model response
        
        Args:
            response: Model response string
            
        Returns:
            Parsed dictionary
        """
        import json
        import re
        
        # Try to extract JSON from response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON from response: {response}")
                return {}
        else:
            logger.warning(f"No JSON found in response: {response}")
            return {}