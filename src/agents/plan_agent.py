"""
Plan Agent - Goal Planning Module (GPM) for OPERA-MAPGRPO
Responsible for high-level strategic decomposition with placeholder dependencies
"""

from typing import Dict, Any, List, Optional
import re
import logging
from ..data.structures import PlanStep, StrategicPlan
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class PlanAgent(BaseAgent):
    """
    Plan Agent for strategic decomposition of complex questions
    Part of the Goal Planning Module (GPM)
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "cuda",
        max_length: int = 2048,
        temperature: float = 0.1
    ):
        """Initialize Plan Agent with 7B model"""
        super().__init__(model_name, device, max_length, temperature)
        
    def get_prompt(self, inputs: Dict[str, Any]) -> str:
        """
        Construct prompt for planning - following paper specification exactly
        
        Args:
            inputs: Dictionary with 'question' key
            
        Returns:
            Formatted prompt string
        """
        question = inputs['question']
        
        # Use the exact prompt template from the paper (Table 2)
        prompt = f"""You are a strategic planning agent. Given a complex multi-hop question, decompose it into a sequence of simpler sub-goals with dependency modeling.

Question: {question}

Please generate a plan with the following JSON format:
[
  {{
    "subgoal_id": 1,
    "subgoal": "First sub-question to answer",
    "dependencies": []
  }},
  {{
    "subgoal_id": 2,
    "subgoal": "Second sub-question using [entity from step 1]",
    "dependencies": [1]
  }}
]

Requirements:
- Use placeholder mechanism: [entity from step X] for dependencies
- Each subgoal should be answerable with a small set of documents
- Maintain logical flow and clear dependencies"""
        
        return prompt
    
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process question and generate strategic plan
        
        Args:
            inputs: Dictionary with 'question' key
            
        Returns:
            Dictionary with 'plan' key containing StrategicPlan object
        """
        self.validate_inputs(inputs, ['question'])
        
        # Generate plan
        prompt = self.get_prompt(inputs)
        response = self.generate(
            prompt,
            max_new_tokens=512,
            temperature=self.temperature
        )
        
        # Parse response
        plan = self._parse_plan(response, inputs['question'])
        
        return {'plan': plan}
    
    def _parse_plan(self, response: str, original_question: str) -> StrategicPlan:
        """
        Parse model response into StrategicPlan
        Following paper format: array of subgoals
        
        Args:
            response: Model response string
            original_question: Original question
            
        Returns:
            StrategicPlan object
        """
        import json
        
        # Try to extract JSON array from response
        json_match = re.search(r'\[.*?\]', response, re.DOTALL)
        if json_match:
            try:
                subgoals = json.loads(json_match.group())
                
                # Convert to PlanStep objects
                sub_questions = []
                for subgoal in subgoals:
                    # Extract placeholders
                    placeholders = self._extract_placeholders(subgoal.get('subgoal', ''))
                    
                    step = PlanStep(
                        step_id=subgoal.get('subgoal_id', len(sub_questions) + 1),
                        sub_question=subgoal.get('subgoal', ''),
                        goal=f"Answer: {subgoal.get('subgoal', '')}",  # Paper format doesn't specify goal
                        dependencies=subgoal.get('dependencies', []),
                        expected_info_type='general',  # Paper format doesn't specify this
                        placeholders=placeholders,
                        confidence=0.8
                    )
                    sub_questions.append(step)
                
                return StrategicPlan(
                    original_question=original_question,
                    sub_questions=sub_questions,
                    reasoning="Strategic decomposition following paper format",
                    confidence_score=0.8 if len(sub_questions) > 1 else 0.5
                )
            except json.JSONDecodeError:
                pass
        
        # Fallback to object format if array parsing fails
        plan_dict = self.parse_json_response(response)
        
        if not plan_dict:
            # Final fallback
            logger.warning("Failed to parse JSON, using fallback parsing")
            return self._fallback_parse(response, original_question)
        
        # Handle object format (backward compatibility)
        sub_questions = []
        if 'sub_questions' in plan_dict:
            for sq in plan_dict.get('sub_questions', []):
                placeholders = self._extract_placeholders(sq.get('sub_question', ''))
                step = PlanStep(
                    step_id=sq.get('step_id', len(sub_questions) + 1),
                    sub_question=sq.get('sub_question', ''),
                    goal=sq.get('goal', ''),
                    dependencies=sq.get('dependencies', []),
                    expected_info_type=sq.get('expected_info_type', 'entity'),
                    placeholders=placeholders,
                    confidence=0.8
                )
                sub_questions.append(step)
        
        return StrategicPlan(
            original_question=original_question,
            sub_questions=sub_questions,
            reasoning=plan_dict.get('reasoning', ''),
            confidence_score=0.8 if len(sub_questions) > 1 else 0.5
        )
    
    def _extract_placeholders(self, text: str) -> Dict[str, int]:
        """
        Extract placeholders from text
        
        Args:
            text: Text containing placeholders
            
        Returns:
            Dictionary mapping placeholder text to dependency step
        """
        placeholders = {}
        
        # Pattern: [type from step N]
        pattern = r'\[([^\]]+) from step (\d+)\]'
        matches = re.findall(pattern, text)
        
        for info_type, step_num in matches:
            placeholders[info_type] = int(step_num)
        
        return placeholders
    
    def _fallback_parse(self, response: str, original_question: str) -> StrategicPlan:
        """
        Fallback parsing when JSON parsing fails
        
        Args:
            response: Model response
            original_question: Original question
            
        Returns:
            StrategicPlan object
        """
        sub_questions = []
        
        # Try to parse step-by-step format
        lines = response.strip().split('\n')
        step_pattern = r'(?:Step\s*)?(\d+)[:\s]*(.+)'
        
        for line in lines:
            match = re.match(step_pattern, line.strip())
            if match:
                step_id = int(match.group(1))
                question = match.group(2).strip()
                
                # Extract dependencies
                deps = []
                if 'step' in question.lower():
                    dep_matches = re.findall(r'step\s*(\d+)', question, re.IGNORECASE)
                    deps = [int(d) for d in dep_matches if int(d) < step_id]
                
                # Extract placeholders
                placeholders = self._extract_placeholders(question)
                
                step = PlanStep(
                    step_id=step_id,
                    sub_question=question,
                    goal=f"Answer sub-question {step_id}",
                    dependencies=deps,
                    placeholders=placeholders,
                    confidence=0.6
                )
                sub_questions.append(step)
        
        # If no steps found, create single step
        if not sub_questions:
            sub_questions = [
                PlanStep(
                    step_id=1,
                    sub_question=original_question,
                    goal="Answer the question directly",
                    dependencies=[],
                    confidence=0.4
                )
            ]
        
        return StrategicPlan(
            original_question=original_question,
            sub_questions=sub_questions,
            reasoning="Fallback parsing used",
            confidence_score=0.5
        )