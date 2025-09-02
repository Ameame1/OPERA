"""
Analysis-Answer Agent - Reason-Execute Module (REM) core component
Responsible for information sufficiency assessment and answer extraction
"""

from typing import Dict, Any, List, Optional
import re
import logging
from ..data.structures import AnalysisResult, Document, InformationStatus
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class AnalysisAnswerAgent(BaseAgent):
    """
    Analysis-Answer Agent for tactical execution
    Core component of the Reason-Execute Module (REM)
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "cuda",
        max_length: int = 2048,
        temperature: float = 0.1
    ):
        """Initialize Analysis-Answer Agent with 7B model"""
        super().__init__(model_name, device, max_length, temperature)
        
    def get_prompt(self, inputs: Dict[str, Any]) -> str:
        """
        Construct prompt for analysis and answer extraction
        Following the exact format from the paper
        
        Args:
            inputs: Dictionary with 'sub_question' and 'documents' keys
            
        Returns:
            Formatted prompt string
        """
        sub_question = inputs['sub_question']
        documents = inputs['documents']
        
        # Format documents
        docs_text = self._format_documents(documents)
        
        # Use the exact prompt template from the paper
        prompt = f"""You are an analysis and answering agent. Given a sub-question and retrieved documents, determine if you can answer the question and provide analysis.
Sub-question: {sub_question}
Retrieved Documents: {docs_text}
Please respond in the following JSON format:
{{
  "status": "yes" or "no",
  "answer": "extracted answer if status is yes, empty if no",
  "analysis": "explain why you can/cannot answer based on the provided documents"
}}
Key principles:
- status="yes": Documents contain sufficient information
- status="no": Documents lack necessary information
- analysis: Always explain your reasoning"""
        
        return prompt
    
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process sub-question and documents to extract answer
        
        Args:
            inputs: Dictionary with 'sub_question' and 'documents' keys
            
        Returns:
            Dictionary with 'result' key containing AnalysisResult
        """
        self.validate_inputs(inputs, ['sub_question', 'documents'])
        
        # Generate analysis
        prompt = self.get_prompt(inputs)
        response = self.generate(
            prompt,
            max_new_tokens=512,
            temperature=self.temperature
        )
        
        # Parse response
        result = self._parse_analysis(response)
        
        # Post-process answer
        if result.is_sufficient() and result.answer:
            result.answer = self._normalize_answer(result.answer)
        
        return {'result': result}
    
    def _format_documents(self, documents: List[Document]) -> str:
        """
        Format documents for prompt
        
        Args:
            documents: List of Document objects
            
        Returns:
            Formatted string
        """
        formatted = []
        for i, doc in enumerate(documents[:5]):  # Limit to top 5
            # Truncate content if too long
            content = doc.content[:300] + "..." if len(doc.content) > 300 else doc.content
            formatted.append(f"[Doc{i+1} - ID: {doc.doc_id}]\n{content}")
        
        return "\n\n".join(formatted)
    
    def _parse_analysis(self, response: str) -> AnalysisResult:
        """
        Parse model response into AnalysisResult
        Following the paper's output format exactly
        
        Args:
            response: Model response string
            
        Returns:
            AnalysisResult object
        """
        # Try to parse JSON response
        result_dict = self.parse_json_response(response)
        
        if not result_dict:
            # Fallback parsing
            logger.warning("Failed to parse JSON, using fallback parsing")
            return self._fallback_parse_analysis(response)
        
        # Determine status
        status_str = result_dict.get('status', 'no').lower()
        status = InformationStatus.SUFFICIENT if status_str == 'yes' else InformationStatus.INSUFFICIENT
        
        # Paper format only has status, answer, and analysis
        return AnalysisResult(
            status=status,
            answer=result_dict.get('answer', '') if status == InformationStatus.SUFFICIENT else None,
            analysis=result_dict.get('analysis', ''),
            supporting_docs=[],  # Not in paper format, keep empty for compatibility
            confidence=0.8 if status == InformationStatus.SUFFICIENT else 0.6
        )
    
    def _fallback_parse_analysis(self, response: str) -> AnalysisResult:
        """
        Fallback parsing when JSON parsing fails
        
        Args:
            response: Model response
            
        Returns:
            AnalysisResult object
        """
        response_lower = response.lower()
        
        # Check for yes/no status
        if 'status: yes' in response_lower or '"yes"' in response_lower:
            status = InformationStatus.SUFFICIENT
            
            # Try to extract answer
            answer = None
            answer_match = re.search(r'answer["\s:]+([^"}\n]+)', response, re.IGNORECASE)
            if answer_match:
                answer = answer_match.group(1).strip()
            
            return AnalysisResult(
                status=status,
                answer=answer,
                analysis=response,
                confidence=0.5
            )
        else:
            return AnalysisResult(
                status=InformationStatus.INSUFFICIENT,
                answer=None,
                analysis=response,
                confidence=0.5
            )
    
    def _normalize_answer(self, answer: str) -> str:
        """
        Normalize answer for evaluation
        
        Args:
            answer: Raw answer string
            
        Returns:
            Normalized answer
        """
        if not answer:
            return answer
        
        # Basic normalization
        answer = answer.strip()
        
        # Remove quotes if present
        if answer.startswith('"') and answer.endswith('"'):
            answer = answer[1:-1]
        if answer.startswith("'") and answer.endswith("'"):
            answer = answer[1:-1]
        
        # Handle common patterns
        # Remove "The answer is" prefix
        patterns = [
            r'^(?:the )?answer (?:is|to the question is)[:\s]*',
            r'^(?:based on the documents?,? )?',
            r'^(?:according to (?:the )?documents?,? )?'
        ]
        
        for pattern in patterns:
            answer = re.sub(pattern, '', answer, flags=re.IGNORECASE).strip()
        
        return answer