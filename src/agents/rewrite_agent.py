"""
Rewrite Agent - Reason-Execute Module (REM) support component
Responsible for reasoning-driven query reformulation
"""

from typing import Dict, Any, List, Optional
import re
import logging
from ..data.structures import RewriteResult
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class RewriteAgent(BaseAgent):
    """
    Rewrite Agent for adaptive retrieval
    Support component of the Reason-Execute Module (REM)
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        device: str = "cuda",
        max_length: int = 1024,
        temperature: float = 0.1
    ):
        """Initialize Rewrite Agent with 3B model for efficiency"""
        super().__init__(model_name, device, max_length, temperature)
        
    def get_prompt(self, inputs: Dict[str, Any]) -> str:
        """
        Construct prompt for query rewriting
        Following the exact format from the paper
        
        Args:
            inputs: Dictionary with 'missing_info', 'original_query', and optionally 'docs_preview' keys
            
        Returns:
            Formatted prompt string
        """
        original_query = inputs['original_query']
        missing_info = inputs.get('missing_info', 'Documents lack necessary information')
        docs_preview = inputs.get('docs_preview', 'No relevant documents found')
        
        # Use the exact prompt template from the paper
        prompt = f"""You are an expert query rewriter for information retrieval.

## Rewrite Task
Original Question: {original_query}
Failure Reason: {missing_info}

## Current Documents Preview
{docs_preview}

## Instructions
1. Analyze why the current query failed to retrieve relevant information
2. Generate an improved search query using keyword expansion and synonyms
3. Focus on key entities, concepts, and alternative phrasings
4. Keep the rewritten query concise but comprehensive

## Output JSON Format
{{
  "rewritten_query": "improved search query with expanded keywords",
  "strategy": "brief explanation of rewrite approach",
  "keywords": ["key", "terms", "and", "synonyms"]
}}

Generate rewrite:"""
        
        return prompt
    
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process missing information and generate rewritten query
        
        Args:
            inputs: Dictionary with 'missing_info', 'original_query', and optionally 'docs_preview' keys
            
        Returns:
            Dictionary with 'result' key containing RewriteResult
        """
        self.validate_inputs(inputs, ['original_query'])  # missing_info and docs_preview are optional
        
        # Generate rewritten query
        prompt = self.get_prompt(inputs)
        response = self.generate(
            prompt,
            max_new_tokens=256,
            temperature=self.temperature
        )
        
        # Parse response
        result = self._parse_rewrite(response, inputs['original_query'])
        
        return {'result': result}
    
    def _parse_rewrite(self, response: str, original_query: str) -> RewriteResult:
        """
        Parse model response into RewriteResult
        
        Args:
            response: Model response string
            original_query: Original query for fallback
            
        Returns:
            RewriteResult object
        """
        # Try to parse JSON response
        result_dict = self.parse_json_response(response)
        
        if not result_dict or 'rewritten_query' not in result_dict:
            # Fallback parsing
            logger.warning("Failed to parse JSON, using fallback parsing")
            return self._fallback_parse_rewrite(response, original_query)
        
        return RewriteResult(
            rewritten_query=result_dict.get('rewritten_query', original_query),
            strategy=result_dict.get('strategy', 'keyword_expansion'),
            keywords=result_dict.get('keywords', []),
            confidence=0.8
        )
    
    def _fallback_parse_rewrite(self, response: str, original_query: str) -> RewriteResult:
        """
        Fallback parsing when JSON parsing fails
        
        Args:
            response: Model response
            original_query: Original query for fallback
            
        Returns:
            RewriteResult object
        """
        # Try to extract rewritten query
        rewritten_query = original_query
        
        # Look for common patterns
        patterns = [
            r'rewritten_query["\s:]+([^"}\n]+)',
            r'query["\s:]+([^"}\n]+)',
            r'"([^"]+)"'  # Any quoted string
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                rewritten_query = match.group(1).strip()
                break
        
        # Extract keywords
        keywords = self._extract_keywords(rewritten_query)
        
        return RewriteResult(
            rewritten_query=rewritten_query,
            strategy='keyword_expansion',
            keywords=keywords,
            confidence=0.5
        )
    
    def _extract_keywords(self, query: str) -> List[str]:
        """
        Extract keywords from query
        
        Args:
            query: Query string
            
        Returns:
            List of keywords
        """
        # Simple keyword extraction
        # Remove common words
        stopwords = {
            'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or',
            'but', 'in', 'with', 'to', 'for', 'of', 'as', 'by', 'that',
            'this', 'it', 'from', 'be', 'are', 'was', 'were', 'been'
        }
        
        # Split and filter
        words = query.lower().split()
        keywords = [
            word.strip('.,!?;:')
            for word in words
            if word.lower() not in stopwords and len(word) > 2
        ]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for keyword in keywords:
            if keyword not in seen:
                seen.add(keyword)
                unique_keywords.append(keyword)
        
        return unique_keywords[:10]  # Limit to 10 keywords