"""
Query Variant Generator - Generate multiple query variants for better retrieval
Part of the framework optimization (not an Agent)
"""

import re
from typing import List, Set
import string


class QueryVariantGenerator:
    """
    Generate query variants to improve retrieval success rate
    Inspired by V1's multi-query strategy which contributed to 68% accuracy
    """
    
    def __init__(self):
        # Common stop words to remove for keyword queries
        self.stop_words = {
            'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or',
            'but', 'in', 'with', 'to', 'for', 'of', 'as', 'by', 'that',
            'this', 'it', 'from', 'be', 'are', 'was', 'were', 'been',
            'what', 'who', 'where', 'when', 'why', 'how'
        }
        
    def generate_variants(self, query: str) -> List[str]:
        """
        Generate up to 3 query variants for better coverage
        
        Args:
            query: Original query
            
        Returns:
            List of query variants (including original)
        """
        variants = [query]  # Always include original
        
        # Strategy 1: Entity-focused query
        entity_query = self._extract_entity_query(query)
        if entity_query and entity_query != query:
            variants.append(entity_query)
        
        # Strategy 2: Statement transformation
        statement_query = self._to_statement(query)
        if statement_query and statement_query != query and statement_query not in variants:
            variants.append(statement_query)
        
        # Strategy 3: Keyword query
        keyword_query = self._to_keywords(query)
        if keyword_query and keyword_query != query and keyword_query not in variants:
            variants.append(keyword_query)
        
        # Ensure we don't have too many variants
        return variants[:3]
    
    def _extract_entity_query(self, query: str) -> str:
        """
        Extract key entities and create focused query
        
        Examples:
        "Who invented the telephone?" -> "invented telephone"
        "What is the capital of France?" -> "capital France"
        """
        # Remove question words
        query_lower = query.lower()
        for qword in ['what', 'who', 'where', 'when', 'why', 'how', 'which']:
            if query_lower.startswith(qword):
                query = query[len(qword):].strip()
                break
        
        # Remove common verbs for cleaner entity extraction
        query = re.sub(r'\b(is|was|are|were|been|be)\b', '', query, flags=re.IGNORECASE)
        
        # Clean up
        query = ' '.join(query.split())
        query = query.strip('?.,')
        
        return query.strip()
    
    def _to_statement(self, query: str) -> str:
        """
        Convert question to statement form
        
        Examples:
        "Who invented the telephone?" -> "invented the telephone"
        "What is the capital of France?" -> "the capital of France"
        "Where was X born?" -> "X was born in"
        """
        query_lower = query.lower().strip('?')
        
        # Pattern-based transformations
        patterns = [
            (r'^who\s+(.+)', r'\1'),
            (r'^what\s+is\s+the\s+(.+)', r'the \1'),
            (r'^what\s+(.+)', r'\1'),
            (r'^where\s+was\s+(.+)\s+born', r'\1 was born in'),
            (r'^where\s+is\s+(.+)', r'\1 is located'),
            (r'^when\s+was\s+(.+)', r'\1 was'),
            (r'^when\s+did\s+(.+)', r'\1'),
            (r'^why\s+(.+)', r'\1 because'),
            (r'^how\s+(.+)', r'\1 by'),
        ]
        
        for pattern, replacement in patterns:
            match = re.match(pattern, query_lower)
            if match:
                return re.sub(pattern, replacement, query_lower).strip()
        
        # Default: remove question mark
        return query.strip('?')
    
    def _to_keywords(self, query: str) -> str:
        """
        Extract keywords for search
        
        Examples:
        "Who invented the telephone?" -> "invented telephone"
        "What is the capital of France?" -> "capital France"
        """
        # Remove punctuation
        translator = str.maketrans('', '', string.punctuation)
        query = query.translate(translator)
        
        # Split into words
        words = query.lower().split()
        
        # Filter out stop words
        keywords = [w for w in words if w not in self.stop_words and len(w) > 2]
        
        # Keep only the most important keywords (up to 4)
        if len(keywords) > 4:
            # Prioritize longer words (often more specific)
            keywords = sorted(keywords, key=len, reverse=True)[:4]
        
        return ' '.join(keywords)
    
    def _clean_query(self, query: str) -> str:
        """Clean up query by removing extra spaces and punctuation"""
        query = ' '.join(query.split())
        query = query.strip('?.,;:')
        return query