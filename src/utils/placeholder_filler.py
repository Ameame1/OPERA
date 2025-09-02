"""
Smart Placeholder Filler - Intelligently fill placeholders with key information
Part of the framework optimization (not an Agent)
"""

import re
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class SmartPlaceholderFiller:
    """
    Intelligently fill placeholders with extracted key information
    Instead of filling with entire answer sentences
    """
    
    def __init__(self):
        # Common patterns to extract key information
        self.extraction_patterns = {
            'entity': [
                # Named entities (capitalized words)
                r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
                # Quoted entities
                r'"([^"]+)"',
                r"'([^']+)'",
            ],
            'date': [
                # Years
                r'\b((?:19|20)\d{2})\b',
                # Full dates
                r'\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b',
                # Numeric dates
                r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b',
            ],
            'location': [
                # Places with prepositions
                r'\b(?:in|at|from|to)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
                # Country/city patterns
                r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),\s+[A-Z][a-z]+\b',
            ],
            'number': [
                # Numbers with units
                r'\b(\d+\.?\d*)\s*(?:million|billion|thousand|hundred)?\b',
                # Plain numbers
                r'\b(\d+\.?\d*)\b',
            ]
        }
    
    def fill_placeholders(self, template: str, collected_facts: Dict[int, str],
                         key_information: Optional[Dict[int, Dict]] = None) -> str:
        """
        Fill placeholders in template with smart extraction
        
        Args:
            template: String with placeholders like [entity from step 1]
            collected_facts: Dict mapping step_id to answer text
            key_information: Optional pre-extracted key information
            
        Returns:
            String with placeholders filled
        """
        # Find all placeholders in the template
        placeholder_pattern = r'\[([^]]+)\s+from\s+step\s+(\d+)\]'
        placeholders = re.findall(placeholder_pattern, template, re.IGNORECASE)
        
        filled_template = template
        for info_type, step_id in placeholders:
            step_id = int(step_id)
            
            if step_id not in collected_facts:
                logger.warning(f"No answer found for step {step_id}")
                continue
            
            # Get the answer for this step
            answer = collected_facts[step_id]
            
            # Extract the appropriate information based on type
            extracted_value = self._extract_by_type(answer, info_type.lower(), key_information, step_id)
            
            # Replace the placeholder
            placeholder = f'[{info_type} from step {step_id}]'
            filled_template = filled_template.replace(placeholder, extracted_value)
            
            logger.info(f"Filled placeholder '{placeholder}' with '{extracted_value}'")
        
        return filled_template
    
    def _extract_by_type(self, answer: str, info_type: str, 
                        key_information: Optional[Dict[int, Dict]], 
                        step_id: int) -> str:
        """
        Extract specific type of information from answer
        
        Args:
            answer: The full answer text
            info_type: Type of information to extract (entity, date, location, etc.)
            key_information: Optional pre-extracted information
            step_id: Step ID for cache lookup
            
        Returns:
            Extracted value
        """
        # First check if we have pre-extracted key information
        if key_information and step_id in key_information:
            cached_info = key_information[step_id]
            if 'key_phrase' in cached_info:
                return cached_info['key_phrase']
            if 'entities' in cached_info and cached_info['entities']:
                return cached_info['entities'][0]
        
        # Clean the answer
        answer = answer.strip()
        
        # Handle specific info types
        if info_type in ['entity', 'person', 'organization', 'thing']:
            return self._extract_entity(answer)
        elif info_type in ['date', 'year', 'time']:
            return self._extract_date(answer)
        elif info_type in ['location', 'place', 'country', 'city']:
            return self._extract_location(answer)
        elif info_type in ['number', 'amount', 'count']:
            return self._extract_number(answer)
        else:
            # For unknown types, try to extract the most relevant part
            return self._extract_key_phrase(answer)
    
    def _extract_entity(self, text: str) -> str:
        """Extract the main entity from text"""
        # Remove common prefixes
        text = re.sub(r'^(The|A|An)\s+', '', text, flags=re.IGNORECASE)
        
        # Try each entity pattern
        for pattern in self.extraction_patterns['entity']:
            matches = re.findall(pattern, text)
            if matches:
                # Return the longest match (usually most complete)
                return max(matches, key=len)
        
        # Fallback: if text is short, return as is
        if len(text.split()) <= 4:
            return text
        
        # Otherwise, try to extract the subject of the sentence
        if ' is ' in text:
            return text.split(' is ')[0].strip()
        elif ' was ' in text:
            return text.split(' was ')[0].strip()
        
        # Last resort: return first few words
        words = text.split()
        return ' '.join(words[:3])
    
    def _extract_date(self, text: str) -> str:
        """Extract date/year from text"""
        for pattern in self.extraction_patterns['date']:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        
        # Fallback: look for any 4-digit number (likely a year)
        year_match = re.search(r'\b(\d{4})\b', text)
        if year_match:
            return year_match.group(1)
        
        return text.strip()
    
    def _extract_location(self, text: str) -> str:
        """Extract location from text"""
        for pattern in self.extraction_patterns['location']:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        
        # Fallback: look for capitalized words
        capitals = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        if capitals:
            # Return the last one (often the most specific location)
            return capitals[-1]
        
        return text.strip()
    
    def _extract_number(self, text: str) -> str:
        """Extract number from text"""
        for pattern in self.extraction_patterns['number']:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        
        return text.strip()
    
    def _extract_key_phrase(self, text: str) -> str:
        """Extract the most relevant phrase from text"""
        # Remove common filler words
        text = re.sub(r'\b(the|a|an|is|was|were|are|been)\b', '', text, flags=re.IGNORECASE)
        text = ' '.join(text.split())  # Clean up extra spaces
        
        # If text is already short, return as is
        if len(text.split()) <= 5:
            return text.strip()
        
        # Try to find the main clause
        if ',' in text:
            # Return the first clause
            return text.split(',')[0].strip()
        elif '.' in text:
            # Return the first sentence
            return text.split('.')[0].strip()
        
        # Return first few words
        words = text.split()
        return ' '.join(words[:5])