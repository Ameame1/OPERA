#!/usr/bin/env python3
"""
MuSiQue Relation Converter
Converts MuSiQue's ">>" notation to natural language questions for MAPGRPO training
Handles entity references (#1, #2 etc.) and maintains grammatical correctness
"""

import re
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class MuSiQueRelationConverter:
    """Converts MuSiQue relation notation to natural language questions"""
    
    def __init__(self):
        # Comprehensive mapping of relation types to natural language templates
        self.relation_templates = {
            # Basic entity properties
            "performer": [
                "Who performed {entity}?",
                "Who is the performer of {entity}?",
                "Which artist performed {entity}?"
            ],
            "distributed by": [
                "Which company distributed {entity}?",
                "Who distributed {entity}?",
                "What company was responsible for distributing {entity}?"
            ],
            "owned by": [
                "Who owns {entity}?",
                "Which company owns {entity}?",
                "What organization owns {entity}?"
            ],
            "spouse": [
                "Who is the spouse of {entity}?",
                "Who is {entity} married to?",
                "Who did {entity} marry?"
            ],
            "headquartered": [
                "Where is {entity} headquartered?",
                "What is the headquarters location of {entity}?",
                "In which city is {entity} headquartered?"
            ],
            "located in the administrative territorial entity": [
                "In which administrative region is {entity} located?",
                "What administrative territory contains {entity}?",
                "Which administrative division is {entity} part of?"
            ],
            
            # Geographic relations
            "capital": [
                "What is the capital of {entity}?",
                "Which city is the capital of {entity}?"
            ],
            "continent": [
                "Which continent is {entity} on?",
                "What continent does {entity} belong to?"
            ],
            "country": [
                "Which country is {entity} in?",
                "What country does {entity} belong to?"
            ],
            "located in": [
                "Where is {entity} located?",
                "In which place is {entity} located?"
            ],
            "part of": [
                "What is {entity} part of?",
                "Which larger entity includes {entity}?"
            ],
            
            # Personal relations
            "father": [
                "Who is the father of {entity}?",
                "Who is {entity}'s father?"
            ],
            "mother": [
                "Who is the mother of {entity}?",
                "Who is {entity}'s mother?"
            ],
            "child": [
                "Who is the child of {entity}?",
                "Who are {entity}'s children?"
            ],
            "sibling": [
                "Who is the sibling of {entity}?",
                "Who are {entity}'s siblings?"
            ],
            "parent": [
                "Who is the parent of {entity}?",
                "Who are {entity}'s parents?"
            ],
            
            # Professional relations
            "employer": [
                "Who is the employer of {entity}?",
                "Which company employs {entity}?"
            ],
            "occupation": [
                "What is the occupation of {entity}?",
                "What is {entity}'s profession?"
            ],
            "position held": [
                "What position does {entity} hold?",
                "What is {entity}'s role?"
            ],
            "member of": [
                "What organization is {entity} a member of?",
                "Which group does {entity} belong to?"
            ],
            
            # Creative works
            "author": [
                "Who is the author of {entity}?",
                "Who wrote {entity}?"
            ],
            "composer": [
                "Who composed {entity}?",
                "Who is the composer of {entity}?"
            ],
            "director": [
                "Who directed {entity}?",
                "Who is the director of {entity}?"
            ],
            "producer": [
                "Who produced {entity}?",
                "Who is the producer of {entity}?"
            ],
            "publisher": [
                "Who published {entity}?",
                "Which company published {entity}?"
            ],
            "record label": [
                "What record label released {entity}?",
                "Which record label is {entity} associated with?"
            ],
            
            # Temporal relations
            "publication date": [
                "When was {entity} published?",
                "What is the publication date of {entity}?"
            ],
            "inception": [
                "When was {entity} founded?",
                "When did {entity} begin?"
            ],
            "end date": [
                "When did {entity} end?",
                "What was the end date of {entity}?"
            ],
            "date of birth": [
                "When was {entity} born?",
                "What is {entity}'s birth date?"
            ],
            "date of death": [
                "When did {entity} die?",
                "What is {entity}'s death date?"
            ],
            
            # Educational relations
            "educated at": [
                "Where was {entity} educated?",
                "Which school did {entity} attend?"
            ],
            "alma mater": [
                "What is {entity}'s alma mater?",
                "Where did {entity} graduate from?"
            ],
            
            # Awards and recognition
            "award received": [
                "What awards has {entity} received?",
                "Which awards did {entity} win?"
            ],
            "nominated for": [
                "What was {entity} nominated for?",
                "Which awards was {entity} nominated for?"
            ],
            
            # Physical properties
            "height": [
                "What is the height of {entity}?",
                "How tall is {entity}?"
            ],
            "weight": [
                "What is the weight of {entity}?",
                "How much does {entity} weigh?"
            ],
            "color": [
                "What color is {entity}?",
                "What is the color of {entity}?"
            ],
            
            # Sports relations
            "sport": [
                "What sport does {entity} play?",
                "Which sport is {entity} associated with?"
            ],
            "team": [
                "Which team does {entity} play for?",
                "What team is {entity} on?"
            ],
            "league": [
                "Which league does {entity} play in?",
                "What league is {entity} part of?"
            ],
            
            # Generic fallback
            "default": [
                "What is the {relation} of {entity}?",
                "Which {relation} is associated with {entity}?"
            ]
        }
        
        # Mapping for common abbreviations and variations
        self.relation_aliases = {
            "place of birth": "birth place",
            "place of death": "death place",
            "place of origin": "origin",
            "work location": "workplace",
            "headquarters location": "headquartered",
            "administrative territorial entity": "located in the administrative territorial entity",
            "follows": "preceded by",
            "followed by": "successor",
        }
    
    def convert_relation_to_question(
        self, 
        entity: str, 
        relation: str, 
        template_index: int = 0
    ) -> str:
        """
        Convert a single entity-relation pair to natural language question
        
        Args:
            entity: The entity name (could be a placeholder like "[entity from step 1]")
            relation: The relation type (e.g., "performer", "distributed by")
            template_index: Which template variant to use (0 for primary)
        
        Returns:
            Natural language question string
        """
        # Normalize relation
        relation = relation.lower().strip()
        
        # Check for aliases
        if relation in self.relation_aliases:
            relation = self.relation_aliases[relation]
        
        # Get appropriate templates
        if relation in self.relation_templates:
            templates = self.relation_templates[relation]
        else:
            logger.warning(f"Unknown relation '{relation}', using default template")
            templates = self.relation_templates["default"]
        
        # Select template (cycle if index exceeds available templates)
        template = templates[template_index % len(templates)]
        
        # Handle default template that needs relation insertion
        if "{relation}" in template:
            question = template.format(entity=entity, relation=relation)
        else:
            question = template.format(entity=entity)
        
        return question
    
    def parse_musique_notation(self, notation: str) -> Tuple[str, str]:
        """
        Parse MuSiQue's ">>" notation to extract entity and relation
        
        Args:
            notation: String like "Green >> performer" or "#1 >> spouse"
        
        Returns:
            Tuple of (entity, relation)
        """
        if " >> " not in notation:
            raise ValueError(f"Invalid MuSiQue notation: {notation}")
        
        parts = notation.split(" >> ", 1)
        entity = parts[0].strip()
        relation = parts[1].strip()
        
        return entity, relation
    
    def convert_entity_reference(self, entity: str) -> str:
        """
        Convert entity references from MuSiQue format to OPERA format
        
        Args:
            entity: Entity string that may contain #N references
        
        Returns:
            Converted entity string with proper placeholder format
        """
        # Convert #N to [entity from step N]
        def replace_ref(match):
            step_num = match.group(1)
            return f"[entity from step {step_num}]"
        
        # Pattern to match #1, #2, etc.
        converted = re.sub(r'#(\d+)', replace_ref, entity)
        
        return converted
    
    def convert_musique_question(
        self, 
        musique_notation: str, 
        template_index: int = 0
    ) -> str:
        """
        Convert full MuSiQue notation to natural language question
        
        Args:
            musique_notation: Full notation like "Green >> performer" or "#1 >> spouse"
            template_index: Which template variant to use
        
        Returns:
            Natural language question
        """
        try:
            entity, relation = self.parse_musique_notation(musique_notation)
            
            # Convert entity references
            converted_entity = self.convert_entity_reference(entity)
            
            # Generate question
            question = self.convert_relation_to_question(
                converted_entity, 
                relation, 
                template_index
            )
            
            return question
            
        except Exception as e:
            logger.error(f"Error converting '{musique_notation}': {e}")
            # Fallback to generic template
            return f"What information can you find about {musique_notation}?"
    
    def convert_question_list(
        self, 
        musique_notations: List[str], 
        vary_templates: bool = True
    ) -> List[str]:
        """
        Convert a list of MuSiQue notations to natural language questions
        
        Args:
            musique_notations: List of notation strings
            vary_templates: Whether to use different template variants for variety
        
        Returns:
            List of natural language questions
        """
        questions = []
        
        for i, notation in enumerate(musique_notations):
            template_index = i % 3 if vary_templates else 0  # Cycle through first 3 templates
            question = self.convert_musique_question(notation, template_index)
            questions.append(question)
        
        return questions
    
    def get_available_relations(self) -> List[str]:
        """Get list of all supported relation types"""
        return list(self.relation_templates.keys())
    
    def add_custom_relation(
        self, 
        relation: str, 
        templates: List[str]
    ) -> None:
        """
        Add a custom relation type with templates
        
        Args:
            relation: The relation name
            templates: List of template strings with {entity} placeholder
        """
        self.relation_templates[relation] = templates
        logger.info(f"Added custom relation '{relation}' with {len(templates)} templates")


def create_enhanced_plan_converter(original_converter_func):
    """
    Enhance the existing plan conversion function with relation conversion
    
    Args:
        original_converter_func: Original _convert_musique_decomposition_to_plan function
    
    Returns:
        Enhanced converter function
    """
    converter = MuSiQueRelationConverter()
    
    def enhanced_converter(question_decomposition: List[Dict]) -> List[Dict]:
        """Enhanced plan converter with natural language questions"""
        plan_steps = original_converter_func(question_decomposition)
        
        # Convert any remaining ">>" notation to natural language
        for step in plan_steps:
            if " >> " in step['subgoal']:
                try:
                    step['subgoal'] = converter.convert_musique_question(step['subgoal'])
                except Exception as e:
                    logger.warning(f"Could not convert subgoal '{step['subgoal']}': {e}")
        
        return plan_steps
    
    return enhanced_converter


# Example usage and testing
if __name__ == "__main__":
    # Test the converter
    converter = MuSiQueRelationConverter()
    
    # Test cases from your examples
    test_cases = [
        "Green >> performer",
        "UHF >> distributed by", 
        "Ciudad Deportiva >> owned by",
        "#1 >> spouse",
        "#1 >> headquartered",
        "#1 >> located in the administrative territorial entity"
    ]
    
    print("MuSiQue Relation Converter Test Results:")
    print("=" * 60)
    
    for notation in test_cases:
        question = converter.convert_musique_question(notation)
        print(f"Input:  {notation}")
        print(f"Output: {question}")
        print("-" * 40)
    
    # Test with template variations
    print("\nTemplate Variations for 'Green >> performer':")
    for i in range(3):
        question = converter.convert_musique_question("Green >> performer", i)
        print(f"Variant {i+1}: {question}")
    
    # Show available relations
    print(f"\nSupported Relations ({len(converter.get_available_relations())}):")
    for relation in sorted(converter.get_available_relations())[:10]:
        print(f"  - {relation}")
    print("  ...")