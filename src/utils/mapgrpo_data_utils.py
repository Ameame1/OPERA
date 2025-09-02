#!/usr/bin/env python3
"""
MAPGRPO Data Utilities
Comprehensive utilities for converting MuSiQue data to MAPGRPO training format
Includes relation conversion, quality validation, and data enhancement
"""

import json
import re
import logging
from typing import List, Dict, Any, Tuple, Optional, Set
from dataclasses import dataclass, asdict
from pathlib import Path

from .musique_relation_converter import MuSiQueRelationConverter

logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """Metrics for assessing data quality"""
    total_samples: int
    successful_conversions: int
    failed_conversions: int
    relation_conversion_rate: float
    avg_plan_length: float
    dependency_accuracy: float


class MAPGRPODataQualityValidator:
    """Validates and assesses quality of MAPGRPO training data"""
    
    def __init__(self):
        self.relation_converter = MuSiQueRelationConverter()
    
    def validate_plan_structure(self, plan_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate plan data structure and quality"""
        issues = []
        valid_samples = 0
        
        for i, sample in enumerate(plan_data):
            sample_issues = []
            
            # Check required fields
            if 'input' not in sample or 'question' not in sample['input']:
                sample_issues.append("Missing input question")
            
            if 'output' not in sample or not isinstance(sample['output'], list):
                sample_issues.append("Missing or invalid output plan")
            else:
                # Validate plan steps
                for j, step in enumerate(sample['output']):
                    if not all(key in step for key in ['subgoal_id', 'subgoal', 'dependencies']):
                        sample_issues.append(f"Step {j+1} missing required fields")
                    
                    if step.get('subgoal_id') != j + 1:
                        sample_issues.append(f"Step {j+1} has incorrect subgoal_id")
                    
                    # Check for proper entity references
                    subgoal = step.get('subgoal', '')
                    if '[entity from step' in subgoal:
                        # Validate dependency tracking
                        referenced_steps = re.findall(r'\[entity from step (\d+)\]', subgoal)
                        expected_deps = [int(step_num) for step_num in referenced_steps]
                        actual_deps = step.get('dependencies', [])
                        if set(expected_deps) != set(actual_deps):
                            sample_issues.append(f"Step {j+1} dependency mismatch")
            
            if sample_issues:
                issues.append(f"Sample {i+1}: {'; '.join(sample_issues)}")
            else:
                valid_samples += 1
        
        return {
            'total_samples': len(plan_data),
            'valid_samples': valid_samples,
            'validation_rate': valid_samples / len(plan_data) if plan_data else 0,
            'issues': issues
        }
    
    def validate_analysis_data(self, analysis_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate analysis-answer data structure and quality"""
        issues = []
        valid_samples = 0
        yes_samples = 0
        no_samples = 0
        
        for i, sample in enumerate(analysis_data):
            sample_issues = []
            
            # Check required input fields
            input_data = sample.get('input', {})
            if 'sub_question' not in input_data:
                sample_issues.append("Missing sub_question")
            if 'documents' not in input_data or not isinstance(input_data['documents'], list):
                sample_issues.append("Missing or invalid documents")
            
            # Check required output fields
            output_data = sample.get('output', {})
            if 'status' not in output_data or output_data['status'] not in ['yes', 'no']:
                sample_issues.append("Missing or invalid status")
            else:
                status = output_data['status']
                if status == 'yes':
                    yes_samples += 1
                    if not output_data.get('answer'):
                        sample_issues.append("YES status but missing answer")
                else:
                    no_samples += 1
                    if output_data.get('answer'):  # Should be empty for NO
                        sample_issues.append("NO status but has answer")
            
            if 'analysis' not in output_data:
                sample_issues.append("Missing analysis field")
            
            if sample_issues:
                issues.append(f"Sample {i+1}: {'; '.join(sample_issues)}")
            else:
                valid_samples += 1
        
        return {
            'total_samples': len(analysis_data),
            'valid_samples': valid_samples,
            'validation_rate': valid_samples / len(analysis_data) if analysis_data else 0,
            'yes_samples': yes_samples,
            'no_samples': no_samples,
            'balance_ratio': yes_samples / no_samples if no_samples > 0 else float('inf'),
            'issues': issues
        }
    
    def validate_rewrite_data(self, rewrite_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate rewrite data structure and quality"""
        issues = []
        valid_samples = 0
        strategy_distribution = {}
        
        for i, sample in enumerate(rewrite_data):
            sample_issues = []
            
            # Check required input fields
            input_data = sample.get('input', {})
            if 'original_query' not in input_data:
                sample_issues.append("Missing original_query")
            if 'failure_info' not in input_data:
                sample_issues.append("Missing failure_info")
            
            # Check required output fields
            output_data = sample.get('output', {})
            required_fields = ['rewritten_query', 'strategy', 'keywords']
            for field in required_fields:
                if field not in output_data:
                    sample_issues.append(f"Missing {field}")
            
            if 'strategy' in output_data:
                strategy = output_data['strategy']
                strategy_distribution[strategy] = strategy_distribution.get(strategy, 0) + 1
            
            if sample_issues:
                issues.append(f"Sample {i+1}: {'; '.join(sample_issues)}")
            else:
                valid_samples += 1
        
        return {
            'total_samples': len(rewrite_data),
            'valid_samples': valid_samples,
            'validation_rate': valid_samples / len(rewrite_data) if rewrite_data else 0,
            'strategy_distribution': strategy_distribution,
            'issues': issues
        }


class MAPGRPODataEnhancer:
    """Enhances MAPGRPO training data with additional features"""
    
    def __init__(self):
        self.relation_converter = MuSiQueRelationConverter()
    
    def enhance_plan_data(
        self, 
        plan_data: List[Dict[str, Any]], 
        add_complexity_metrics: bool = True
    ) -> List[Dict[str, Any]]:
        """Enhance plan data with additional metadata and metrics"""
        enhanced_data = []
        
        for sample in plan_data:
            enhanced_sample = sample.copy()
            
            if add_complexity_metrics:
                plan_steps = sample.get('output', [])
                
                # Calculate complexity metrics
                complexity_metrics = {
                    'num_steps': len(plan_steps),
                    'max_dependencies': max(len(step.get('dependencies', [])) for step in plan_steps) if plan_steps else 0,
                    'total_dependencies': sum(len(step.get('dependencies', [])) for step in plan_steps),
                    'avg_subgoal_length': sum(len(step.get('subgoal', '').split()) for step in plan_steps) / len(plan_steps) if plan_steps else 0,
                    'has_entity_references': any('[entity from step' in step.get('subgoal', '') for step in plan_steps),
                    'relation_conversions': sum(1 for step in plan_steps if self._was_relation_converted(step.get('subgoal', '')))
                }
                
                enhanced_sample['metadata']['complexity_metrics'] = complexity_metrics
            
            enhanced_data.append(enhanced_sample)
        
        return enhanced_data
    
    def _was_relation_converted(self, subgoal: str) -> bool:
        """Check if a subgoal likely came from relation conversion"""
        # Heuristic: check for question patterns typical of converted relations
        question_starters = ['Who is', 'What is', 'Which', 'Where is', 'When was', 'How']
        return any(subgoal.strip().startswith(starter) for starter in question_starters)
    
    def enhance_analysis_data(
        self, 
        analysis_data: List[Dict[str, Any]], 
        add_document_metrics: bool = True
    ) -> List[Dict[str, Any]]:
        """Enhance analysis data with document and complexity metrics"""
        enhanced_data = []
        
        for sample in analysis_data:
            enhanced_sample = sample.copy()
            
            if add_document_metrics:
                documents = sample.get('input', {}).get('documents', [])
                sub_question = sample.get('input', {}).get('sub_question', '')
                
                doc_metrics = {
                    'num_documents': len(documents),
                    'total_content_length': sum(len(doc.get('content', '')) for doc in documents),
                    'avg_content_length': sum(len(doc.get('content', '')) for doc in documents) / len(documents) if documents else 0,
                    'question_length': len(sub_question.split()),
                    'has_entity_reference': '[entity from step' in sub_question,
                    'is_relation_derived': self._was_relation_converted(sub_question)
                }
                
                enhanced_sample['metadata']['document_metrics'] = doc_metrics
            
            enhanced_data.append(enhanced_sample)
        
        return enhanced_data
    
    def create_difficulty_variants(
        self, 
        plan_data: List[Dict[str, Any]], 
        max_variants: int = 3
    ) -> List[Dict[str, Any]]:
        """Create difficulty variants of plan data for robust training"""
        variant_data = []
        
        for sample in plan_data:
            # Original sample
            variant_data.append(sample)
            
            # Create variants with different phrasings
            original_steps = sample.get('output', [])
            
            for variant_idx in range(min(max_variants, 3)):
                variant_sample = sample.copy()
                variant_steps = []
                
                for step in original_steps:
                    variant_step = step.copy()
                    subgoal = step.get('subgoal', '')
                    
                    # If this contains entity references and looks like a relation question
                    if '[entity from step' in subgoal and '?' in subgoal:
                        # Try to generate a variant using different template
                        try:
                            # Extract entity and relation pattern (simplified)
                            if 'Who is' in subgoal or 'What is' in subgoal:
                                # Attempt to create variant phrasing
                                if 'Who is the' in subgoal:
                                    variant_step['subgoal'] = subgoal.replace('Who is the', 'What is the name of the')
                                elif 'What is the' in subgoal:
                                    variant_step['subgoal'] = subgoal.replace('What is the', 'Which')
                        except:
                            pass  # Keep original if variant generation fails
                    
                    variant_steps.append(variant_step)
                
                variant_sample['output'] = variant_steps
                variant_sample['metadata']['is_variant'] = True
                variant_sample['metadata']['variant_index'] = variant_idx + 1
                variant_sample['metadata']['original_id'] = sample['metadata'].get('id', '')
                
                variant_data.append(variant_sample)
        
        return variant_data


class MAPGRPODatasetBuilder:
    """Main class for building complete MAPGRPO datasets"""
    
    def __init__(self):
        self.relation_converter = MuSiQueRelationConverter()
        self.validator = MAPGRPODataQualityValidator()
        self.enhancer = MAPGRPODataEnhancer()
    
    def build_complete_dataset(
        self, 
        musique_data_path: str,
        output_dir: str,
        num_samples: Optional[int] = None,
        enhance_data: bool = True,
        create_variants: bool = False,
        validation_split: float = 0.1
    ) -> Dict[str, Any]:
        """Build complete MAPGRPO dataset with all enhancements"""
        
        logger.info(f"Building MAPGRPO dataset from {musique_data_path}")
        
        # Import generator here to avoid circular imports
        # Add scripts to path
        import sys
        import os
        script_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'scripts')
        if script_dir not in sys.path:
            sys.path.append(script_dir)
        
        from generate_training_data_improved import MAPGRPODataGenerator
        
        # Generate base data
        generator = MAPGRPODataGenerator()
        generator.load_musique_data(musique_data_path)
        
        plan_data = generator.generate_plan_training_data(num_samples)
        analysis_data = generator.generate_analysis_training_data(num_samples)
        rewrite_data = generator.generate_rewrite_training_data(analysis_data)
        expert_demos = generator.generate_expert_demonstrations(plan_data, analysis_data)
        
        # Enhance data if requested
        if enhance_data:
            plan_data = self.enhancer.enhance_plan_data(plan_data)
            analysis_data = self.enhancer.enhance_analysis_data(analysis_data)
        
        # Create variants if requested
        if create_variants:
            plan_data = self.enhancer.create_difficulty_variants(plan_data)
        
        # Validate data quality
        plan_validation = self.validator.validate_plan_structure(plan_data)
        analysis_validation = self.validator.validate_analysis_data(analysis_data)
        rewrite_validation = self.validator.validate_rewrite_data(rewrite_data)
        
        # Split data for validation
        if validation_split > 0:
            train_plan, val_plan = self._split_data(plan_data, validation_split)
            train_analysis, val_analysis = self._split_data(analysis_data, validation_split)
            train_rewrite, val_rewrite = self._split_data(rewrite_data, validation_split)
        else:
            train_plan, val_plan = plan_data, []
            train_analysis, val_analysis = analysis_data, []
            train_rewrite, val_rewrite = rewrite_data, []
        
        # Save everything
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save training data
        self._save_data(output_path / "train_plan.json", train_plan, "plan")
        self._save_data(output_path / "train_analysis.json", train_analysis, "analysis")
        self._save_data(output_path / "train_rewrite.json", train_rewrite, "rewrite")
        
        # Save validation data if exists
        if val_plan:
            self._save_data(output_path / "val_plan.json", val_plan, "plan")
            self._save_data(output_path / "val_analysis.json", val_analysis, "analysis")
            self._save_data(output_path / "val_rewrite.json", val_rewrite, "rewrite")
        
        # Save expert demonstrations
        with open(output_path / "expert_demonstrations.json", 'w') as f:
            json.dump(expert_demos, f, indent=2)
        
        # Save quality report
        quality_report = {
            'dataset_info': {
                'source': str(musique_data_path),
                'num_samples_processed': num_samples or 'all',
                'enhanced': enhance_data,
                'variants_created': create_variants,
                'validation_split': validation_split
            },
            'data_counts': {
                'train_plan': len(train_plan),
                'train_analysis': len(train_analysis),
                'train_rewrite': len(train_rewrite),
                'val_plan': len(val_plan),
                'val_analysis': len(val_analysis),
                'val_rewrite': len(val_rewrite),
                'expert_demonstrations': {k: len(v) for k, v in expert_demos.items()}
            },
            'quality_validation': {
                'plan_validation': plan_validation,
                'analysis_validation': analysis_validation,
                'rewrite_validation': rewrite_validation
            },
            'relation_conversion_stats': {
                'supported_relations': len(self.relation_converter.get_available_relations()),
                'total_conversions': sum(1 for sample in plan_data 
                                       for step in sample.get('output', [])
                                       if self.enhancer._was_relation_converted(step.get('subgoal', '')))
            }
        }
        
        with open(output_path / "quality_report.json", 'w') as f:
            json.dump(quality_report, f, indent=2)
        
        logger.info(f"Complete MAPGRPO dataset saved to {output_dir}")
        return quality_report
    
    def _split_data(self, data: List[Dict], validation_split: float) -> Tuple[List[Dict], List[Dict]]:
        """Split data into train and validation sets"""
        if not data:
            return [], []
        
        split_idx = int(len(data) * (1 - validation_split))
        return data[:split_idx], data[split_idx:]
    
    def _save_data(self, filepath: Path, data: List[Dict], agent_type: str):
        """Save data with metadata"""
        with open(filepath, 'w') as f:
            json.dump({
                'metadata': {
                    'agent': agent_type,
                    'total_samples': len(data),
                    'created_by': 'MAPGRPODatasetBuilder'
                },
                'samples': data
            }, f, indent=2)


# Convenience functions
def convert_musique_questions_batch(
    musique_questions: List[str], 
    vary_templates: bool = True
) -> List[str]:
    """Convenience function to convert a batch of MuSiQue questions"""
    converter = MuSiQueRelationConverter()
    return converter.convert_question_list(musique_questions, vary_templates)


def validate_mapgrpo_dataset(dataset_dir: str) -> Dict[str, Any]:
    """Convenience function to validate an existing MAPGRPO dataset"""
    validator = MAPGRPODataQualityValidator()
    
    dataset_path = Path(dataset_dir)
    validation_results = {}
    
    # Validate each component if it exists
    for component in ['plan', 'analysis', 'rewrite']:
        train_file = dataset_path / f"train_{component}.json"
        if train_file.exists():
            with open(train_file, 'r') as f:
                data = json.load(f)
                samples = data.get('samples', [])
                
                if component == 'plan':
                    validation_results[f'train_{component}'] = validator.validate_plan_structure(samples)
                elif component == 'analysis':
                    validation_results[f'train_{component}'] = validator.validate_analysis_data(samples)
                elif component == 'rewrite':
                    validation_results[f'train_{component}'] = validator.validate_rewrite_data(samples)
    
    return validation_results