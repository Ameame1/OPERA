# OPERA Training Data

This directory contains all training data samples for the OPERA system.

## Data Files

### Agent Training Data
- **plan_agent_training_data.json**: 30 question decomposition examples
- **analysis_agent_training_data.json**: 30 document analysis examples (15 YES/15 NO)  
- **rewrite_agent_training_data.json**: 30 query reformulation examples

### GRPO Training Components
- **expert_demonstrations.json**: 4 gold-standard examples for high rewards
- **grpo_training_trajectories.json**: Complete execution traces with rewards
- **reward_scoring_examples.json**: Detailed reward calculation breakdowns

### Evaluation & Testing
- **evaluation_test_data.json**: 10 standardized test questions with ground truth
- **initial_question_dataset.json**: 18 complex multi-hop questions
- **failure_case_examples.json**: 5 edge cases for robustness training

## Data Format

All files follow consistent JSON formatting with clear structure for easy parsing and usage in training pipelines.