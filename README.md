# OPERA: Orchestrated Planner-Executor Reasoning Architecture

## Overview

This repository contains code implementation for OPERA system. The code includes the three-agent architecture and MAPGRPO (Multi-Agent Progressive Group Relative Policy Optimization) training framework.

## Components

This submission includes the following components:

1. **Three-Agent Architecture**: Implementation of Plan, Analysis-Answer, and Rewrite agents
2. **MAPGRPO Training**: Progressive training framework with group-based advantage computation
3. **Placeholder Mechanism**: Dependency management through `[entity from step X]` references
4. **Role-Specific Rewards**: Multi-dimensional evaluation metrics for each agent

## Package Structure

```
.
├── src/                        # Core source code
│   ├── agents/                 # Three agent implementations
│   ├── core/                   # Orchestration and state management
│   ├── data/                   # Core data structures
│   ├── training/               # MAPGRPO training framework
│   └── utils/                  # Utilities and helpers
├── data/                       # Complete training datasets
├── config/                     # System configuration
├── opera_cot_rag_baseline.py  # Baseline implementation
├── example_usage.py            # Demonstration script
└── requirements.txt            # Dependencies
```

## Core Components

### 1. Three-Agent Architecture

- **Plan Agent** (`src/agents/plan_agent.py`): Question decomposition with dependency tracking
- **Analysis-Answer Agent** (`src/agents/analysis_answer_agent.py`): Document analysis and information extraction
- **Rewrite Agent** (`src/agents/rewrite_agent.py`): Query reformulation

### 2. Orchestration System

- **Orchestrator** (`src/core/orchestrator.py`): Coordinates agent collaboration
- **State Manager** (`src/core/reasoning_state_manager.py`): Manages reasoning state
- **Trajectory Memory** (`src/core/trajectory_memory.py`): Records complete execution traces

### 3. MAPGRPO Training Framework

- **Base Framework** (`src/training/mapgrpo_base.py`): Group-based training implementation
- **Agent Trainers** (`src/training/*_trainer.py`): Agent-specific training modules
- **Reward Functions** (`src/training/reward_functions.py`): Evaluation metrics for each agent

### 4. Supporting Components

- **Data Structures** (`src/data/structures.py`): Data types and dependency management
- **Placeholder Filler** (`src/utils/placeholder_filler.py`): Information extraction utilities
- **Baseline** (`opera_cot_rag_baseline.py`): CoT+RAG baseline implementation

## Data

The `data/` directory contains training data samples used in our experiments:

### Agent Training Data
- `plan_agent_training_data.json`: 30 question decomposition examples
- `analysis_agent_training_data.json`: 30 document analysis examples (15 YES/15 NO)
- `rewrite_agent_training_data.json`: 30 query reformulation examples

### GRPO Training Components
- `expert_demonstrations.json`: 4 gold-standard examples for high rewards
- `grpo_training_trajectories.json`: Complete execution traces with rewards
- `reward_scoring_examples.json`: Detailed reward calculation breakdowns

### Evaluation & Testing
- `evaluation_test_data.json`: 10 standardized test questions with ground truth
- `initial_question_dataset.json`: 18 complex multi-hop questions
- `failure_case_examples.json`: 5 edge cases for robustness training

## Technical Details

### MAPGRPO Training

The training framework implements:

1. **Progressive Training**: Agents trained in sequential stages
2. **Group-Based Scoring**: Relative advantage computation within candidate groups
3. **Execution-Based Evaluation**: Plan quality assessed through downstream execution
4. **Agent-Specific Rewards**: Different reward functions for each agent type

### Placeholder Mechanism

The system handles multi-step dependencies through:
- Explicit dependency tracking in plan structure
- Placeholder replacement with information from previous steps
- Information extraction for context propagation

## Installation & Usage

### Installation

```bash
pip install -r requirements.txt
```

### Quick Start

```bash
python example_usage.py
```

### Programmatic Usage

```python
from opera_cot_rag_baseline import OPERACoTRAGBaseline

# Initialize system
opera = OPERACoTRAGBaseline()

# Answer complex questions
answer, trajectory = opera.answer_question(
    "What is the GDP per capita of the country where the headquarters "
    "of the company that acquired GitHub is located?"
)

print(f"Answer: {answer}")
print(f"Reasoning steps: {len(trajectory.execution_traces)}")
```

## Configuration

System configuration is provided in `config/opera_config.yaml` with parameters matching paper specifications:

- Model configurations for each agent
- MAPGRPO hyperparameters (group size, learning rates, KL coefficients)
- Training epochs and batch sizes
- Reward function weights

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.35+
- See `requirements.txt` for complete list

## Note

This is the core implementation of the OPERA system. For questions about the research, please refer to our paper.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{opera2024,
  title={OPERA: Orchestrated Planner-Executor Reasoning Architecture for Reasoning-Centric Retrieval},
  author={Anonymous},
  journal={arXiv preprint},
  year={2024}
}
```