# OPERA: Orchestrated Planner-Executor Reasoning Architecture

[![arXiv](https://img.shields.io/badge/arXiv-2024-b31b1b.svg)](https://arxiv.org/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 🎯 Overview

OPERA is a reinforcement learning-enhanced framework for reasoning-centric multi-hop retrieval that introduces a hierarchical three-agent architecture to systematically decouple strategic planning from tactical execution.

<div align="center">
  <img src="docs/figures/OPERA-Figure-1-7.png" alt="OPERA Architecture" width="100%">
  <p><i>Figure 1: OPERA's three-agent architecture with placeholder mechanism and MAPGRPO training</i></p>
</div>

## 🚀 Key Innovations

### 1️⃣ **Hierarchical Three-Agent Architecture**
- **Plan Agent**: Decomposes complex queries into logical sub-questions with dependency tracking
- **Analysis-Answer Agent**: Executes plans and extracts information from retrieved documents  
- **Rewrite Agent**: Optimizes queries for better retrieval performance

### 2️⃣ **Placeholder Mechanism**
Revolutionary dependency tracking through `[entity from step X]` references that ensures proper information flow between reasoning steps.

### 3️⃣ **MAPGRPO Training**
Multi-Agent Progressive Group Relative Policy Optimization with role-specific reward functions:

<div align="center">
  <img src="docs/figures/opera_stages.png" alt="Training Pipeline" width="80%">
  <p><i>Progressive training pipeline with stage-wise agent optimization</i></p>
</div>

## 📊 Performance

OPERA achieves state-of-the-art results on multi-hop QA benchmarks:

| Model | HotpotQA EM | HotpotQA F1 | 2WikiMQA EM | 2WikiMQA F1 | MuSiQue EM | MuSiQue F1 |
|-------|-------------|-------------|-------------|-------------|------------|------------|
| Self-RAG | 29.1 | 37.6 | 31.2 | 38.9 | 12.1 | 16.7 |
| Self-ReasoningRAG | 32.3 | 43.1 | 35.4 | 42.7 | 16.8 | 20.3 |
| IRCoT | 36.2 | 44.8 | 38.1 | 45.2 | - | 19.1 |
| DSP + CoT | 36.7 | 45.1 | - | 44.5 | 18.5 | 21.9 |
| ReAct | 35.1 | - | 33.8 | - | - | - |
| MCS-RAG | 37.4 | 45.8 | 35.2 | 43.5 | 17.1 | 20.9 |
| **OPERA (Ours)** | **42.6** | **53.7** | **44.8** | **54.2** | **23.4** | **28.6** |

## 🛠️ Installation

### Requirements
- Python 3.8+
- PyTorch 2.0+
- Transformers 4.35+

### Setup
```bash
# Clone the repository
git clone https://github.com/Ameame1/OPERA.git
cd OPERA

# Install dependencies
pip install -r requirements.txt

# Download model checkpoints (if available)
python download_models.py
```

## 🚦 Quick Start

### Basic Usage

```python
from opera_cot_rag_baseline import OPERACoTRAGBaseline

# Initialize OPERA system
opera = OPERACoTRAGBaseline()

# Complex multi-hop question
question = "What is the GDP per capita of the country where the headquarters of the company that acquired GitHub is located?"

# Get answer with full reasoning trajectory
answer, trajectory = opera.answer_question(question)

print(f"Answer: {answer}")
print(f"Reasoning steps: {len(trajectory.execution_traces)}")
```

### Example Output

```python
# Plan Agent Output
Plan:
1. Which company acquired GitHub?
2. Where is [company from step 1] headquartered?
3. What country is [location from step 2] in?
4. What is the GDP per capita of [country from step 3]?

# Analysis-Answer Agent Execution
Step 1: Microsoft acquired GitHub
Step 2: Microsoft is headquartered in Redmond, Washington
Step 3: Redmond is in the United States
Step 4: US GDP per capita is approximately $76,000

# Final Answer
$76,000
```

## 📁 Project Structure

```
OPERA/
├── src/
│   ├── agents/                 # Three agent implementations
│   │   ├── plan_agent.py       # Strategic planning
│   │   ├── analysis_answer_agent.py  # Execution & analysis
│   │   └── rewrite_agent.py    # Query optimization
│   ├── core/
│   │   ├── orchestrator.py     # Agent coordination
│   │   └── trajectory_memory.py # Execution tracking
│   ├── training/
│   │   ├── mapgrpo_base.py     # MAPGRPO framework
│   │   └── reward_functions.py # Multi-dimensional rewards
│   └── utils/
│       └── placeholder_filler.py # Dependency resolution
├── data/                        # Training datasets
├── config/
│   └── opera_config.yaml       # System configuration
├── docs/                        # Documentation & figures
└── example_usage.py            # Demo script
```

## 🎓 Training

### MAPGRPO Training Pipeline

```python
from src.training.mapgrpo_trainer import MAPGRPOTrainer

# Initialize trainer
trainer = MAPGRPOTrainer(
    group_size=5,
    kl_coeff=0.01,
    learning_rate=1e-5
)

# Stage 1: Train Plan Agent
trainer.train_plan_agent(epochs=3)

# Stage 2: Train Analysis Agent (Plan Agent frozen)
trainer.train_analysis_agent(epochs=3)

# Stage 3: Train Rewrite Agent (Others frozen)
trainer.train_rewrite_agent(epochs=3)
```

### Reward Functions

Each agent has specialized reward components:

- **Plan Agent**: Correctness, coherence, efficiency
- **Analysis Agent**: Answer accuracy, evidence quality
- **Rewrite Agent**: Query clarity, retrieval effectiveness

## 📈 Evaluation

```bash
# Run evaluation on HotpotQA
python evaluate.py --dataset hotpotqa --split test

# Run evaluation on MuSiQue
python evaluate.py --dataset musique --split test

# Generate detailed analysis
python analyze_results.py --input results/opera_predictions.json
```

## 🔧 Configuration

Modify `config/opera_config.yaml` to customize:

```yaml
agents:
  plan_agent:
    model: "Qwen/Qwen2.5-7B-Instruct"
    temperature: 0.7
  analysis_agent:
    model: "Qwen/Qwen2.5-7B-Instruct"
    temperature: 0.5
  rewrite_agent:
    model: "Qwen/Qwen2.5-3B-Instruct"
    temperature: 0.3

training:
  mapgrpo:
    group_size: 5
    kl_coefficient: 0.01
    learning_rate: 1e-5
    epochs: 3
```

## 📚 Documentation

For detailed documentation, visit our [GitHub Pages](https://ameame1.github.io/OPERA/)

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📝 Citation

If you use OPERA in your research, please cite:

```bibtex
@article{opera2024,
  title={OPERA: Orchestrated Planner-Executor Reasoning Architecture for Reasoning-Centric Retrieval},
  author={Anonymous},
  journal={arXiv preprint},
  year={2024}
}
```

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

We thank the open-source community for their valuable contributions and the anonymous reviewers for their insightful feedback.

## 📧 Contact

For questions and discussions, please open an issue on GitHub.

---

<div align="center">
  <b>🌟 Star us on GitHub if you find this work helpful!</b>
</div>