# Visual Persuasion: What Influences Decisions of Vision-Language Models?

![Python version](https://img.shields.io/badge/python-3.10-blue)
![Package version](https://img.shields.io/badge/version-0.1.0-green)
![GitHub license](https://img.shields.io/github/license/PapayaResearch/visual-nudging)

> [!NOTE]
> Code for the paper **[Visual Persuasion: What Influences Decisions of Vision-Language Models?]([PLACEHOLDER])**.

Decisions that once relied on human visual judgment are increasingly delegated to AI agents powered by vision-language models. These agents choose which products to purchase, which candidates to hire, which properties to consider, and which hotels to book, all based on visual information. Current evaluations of these models focus almost entirely on accuracy: can they identify objects, answer questions, follow instructions? But accuracy tells only part of the story. When agents make preference-based decisions on our behalf, we need to understand what visual features actually drive their choices and whether these align with human preferences. We introduce **Visual Persuasion**, a framework for systematically probing and optimizing visual influence on AI decision-making. Through iterative image refinement and competitive evaluation, we reveal how targeted visual modifications can substantially shift model preferences across realistic choice scenarios in personnel selection, product marketing, real estate, and hospitality.

## Features

- 🎨 Multiple optimization strategies: Competitive Visual Prompt Optimization (CVPO), VisualTextGrad (VTG), VisualFeedbackDescent (VFD), Formula-based
- 🤖 Support for multiple vision-language models through LiteLLM (e.g. GPT, Claude, Gemini, Qwen, Llama)
- 🎯 Realistic decision-making environments across hiring, e-commerce, real estate, and hospitality domains
- 📊 Comprehensive evaluation methods: pairwise comparison, solo assessment, chain analysis, and cross-method benchmarking
- 🔍 Automated interpretability pipeline for extracting visual patterns from optimized images
- ⚙️ Hydra-based configuration system for reproducible experiments

## Prerequisites

- Python 3.10 or higher
- API keys for LLM providers

## Installation

### 1. Create Python Environment

```bash
conda create -n vnudging python=3.10
conda activate vnudging
```

### 2. Install Dependencies

Using pip:
```bash
pip install -e .
```

Or using `uv` (recommended for faster installs):
```bash
pip install uv
uv pip install -e .
```

### 3. Configure Environment Variables

Create a `.env` file in both `nudging/` and `setup/` directories with your API keys:

> [!IMPORTANT]
> You must configure API keys for at least one LLM provider to run the framework.

```bash
# LLM API Keys (add only the ones you plan to use)
OPENAI_API_KEY="<YOUR_OPENAI_KEY>"
ANTHROPIC_API_KEY="<YOUR_ANTHROPIC_KEY>"
GEMINI_API_KEY="<YOUR_GEMINI_KEY>"

# AWS Bedrock Configuration (if using AWS models)
AWS_REGION_NAME="<AWS_REGION>"
AWS_ACCESS_KEY_ID="<AWS_ACCESS_KEY_ID>"
AWS_SECRET_ACCESS_KEY="<YOUR_AWS_SECRET_KEY>"
```

## Quick Start

### 1. Preprocess Dataset

Before running optimization, prepare your image dataset:

```bash
cd setup
python main.py \
  general.src_dir=./data/your_dataset \
  general.dst_dir=../nudging/data/processed \
  dataset.name=your_dataset \
  general.max_workers=8
```

### 2. Run Optimization

Optimize images using the competition strategy:

```bash
cd nudging
python run_optimization.py \
  strategy=competition \
  task=people \
  general.data_dir=./data/processed/your_dataset \
  general.max_workers=8
```

### 3. Evaluate Results

Evaluate the optimized images using pair-wise comparison:

```bash
cd nudging
python run_evaluation.py \
  evaluate=pairs \
  general.data_dir=./results/competition/TIMESTAMP \
  general.max_workers=8
```

Results are saved in `nudging/results/` with timestamps for easy tracking.

## Optimization Strategies

`Visual Persuasion` supports multiple optimization strategies:

### CVPO (Competitive Visual Prompt Optimization)

A novel visual prompt optimization algorithm that uses a competitive selection process with judge-based evaluation to iteratively refine prompts.

```bash
python run_optimization.py strategy=competition task=people
```

### VTG (VisualTextGrad)

An adaptation of the text-based gradient method from [TextGrad](https://arxiv.org/abs/2406.07496) to the visual domain by treating the image editing prompt as a differentiable textual object.

```bash
python run_optimization.py strategy=textgrad task=hotels
```

### VFD (Visual Feedback Descent)

An optimization method based on the [Feedback Descent](https://arxiv.org/abs/2511.07919) method, following a proposal-and-evaluation loop.

```bash
python run_optimization.py strategy=feedback-descent task=houses
```

### Formula-based

A simple optimization strategy that applies a fixed formula to modify images based on domain-specific heuristics.

```bash
python run_optimization.py strategy=formula task=products
```

## Evaluation Methods

`Visual Persuasion` provides multiple evaluation approaches to assess optimized images:

### Pair-wise Evaluation

Compare images head-to-head using LLM judges:

```bash
python run_evaluation.py \
  evaluate=pairs \
  general.data_dir=./results/competition/TIMESTAMP
```

### Solo Evaluation

Evaluate individual images in isolation:

```bash
python run_evaluation.py \
  evaluate=solo \
  general.data_dir=./results/competition/TIMESTAMP
```

### Chain Evaluation

Evaluate sequences of refinements to track optimization trajectory:

```bash
python run_evaluation.py \
  evaluate=chain \
  general.data_dir=./results/competition/TIMESTAMP
```

### Automated Interpretability

Generate natural language interpretations of visual differences:

```bash
python run_evaluation.py \
  evaluate=autointerp \
  general.data_dir=./results/competition/TIMESTAMP
```

### Method Comparison

Compare different optimization strategies:

```bash
python run_evaluation.py \
  evaluate=methods \
  general.data_dir=./results/
```

### Mitigation

Evaluate the effectiveness of mitigation strategies in reducing visual influence:

```bash
python run_evaluation.py \
  evaluate=solutions \
  general.data_dir=./results/competition/TIMESTAMP
```

## Advanced Usage & Customization

### Project Structure

```
visual-nudging/
├── nudging/                    # Main optimization and evaluation modules
│   ├── run_optimization.py     # Entry point for optimization
│   ├── run_evaluation.py       # Entry point for evaluation
│   ├── competition.py          # Competition-based optimization
│   ├── formula.py              # Formula-based optimization
│   ├── baseline_*.py           # Baseline optimization methods
│   ├── evaluate_*.py           # Evaluation implementations
│   ├── schema.py               # Pydantic schemas for structured output
│   ├── config.py               # Hydra configuration dataclasses
│   ├── conf/                   # Hydra configuration files
│   │   ├── config.yaml         # Main configuration
│   │   ├── task/               # Task-specific configs
│   │   ├── strategy/           # Optimization strategy configs
│   │   ├── evaluate/           # Evaluation method configs
│   │   ├── llm/                # LLM provider configs
│   │   └── editor/             # Image editor configs
│   ├── data/                   # Input datasets and processed data
│   ├── results/                # Optimization results
│   ├── outputs/                # Hydra outputs
│   └── logs/                   # Execution logs
│
├── setup/                      # Dataset preprocessing
│   ├── main.py                 # Preprocessing entry point
│   ├── enhance.py              # Image enhancement
│   ├── strategy.py             # Sampling strategies
│   ├── config.py               # Hydra configuration dataclasses
│   └── conf/                   # Preprocessing configs
│
├── scripts/                    # Analysis and utility scripts
│   ├── analyze_*.py            # Result analysis scripts
│   ├── combine_results_*.py    # Result aggregation
│   ├── generate_prolific_*.py  # Experiment generation
│   └── sample_*.py             # Dataset sampling
│
├── utils/                      # Shared utilities
│   ├── api.py                  # API wrappers
│   ├── models.py               # Model implementations
│   ├── wrappers.py             # LLM and image model wrappers
│   └── misc.py                 # Miscellaneous utilities
│
├── pyproject.toml              # Project dependencies
└── README.md                   # This file
```

### Configuration with Hydra

`Visual Persuasion` uses [Hydra](https://hydra.cc/) for hierarchical configuration management. Override any parameter from the command line:

```bash
# Modify strategy parameters
python run_optimization.py \
  strategy=competition \
  strategy.min_rounds_before_equilibrium=10 \
  strategy.max_rounds_per_pair=20
```

#### Customizing LLM Providers

The framework uses [LiteLLM](https://www.litellm.ai/) for unified API access. Configure different models in `nudging/conf/llm/`:

```bash
# Use GPT-4o for image editing
python run_optimization.py llm=gpt-5-2

# Use Claude for evaluation
python run_evaluation.py llm=claude-4-5-sonnet

# Use Gemini for optimization
python run_optimization.py llm=gemini-3-pro
```

#### Resuming Interrupted Runs

Resume optimization from the most recent checkpoint:

```bash
python run_optimization.py \
  strategy=competition \
  task=people \
  general.resume=true
```

### Creating Custom Tasks

Create a new task configuration in `nudging/conf/task/your_task.yaml`:

```yaml
name: "your_task"

competition:
  # Prompts for the CVPO strategy

textgrad:
  # Prompts for the VTG strategy

feedback_descent:
  # Prompts for the VFD strategy

evaluation:
  # Prompts for evaluation methods
```

Then run:

```bash
python run_optimization.py task=your_task
```

## FAQs

### Can I access the data from the experiments in the paper?

Reach out to us! We have hundreds of GBs of data.

## Citing & Acknowledgements

If you use `Visual Persuasion` in your research, please cite the following paper:
```bibtex
[PLACEHOLDER]
```

[PLACEHOLDER]
