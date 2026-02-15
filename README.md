# Visual Persuasion: What Influences Decisions of Vision-Language Models?

![Python version](https://img.shields.io/badge/python-3.11-blue)
![Package version](https://img.shields.io/badge/version-0.1.0-green)
![GitHub license](https://img.shields.io/github/license/PapayaResearch/visual-persuasion)

> [!NOTE]
> Code for the paper **[Visual Persuasion: What Influences Decisions of Vision-Language Models?]([PLACEHOLDER])**

The web is littered with images, once created for human consumption and now increasingly interpreted by agents using vision-language models (VLMs). These agents make visual decisions at scale, deciding what to click, recommend, or buy. Yet, we know little about the structure of their visual preferences. We introduce a framework for studying this by placing VLMs in controlled image-based choice tasks and systematically perturbing their inputs. Our key idea is to treat the agent's decision function as a latent visual utility that can be inferred through revealed preference: choices between systematically edited images. Starting from common images, such as product photos, we propose methods for visual prompt optimization, adapting text optimization methods to iteratively propose and apply visually plausible modifications using an image generation model (such as in composition, lighting, or background). We then evaluate which edits increase selection probability. Through large-scale experiments on frontier VLMs, we demonstrate that optimized edits significantly shift choice probabilities in head-to-head comparisons. We develop an automatic interpretability pipeline to explain these preferences, identifying consistent visual themes that drive selection. We argue that this approach offers a practical and efficient way to surface visual vulnerabilities, safety concerns that might otherwise be discovered implicitly in the wild, supporting more proactive auditing and governance of image-based AI agents.

## Features

- 🎨 Multiple optimization strategies: Competitive Visual Prompt Optimization (CVPO), VisualTextGrad (VTG), VisualFeedbackDescent (VFD), Formula-based
- 🤖 Support for multiple vision-language models through LiteLLM (e.g. GPT, Claude, Gemini, Qwen, Llama)
- 🎯 Realistic decision-making environments across hiring, e-commerce, real estate, and hospitality domains
- 📊 Comprehensive evaluation methods: pairwise comparison, solo assessment, chain analysis, and cross-method benchmarking
- 🔍 Automated interpretability pipeline for extracting visual patterns from optimized images
- ⚙️ Hydra-based configuration system for reproducible experiments

## Installation

### 1. Create Python Environment

```bash
conda create -n vp python=3.11
conda activate vp
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

Create a `.env` file in the project root with your API keys:

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
cd visualpersuasion
python preprocess/run.py \
  preprocess=random-sampling \
  general.data_dir=./data/your_dataset \
  preprocess.results_dir=./data/processed \
  general.max_workers=8
```

Optionally enhance images with background removal and standardization:

```bash
python preprocess/run.py \
  preprocess=enhance \
  general.data_dir=./data/processed \
  preprocess.results_dir=./data/enhanced \
  general.max_workers=8
```

### 2. Run Optimization

Optimize images using the competition strategy:

```bash
python optimization/run.py \
  strategy=cvpo \
  task=people \
  general.data_dir=./data/enhanced \
  general.max_workers=8
```

### 3. Evaluate Results

Evaluate the optimized images using pair-wise comparison:

```bash
python evaluation/run.py \
  evaluate=pairs \
  general.data_dir=./results/cvpo/TIMESTAMP \
  general.max_workers=8
```

Results are saved in `visualpersuasion/results/` with timestamps for easy tracking.

## Optimization Strategies

`Visual Persuasion` supports multiple optimization strategies:

### CVPO (Competitive Visual Prompt Optimization)

A novel visual prompt optimization algorithm that uses a competitive selection process with judge-based evaluation to iteratively refine prompts.

```bash
python optimization/run.py strategy=cvpo task=people
```

### VTG (VisualTextGrad)

An adaptation of the text-based gradient method from [TextGrad](https://arxiv.org/abs/2406.07496) to the visual domain by treating the image editing prompt as a differentiable textual object.

```bash
python optimization/run.py strategy=vtg task=hotels
```

### VFD (Visual Feedback Descent)

An optimization method based on the [Feedback Descent](https://arxiv.org/abs/2511.07919) method, following a proposal-and-evaluation loop.

```bash
python optimization/run.py strategy=vfd task=houses
```

### Distillation

A simple optimization strategy that applies a fixed formula to modify images based on domain-specific heuristics.

```bash
python optimization/run.py strategy=distillation task=products
```

## Evaluation Methods

`Visual Persuasion` provides multiple evaluation approaches to assess optimized images:

### Pair-wise Evaluation

Compare images head-to-head using LLM judges:

```bash
python evaluation/run.py \
  evaluate=pairs \
  general.data_dir=./results/cvpo/TIMESTAMP
```

### Automated Interpretability

Generate natural language interpretations of visual differences:

```bash
python evaluation/run.py \
  evaluate=autointerp \
  general.data_dir=./results/cvpo/TIMESTAMP
```

### Method Comparison

Compare different optimization strategies:

```bash
python evaluation/run.py \
  evaluate=strategies \
  general.data_dir=./results/
```

### Mitigation

Evaluate the effectiveness of mitigation strategies in reducing visual influence:

```bash
python evaluation/run.py \
  evaluate=mitigations \
  general.data_dir=./results/cvpo/TIMESTAMP
```

## Advanced Usage & Customization

### Project Structure

```
visual-persuasion/
├── visualpersuasion/           # Main package
│   ├── optimization/           # Optimization strategies
│   │   ├── run.py              # Entry point for optimization
│   │   ├── cvpo.py             # Competition-based optimization
│   │   ├── vtg.py              # VisualTextGrad optimization
│   │   ├── vfd.py              # Visual Feedback Descent
│   │   └── distillation.py     # Distillation-based optimization
│   ├── evaluation/             # Evaluation methods
│   │   ├── run.py              # Entry point for evaluation
│   │   ├── pairs.py            # Pairwise comparison
│   │   ├── autointerp.py       # Automated interpretability
│   │   ├── strategies.py       # Strategy comparison
│   │   └── mitigations.py      # Mitigation evaluation
│   ├── preprocess/             # Dataset preprocessing
│   │   ├── run.py              # Preprocessing entry point
│   │   ├── enhance.py          # Image enhancement
│   │   └── strategy.py         # Sampling strategies
│   ├── utils/                  # Shared utilities
│   │   ├── api.py              # API wrappers
│   │   ├── models.py           # Model implementations
│   │   ├── wrappers.py         # LLM and image model wrappers
│   │   └── misc.py             # Miscellaneous utilities
│   ├── conf/                   # Hydra configuration files
│   │   ├── config.yaml         # Main configuration
│   │   ├── task/               # Task-specific configs
│   │   ├── strategy/           # Optimization strategy configs
│   │   ├── evaluate/           # Evaluation method configs
│   │   ├── preprocess/         # Preprocessing configs
│   │   ├── llm/                # LLM provider configs
│   │   └── editor/             # Image editor configs
│   ├── config.py               # Hydra configuration dataclasses
│   └── schema.py               # Pydantic schemas for structured output
│
├── scripts/                    # Analysis and utility scripts
│   ├── analyze_*.py            # Result analysis scripts
│   ├── combine_results_*.py    # Result aggregation
│   └── generate_prolific_*.py  # Experiment generation
│
├── pyproject.toml              # Project dependencies
└── README.md                   # This file
```

### Configuration with Hydra

`Visual Persuasion` uses [Hydra](https://hydra.cc/) for hierarchical configuration management. Override any parameter from the command line:

```bash
# Modify strategy parameters
python optimization/run.py \
  strategy=cvpo \
  strategy.min_rounds_before_equilibrium=10 \
  strategy.max_rounds_per_pair=20
```

#### Customizing LLM Providers

The framework uses [LiteLLM](https://www.litellm.ai/) for unified API access. Configure different models in `visualpersuasion/conf/llm/`:

```bash
# Use GPT-4o for image editing
python optimization/run.py llm=gpt-4o

# Use Claude for evaluation
python evaluation/run.py llm=claude-sonnet-4-5

# Use Gemini for optimization
python optimization/run.py llm=gemini-2-flash
```

#### Resuming Interrupted Runs

Resume optimization from the most recent checkpoint:

```bash
python optimization/run.py \
  strategy=cvpo \
  task=people \
  general.resume=true
```

### Creating Custom Tasks

Create a new task configuration in `visualpersuasion/conf/task/your_task.yaml`:

```yaml
name: "your_task"

cvpo:
  # Prompts for the CVPO strategy

vtg:
  # Prompts for the VTG strategy

vfd:
  # Prompts for the VFD strategy

evaluation:
  # Prompts for evaluation methods
```

Then run:

```bash
python optimization/run.py task=your_task
```

## FAQs

### Can I access the data from the experiments in the paper?

Reach out to us! We have hundreds of GBs of data.

## Citing & Acknowledgements

If you use `Visual Persuasion` in your research, please cite the following paper:

```bibtex
[PLACEHOLDER]
```

We received funding from SK Telecom with MIT's Generative AI Impact Consortium (MGAIC). Research reported in this publication was supported by an Amazon Research Award, Fall 2024. Google made this project possible through a Gemini Academic Program Award. Other experiments conducted in this paper were generously supported via API credits provided by OpenAI and Anthropic. MC is supported by a fellowship from "la Caixa" Foundation (ID 100010434) with code LCF/BQ/EU23/12010079.
