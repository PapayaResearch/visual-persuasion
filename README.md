# Visual Nudging

## Installation

```bash
conda create -n vnudging python=3.11
conda activate vnudging
```

```bash
pip install uv
uv pip install -e .
```

## Environment Setup

Create a `.env` file in the project root with your API keys:

```bash
# .env
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GEMINI_API_KEY=your_gemini_key
AWS_REGION_NAME=your_aws_region
AWS_ACCESS_KEY_ID=your_aws_key_id
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
```

## Usage

You first need to download a dataset.

### Dataset Preprocessing

Prepare your image dataset:

```bash
cd setup
python main.py
```

**Hydra Parameters:**

```bash
# Use different sampling strategy
python main.py strategy=random-sampling

# Use different LLM
python main.py llm=gpt-5-mini

# Use different image editor
python main.py editor=nanobanana-litellm

# Override general settings
python main.py general.source_dir=/path/to/images general.dest_dir=/path/to/output

# Enable/disable image enhancement
python main.py general.enhance_images=true

# Configure dataset sampling
python main.py dataset.num_samples=50

# Enable background normalization
python main.py background_processor.normalize_backgrounds=true
```

### Run Nudging Pipeline

Execute the visual nudging optimization:

```bash
cd nudging
python run_nudging.py
```

**Hydra Parameters:**

```bash
# Change number of iterations
python run_nudging.py general.iterations=5

# Use different strategy
python run_nudging.py strategy=zero-shot
python run_nudging.py strategy=prompt-optimization
python run_nudging.py strategy=tournament-of-images
python run_nudging.py strategy=tournament-of-prompts

# Use different LLM for optimization
python run_nudging.py llm=gpt-5
python run_nudging.py llm=claude-4-5-sonnet
python run_nudging.py llm=gemini-2-5-pro
python run_nudging.py llm=o3

# Use different image editor
python run_nudging.py editor=nanobanana-gemini
python run_nudging.py editor=nanobanana-litellm

# Override specific models in the pipeline
python run_nudging.py nudge.evaluator_model.model=gpt-5-2025-08-07
python run_nudging.py nudge.loss_model.model=claude-sonnet-4-5-20250924
python run_nudging.py nudge.optimizer_model.model=gemini-2.5-pro

# Change data directory
python run_nudging.py general.data_dir=/path/to/images

# Combine multiple overrides
python run_nudging.py general.iterations=10 strategy=tournament-of-prompts llm=claude-4-5-sonnet
```

### Run Evaluation

Evaluate the generated images:

```bash
cd nudging
python run_evaluation.py general.evaluation_dir=/path/to/results
```

**Hydra Parameters:**

```bash
# Evaluate specific results directory
python run_evaluation.py general.evaluation_dir=/path/to/nudging/results

# Use different evaluator model
python run_evaluation.py llm=claude-4-5-sonnet

# Change evaluator model directly
python run_evaluation.py evaluate.evaluator_model.model=gpt-5-2025-08-07
```

### Run Analysis

Generate statistics and visualizations:

```bash
cd nudging
python run_analysis.py general.analysis_csv=/path/to/evaluation/results.csv
```

**Hydra Parameters:**

```bash
# Analyze specific CSV results
python run_analysis.py general.analysis_csv=/path/to/results/evaluation.csv

# Customize visualization settings
python run_analysis.py analyze.num_previews=10
python run_analysis.py analyze.dpi=150
python run_analysis.py analyze.figsize=[12,8]
```
