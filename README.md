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

## Task Configurations

The system includes task-specific prompts for different domains. Task configurations are located in `nudging/conf/tasks/`:

- **people.yaml** - Job candidate images
- **products.yaml** - Product images
- **houses.yaml** - Real estate images

Each task contains prompts for both optimization (during image generation) and evaluation (when comparing results). To use a different task, specify it when running:

```bash
python run_optimization.py tasks=products ...
python run_evaluation.py tasks=houses ...
```

## Dataset Preprocessing

```bash
cd setup
python main.py general.src_dir=/path/to/visual-nudging/setup/data/abod general.dst_dir=/path/to/visual-nudging/nudging/data dataset.name=abod general.max_workers=32
```

## Optimization Pipeline

### Competition Strategy

```bash
cd nudging
python run_optimization.py strategy=competition task=people general.data_dir=/path/to/visual-nudging/nudging/data/abod_enhanced/ general.max_workers=64
```

## Evaluation

### Pair-wise Evaluation

```bash
cd nudging
python run_evaluation.py evaluate=pairs general.data_dir=/path/to/visual-nudging/nudging/results/competition/TIMESTAMP/ general.max_workers=64
```

### Solo Evaluation

```bash
cd nudging
python run_evaluation.py evaluate=solo general.data_dir=/path/to/visual-nudging/nudging/results/competition/TIMESTAMP/ general.max_workers=64
```

### Chain Evaluation

```bash
cd nudging
python run_evaluation.py evaluate=chain general.data_dir=/path/to/visual-nudging/nudging/results/competition/TIMESTAMP/ general.max_workers=64
```

## Interpretation of Results

```bash
cd nudging
python run_evaluation.py evaluate=autointerp general.data_dir=/path/to/visual-nudging/nudging/results/competition/TIMESTAMP general.max_workers=64
```
