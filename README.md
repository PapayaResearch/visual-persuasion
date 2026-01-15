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

## Dataset Preprocessing

```bash
cd setup
python main.py general.src_dir=/path/to/visual-nudging/setup/data/abod general.dst_dir=/path/to/visual-nudging/nudging/data dataset.name=abod general.max_workers=32
```

## Comparability Evaluation

```bash
cd nudging
python run_priors.py general.data_dir=/path/to/visual-nudging/nudging/data/abod/ strategy=competition general.max_workers=32
```

## Optimization Pipeline

### Zero-shot Strategy

```bash
cd nudging
python run_optimization.py general.data_dir=/path/to/visual-nudging/nudging/data/abod/ strategy=zero-shot general.max_workers=32
```

### Competition Strategy

#### NOTE: Comparability evaluation (run_priors.py) must be run before using the competition strategy

```bash
cd nudging
python run_optimization.py general.data_dir=/path/to/visual-nudging/nudging/data/abod/ strategy=competition general.max_workers=32
```

## Evaluation

### Pair-wise Evaluation

```bash
cd nudging
python run_evaluation.py evaluate=pairs general.data_dir=results/competition/TIMESTAMP/ general.max_workers=32
```

### Solo Evaluation

```bash
cd nudging
python run_evaluation.py evaluate=solo general.data_dir=results/competition/TIMESTAMP/ general.max_workers=32
```

## Interpretation of Results

```bash
cd nudging
python run_interpret.py interpret.results_dir=results/nudging/model-name/test general.max_workers=32
```
