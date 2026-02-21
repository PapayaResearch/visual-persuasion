# Visual Persuasion: Data Analysis

Analysis repository for visual persuasion experiments across model and human evaluations, including:
- Paired choice analyses (R)
- Mitigation & distillation comparisons (R)
- Image similarity analysis pipelines (Python)
- Hierarchical auto-interpretability summarization (Python)

## Outputs
- PDF plots in `plots/`
- LaTeX tables in `tables/`
- Long-form similarity CSVs and caches in `similarity_analysis/similarity_analysis_out/` (or configured output directory)

## Environment Setup

### System prerequisites

The original environment uses:
- Python 3.11.14
- R 4.5.2 ([Not] Part in a Rumble)


You may also want to configure AWS credentials (required for the S3-backed similarity scripts):
```bash
export AWS_REGION_NAME="<AWS_REGION>"
export AWS_ACCESS_KEY_ID="<AWS_ACCESS_KEY_ID>"
export AWS_SECRET_ACCESS_KEY="<YOUR_AWS_SECRET_KEY>"
```

### Python environment

```bash
conda create -n vpa python=3.11.14
pip install uv
uv pip install -r requirements.txt
```

Notes:
- `rembg` may require additional runtime dependencies depending on platform.
- GPU is optional but recommended for LPIPS/CLIP workloads.

### R environment

Install required R packages:

```r
install.packages(c(
  "tidyverse", "fixest", "emmeans", "ggpubr", "ggsci",
  "showtext", "cowplot", "patchwork", "knitr", "kableExtra"
))
```

### Input data
CSV files in `data/` are used directly by R scripts: check the filenames per script to map file names accordingly when you generate results. The suffix `-human` is used here to signify data from online experiments with participants.

## Reproducible Execution Workflows

## A) Produce R analyses with your data
Run from repository root:

```bash
# Main results for models
Rscript choice_combined.R
Rscript choice_head2head.R
Rscript choice_mitigations.R
Rscript choice_distilled.R

# Main results for humans
Rscript choice_human.R

# Analysis of image similarities
Rscript image_analysis.R
```

Outputs are written to:
- `plots/`
- `tables/`

## B) Compute similarity metrics from S3 (canonical path)
Use orchestrator:

```bash
python -m similarity_analysis.run_everything \
  --cache_dir s3://<bucket-name>/cache/ \
  --data_dir s3://<bucket-name>/data/ \
  --out_dir similarity_analysis/similarity_analysis_out
```

Useful flags:

- `--no_ssim`
- `--no_lpips`
- `--no_embedding`
- `--no_matted_ssim`
- `--no_matted_embedding`
- `--ssim_size 224` (`0` disables resize)
- `--lpips_size 224` (`0` disables resize)
- `--embedding_model ViT-B/32`
- `--matting_model u2netp`
- `--background white`

Then regenerate aggregate similarity plot in R:

```bash
Rscript image_analysis.R
```

## C) Run hierarchical autointerp summarization

```bash
python autointerp/run_autointerp.py \
  --results_dir data/results-autointerp \
  --out_dir data/hierarchical_autointerp \
  --include_config
```

Note that this expects the raw output of `evaluate/run.py` (see [main README](/README.md)).

Useful flags:

- `--groupby_cols task.name`
- `--strategy_col evaluate.strategy_name`
- `--embedding_model text-embedding-3-small`
- `--summarizer_model gemini/gemini-3-flash-preview`
- `--matryoshka`
- `--save_embeddings`

## Script-by-Script Reference

### R scripts

- `choice_combined.R`: Main combined status/type-wise choice analysis across tasks. Writes task-level and pooled plots + summary tables
- `choice_head2head.R`: Strategy-vs-strategy head-to-head analysis for model judgments
- `choice_distilled.R`: Distillation analysis
- `choice_mitigations.R`: Mitigation-run analysis with explicit `inconsistent` handling
- `choice_human.R`: Human participant analyses, including type/status choice, strategy head-to-head comparisons, and mitigation-specific human analysis pipelines
- `image_analysis.R`: Consumes long-form similarity CSVs and builds a consolidated panel plot
- `utils.R`: Shared R utilities for cleaning, modeling, contrasts, plotting, and output

### Python scripts

- `similarity_analysis/run_everything.py`: End-to-end in-process runner for SSIM, LPIPS, embedding, and matted variants. See arguments for running subsets of these
- `similarity_analysis/s3_image_store.py`: Shared cached S3/PIL/array/tensor access layer
- `similarity_analysis/sim_utils.py`: Shared metadata parsing, pair filtering, CI computation, plotting helper
- `autointerp/run_autointerp.py`: Hierarchical clustering + LLM summarization pipeline
