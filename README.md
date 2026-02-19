# Visual Persuasion: What Influences Decisions of Vision-Language Models?

![Python version](https://img.shields.io/badge/python-3.11-blue)
![Package version](https://img.shields.io/badge/version-0.1.0-green)
![GitHub license](https://img.shields.io/github/license/PapayaResearch/visual-persuasion)

> [!NOTE]
> Code for the paper **[Visual Persuasion: What Influences Decisions of Vision-Language Models?](https://arxiv.org/abs/2602.15278)**

<img width="1920" height="751" alt="banner" src="https://github.com/user-attachments/assets/588977ec-d7fc-4a2e-bba0-0039bba8dca8" />

The web is littered with images, once created for human consumption and now increasingly interpreted by agents using vision-language models (VLMs). These agents make visual decisions at scale, deciding what to click, recommend, or buy. Yet, we know little about the structure of their visual preferences. We introduce a framework for studying this by placing VLMs in controlled image-based choice tasks and systematically perturbing their inputs. Our key idea is to treat the agent's decision function as a latent visual utility that can be inferred through revealed preference: choices between systematically edited images. Starting from common images, such as product photos, we propose methods for visual prompt optimization, adapting text optimization methods to iteratively propose and apply visually plausible modifications using an image generation model (such as in composition, lighting, or background). We then evaluate which edits increase selection probability. Through large-scale experiments on frontier VLMs, we demonstrate that optimized edits significantly shift choice probabilities in head-to-head comparisons. We develop an automatic interpretability pipeline to explain these preferences, identifying consistent visual themes that drive selection. We argue that this approach offers a practical and efficient way to surface visual vulnerabilities, safety concerns that might otherwise be discovered implicitly in the wild, supporting more proactive auditing and governance of image-based AI agents.

## Features

- 🎨 Multiple optimization strategies: Competitive Visual Prompt Optimization (CVPO), VisualTextGrad (VTG), VisualFeedbackDescent (VFD)
- 🤖 Support for multiple VLMs through LiteLLM (e.g. OpenAI, Anthropic, Gemini, Qwen, Llama)
- 🎯 Default tasks for people (hiring), products (buying), houses (buying), and hotels (booking)
- 📊 Evaluation methods: pairwise comparison and cross-strategy
- 🔬 Automated interpretability pipeline for extracting visual patterns from optimized images
- 🧪 Distillation method to edit images using what was learned with the interpretability pipeline
- 🛡️ Mitigation pipeline to reduce the effect of the optimization
- ⚙️ Hydra-based configuration system for reproducible experiments

## Installation

Create a Python environment:

```bash
conda create -n vp python=3.11
conda activate vp
```

install the dependencies:

```bash
pip install uv
uv pip install -e .
```

and create a `.env` file in the project root with your API keys:

> [!IMPORTANT]
> You must include an API key for Gemini or setup a different editor.

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

> [!NOTE]
> All commands run from the `visualpersuasion/` directory.

## Optimization

```bash
python optimization/run.py \
  strategy=cvpo \
  task=houses \
  general.data_dir=data/houses_enhanced
```

**Options:**
- `strategy`: cvpo, vtg, vfd, distillation
- `task`: people, products, houses, hotels
- `llm`: gemini-3-flash, claude-4-5-haiku, gpt-5-2, etc.
- `general.resume`: true/false (copies and continues last run)
- `general.max_workers`: number of parallel workers (default: 8)

Results are saved in `results/` with timestamps.

## Evaluation

```bash
python evaluation/run.py \
  evaluate=pairs \
  general.data_dir=results/cvpo/TIMESTAMP
```

**Options:**
- `evaluate`: pairs (head-to-head comparisons), strategies (cross-strategy comparison)
- `strategy`: cvpo, vtg, vfd, distillation
- `task`: people, products, houses, hotels
- `llm`: gemini-3-flash, claude-4-5-haiku, gpt-5-2, etc.
- `evaluate.max_comparisons`: limit number of comparisons (-1 for all)
- `general.max_workers`: number of parallel workers (default: 8)

Results for pairs are saved in `results/cvpo/TIMESTAMP/evaluation/MODEL`, and for strategies in `results-cross-strategies/` by default.

## Automated Interpretability & Distillation

Generate natural language explanations of visual patterns:

```bash
python evaluation/run.py evaluate=autointerp general.data_dir=results/cvpo/TIMESTAMP
```

Results are saved in `results/cvpo/TIMESTAMP/evaluation/MODEL`.

Zero-shot edits with learned patterns:

```bash
python optimization/run.py strategy=distillation task=houses general.data_dir=data/houses_enhanced
```

Results are saved in `results/` with timestamps.

## Mitigation

Evaluate mitigation techniques:

```bash
python evaluation/run.py evaluate=mitigations general.data_dir=results/cvpo/TIMESTAMP
```

**Options:**
- `llm`: gemini-3-flash, claude-4-5-haiku, gpt-5-2, etc.
- `evaluate.max_comparisons`: limit number of comparisons (-1 for all)
- `evaluate.iterations`: number of debiasing rounds (default: 1)
- `general.max_workers`: number of parallel workers (default: 8)

This saves all debiased images in `cache`, and the results fo the evaluation in `results/cvpo/TIMESTAMP/evaluation/MODEL`.

## Configuration

The framework uses [Hydra](https://hydra.cc/) for configuration management. Take a look into `visualpersuasion/conf/` to see all the details and options.

## Preprocessing (Optional)

We have included `visualpersuasion/data/` with the enhanced images we use as originals, but we also provide the enhancing script if you need it for your own dataset. It expects a directory with categories inside, which in turn contain the images.

```bash
python preprocess/run.py \
  preprocess=enhance \
  general.data_dir=data/your_dataset \
  preprocess.results_dir=data/your_dataset_enhanced
```

## Custom Tasks

Create `conf/task/your_task.yaml` and run with `task=your_task`.

## Analysis

The analysis code has its own README!

## FAQs

### Can I access the data from the experiments in the paper?

Reach out to us! We have hundreds of GBs of data.

## Citing & Acknowledgements

If you use `Visual Persuasion` in your research, please cite the following paper:

```bibtex
@inproceedings{cherep2026visualpersuasion,
  title     = {Visual Persuasion: What Influences Decisions of Vision-Language Models?},
  author    = {Cherep, Manuel and M R, Pranav and Maes, Pattie and Singh, Nikhil},
  year      = {2026},
  url       = {https://arxiv.org/abs/2602.15278}
}
```

We received funding from SK Telecom with MIT’s Generative AI Impact Consortium (MGAIC). Research reported in this publication was supported by an Amazon Research Award, Fall 2024. Google made this project possible through a Gemini Academic Program Award. Other experiments conducted in this paper were generously supported via API credits provided by OpenAI and Anthropic. MC is supported by a fellowship from “la Caixa” Foundation (ID 100010434) with code LCF/BQ/EU23/12010079.
