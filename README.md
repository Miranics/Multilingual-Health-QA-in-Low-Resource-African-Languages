# Multilingual Health QA in Low-Resource African Languages

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Miranics/Multilingual-Health-QA-in-Low-Resource-African-Languages/blob/main/notebooks/01_baseline.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

Final project for Machine Learning Techniques I — Zindi Competition.

---

## Project structure...

```
├── data/
│   ├── raw/               # Train.csv, Val.csv, Test.csv (not committed)
│   ├── processed/         # Tokenized datasets
│   └── samples/           # Small subsets for quick testing
│
├── notebooks/
│   ├── 01_baseline.ipynb          # EDA + zero-shot baseline
│   ├── 02_preprocessing.ipynb     # Prompt engineering experiments
│   ├── 03_finetuning.ipynb        # mT5-small + QLoRA fine-tuning
│   ├── 04_evaluation.ipynb        # ROUGE analysis + per-language breakdown
│   └── 05_final_submission.ipynb  # Best model → final submission CSV
│
├── src/
│   ├── preprocessing/preprocess.py
│   ├── training/trainer.py
│   ├── inference/predict.py
│   ├── evaluation/metrics.py
│   └── utils/helpers.py
│
├── configs/
│   ├── baseline_config.yaml
│   ├── lora_config.yaml
│   └── training_config.yaml
│
├── outputs/
│   ├── submissions/
│   ├── logs/
│   └── checkpoints/
│
└── assets/leaderboard_screenshots/
```

---

## Dataset

| File | Rows | Columns |
|------|------|---------|
| Train.csv | ~29,815 | ID, prompt, response, language, country |
| Val.csv | ~6,686 | ID, prompt, response, language, country |
| Test.csv | 2,618 | ID, prompt, language, country |

Languages: Akan · Amharic · Luganda · Swahili · English

---

## Evaluation metrics

| Metric | Weight |
|--------|--------|
| ROUGE-1 F1 | 0.37 |
| ROUGE-L F1 | 0.37 |
| LLM-as-a-Judge | 0.26 |

Submission format: `ID | TargetRLF1 | TargetR1F1 | TargetLLM`
All three Target columns hold the same predicted answer string.

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/Miranics/Multilingual-Health-QA-in-Low-Resource-African-Languages.git
cd Multilingual-Health-QA-in-Low-Resource-African-Languages

# 2. Install
pip install -r requirements.txt
```

Upload Train.csv, Val.csv, Test.csv to `MyDrive/health_qa/` in Google Drive, then open `notebooks/01_baseline.ipynb` in Colab.

---

## Experiment log

| # | Description | Model | ROUGE-1 | ROUGE-L | LB Score |
|---|-------------|-------|---------|---------|----------|
| 01 | Zero-shot baseline | mT5-small | — | — | — |
| 02 | Prompt engineering v1 | mT5-small | — | — | — |
| 03 | Fine-tune QLoRA r=8 | mT5-small | — | — | — |
| 04 | QLoRA r=16 | mT5-small | — | — | — |
| 05 | Longer training | mT5-small | — | — | — |
| 06 | Scale up | mT5-base | — | — | — |
| 07 | Beam search tuning | mT5-base | — | — | — |
| 08 | Train + Val combined | mT5-base | — | — | — |
| 09 | Prompt v2 | mT5-base | — | — | — |
| 10 | AfroXLMR | AfroXLMR | — | — | — |

---

## Reproducibility

All random seeds fixed via `set_seed(42)`. All hyperparameters versioned in `configs/`. Every notebook runs on Colab Free (T4 GPU).
