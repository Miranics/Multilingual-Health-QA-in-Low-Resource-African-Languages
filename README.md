# Multilingual Health Question Answering in Low-Resource African Languages

Parameter-efficient fine-tuning of **mT5** with **LoRA** for answering health questions in low-resource African languages, built for the [Zindi Multilingual Health QA Challenge](https://zindi.africa).

**Author:** Nanen Miracle Mbanaade  ·  **Kaggle:** Miranics
**Best public leaderboard score:** **0.2700**  ·  **Best rank:** **468**

---

## Overview

Speakers of languages such as Luganda, Swahili, Akan, and Amharic are underserved by English-centric language models, especially for sensitive health topics. This project fine-tunes the multilingual T5 (mT5) encoder–decoder model to take a health question in one of these languages and produce a fluent, accurate answer in the **same** language.

The work is organised as **eleven documented experiments** that progress from a zero-shot baseline to the best configuration, changing one factor at a time so that every score change is attributable to a specific, motivated decision. The public score improved from a near-zero zero-shot floor (0.0017) to **0.2700**, with the single largest gain coming from switching the inference decoder from greedy to **beam search**.

## Key Results

| Exp | Model | Key change | Decoding | Zindi LB | ROUGE-1 | ROUGE-L | LLM-Judge |
|-----|-------|-----------|----------|----------|---------|---------|-----------|
| 01 | mt5-small | zero-shot baseline | greedy | 0.0017 | 0.0001 | 0.0001 | — |
| 03 | mt5-small | LoRA r=8, attn (q,v) | greedy | — | 0.1171\* | 0.1000\* | — |
| 04 | mt5-small | LoRA r=16, attn (q,v) | greedy | 0.1218 | 0.1839 | 0.1453 | 0.0 |
| 05 | mt5-small | LoRA full attn + FFN | greedy | 0.1598 | 0.1713 | 0.1380 | 0.1746 |
| 06 | mt5-base | scaled model | greedy | 0.1811 | 0.2077 | 0.1668 | 0.1638 |
| 07 | mt5-base | beam search (4) | beam=4 | 0.2536 | 0.2530 | 0.1811 | 0.3576 |
| 08 | mt5-base | 5 epochs | beam=4 | 0.2613 | 0.2760 | 0.1923 | 0.3384 |
| 09 | mt5-base | LoRA r=32 | beam=4 | 0.2554 | 0.2678 | 0.1879 | 0.3337 |
| 10 | mt5-base | richer prompt | beam=4 | 0.2533 | 0.2615 | 0.1856 | 0.3381 |
| **11** | **mt5-base** | **lower LR 2e-4 (champion)** | **beam=4** | **0.2700** | **0.2905** | **0.1939** | **0.3493** |

\* Exp 03 validation metrics (baseline reference, not submitted). Exp 10 and 11 were submitted just after the official close; their scores are genuine Zindi evaluations that did not affect the graded standing. **Exp 11 is the best overall result.**

The final score is the weighted mean of the three metrics: `0.37 × ROUGE-1 + 0.37 × ROUGE-L + 0.26 × LLM-Judge`.

## Repository Structure

```
.
├── notebooks/
│   ├── 01_baseline.ipynb        # Zero-shot baseline (performance floor)
│   ├── 03_finetuning.ipynb      # LoRA r=8, attention only (baseline fine-tune)
│   ├── 04_finetuning.ipynb      # LoRA r=16
│   ├── 05_finetuning.ipynb      # LoRA full attention + FFN targets
│   ├── 06_finetuning.ipynb      # mT5-base
│   ├── 07_finetuning.ipynb      # Beam search inference (largest single gain)
│   ├── 08_finetuning.ipynb      # 5 epochs
│   ├── 09_finetuning.ipynb      # LoRA r=32 (negative result)
│   ├── 10_finetuning.ipynb      # Richer prompt + length penalty
│   └── 11_finetuning.ipynb      # Lower learning rate (champion, 0.2700)
├── report/
│   └── Nanen_Mbanaade_FinalProject.pdf
├── screenshots/                 # Leaderboard score & ranking evidence
├── outputs/                     # Submission CSVs, learning curves, plots
└── README.md
```

> Each notebook is self-contained: it loads the data, preprocesses, fine-tunes (where applicable), evaluates with ROUGE, and generates a Zindi submission file. Every code cell has a Markdown explanation above it.

## Quickstart

### Option A — Google Colab (recommended)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/USERNAME/REPO/blob/main/notebooks/11_finetuning.ipynb)

1. Click the badge above (update `USERNAME/REPO` to your GitHub path after pushing).
2. Set the runtime to a GPU: **Runtime → Change runtime type → T4 GPU**.
3. Upload the dataset (`Train.csv`, `Val.csv`, `Test.csv`) to your Google Drive and update the `DATA_DIR` path in the paths cell.
4. **Runtime → Run all.**

### Option B — Kaggle

1. Create a new Kaggle Notebook and attach the dataset (`Train.csv`, `Val.csv`, `Test.csv`).
2. Settings → Accelerator → **GPU T4 x2** (use **T4**, not P100).
3. Update `DATA_DIR` to your dataset path (e.g. `/kaggle/input/<your-dataset>/`).
4. For long runs, use **Save Version → Save & Run All (Commit)** so the notebook runs to completion in the background.

### Environment

```bash
pip install "peft==0.14.0" transformers datasets accelerate rouge-score sentencepiece
```

> **Important:** do not let `pip` upgrade the platform's pre-installed PyTorch — it keeps the CUDA build matched to the GPU. `peft` is pinned to `0.14.0` to avoid an incompatible `torchao` requirement.

## Method

- **Model:** `google/mt5-small` (≈300M) for fast iteration, `google/mt5-base` (≈580M) for the strongest runs.
- **Fine-tuning:** LoRA (Low-Rank Adaptation) — pretrained weights frozen, small low-rank adapters trained. Explored hyperparameters: rank (8 / 16 / 32) and target modules (attention only vs. attention + feed-forward).
- **Precision:** **bfloat16** throughout. mT5 was pretrained in bf16 and overflows under fp16, which causes the loss to collapse to `nan` — bf16 is essential.
- **Prompting:** each input is conditioned on the target language and country to encourage same-language answers.
- **Decoding:** beam search (`num_beams=4`) with repetition and length controls — the change that produced the largest single improvement.
- **Evaluation:** ROUGE-1 F1, ROUGE-L F1 (via `rouge-score`), and the competition's LLM-as-a-Judge, tracked separately to distinguish lexical overlap from genuine fluency.

## Key Insights

- **Decoding strategy is a first-class lever.** Switching greedy → beam search (Exp 07) raised the score ~40% and more than doubled the LLM-Judge, with no retraining.
- **Where adapters go matters as much as how many.** Extending LoRA to the feed-forward layers (Exp 05) unlocked fluency: LLM-Judge went from 0.0 to 0.17.
- **More is not always better.** Doubling LoRA rank (Exp 09) and adding epochs both showed diminishing or negative returns.
- **Trust full evaluation over small samples.** The champion (Exp 11) was under-estimated on the validation sample and only revealed as best on the full test set.

## Reproducibility Notes

- Random seeds are fixed (`SEED = 42`) across `random`, `numpy`, and `torch`.
- Maximum input/target lengths (384 / 200 tokens) are derived from the data's length distribution (see EDA in the notebooks).
- Submission format follows the Zindi multi-metric spec: columns `ID`, `TargetRLF1`, `TargetR1F1`, `TargetLLM`.

## Limitations & Future Work

The largest model used was mT5-base (free-GPU constraints), and per-language performance is uneven — stronger for English and Akan, weaker for Amharic and Luganda, reflecting data imbalance. Future directions: larger or African-language-specific models, data augmentation for under-resourced languages, retrieval-augmented generation for factual grounding, and a joint hyperparameter search.

## Ethical Considerations

Health information is high-stakes: a fluent but factually wrong answer can cause harm, so any real deployment would require clinical validation and human oversight. Building for low-resource languages is itself a step toward equity, but uneven per-language quality must be disclosed and addressed.

## Acknowledgements & AI Use

Dataset and challenge by the [Hub for Artificial Intelligence in Maternal, Sexual and Reproductive Health (HASH)](https://hash.theacademy.co.ug/) via Zindi. AI tools were used as a support aid for debugging, structuring experiments, and drafting documentation; all experimental decisions, interpretations, and results are the author's own and were executed and validated by the author. See the report's "Statement on the Use of AI Tools" for details.

## References

- Hu et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR.
- Xue et al. (2021). *mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer.* NAACL.
- Raffel et al. (2020). *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer.* JMLR.
- Lin (2004). *ROUGE: A Package for Automatic Evaluation of Summaries.* ACL Workshop.
