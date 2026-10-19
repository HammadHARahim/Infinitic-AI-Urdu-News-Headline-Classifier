# Infinitic AI Urdu News Headline Classifier

Infinitic AI research project for classifying Urdu news headlines. Stage 1
covers four labels from one licensed research corpus. A later stage will add
politics, health, and world news after a second licensed source is cleared.

## Key points

- **Two tracks.** Track B (this work): fine-tune a pretrained multilingual
  transformer. Track A (later): Urdu tokenizer and transformer trained from
  scratch.
- **Stage 1 labels.** `sports`, `business`, `entertainment`, `technology`
  (mapped from Urdu News Dataset 1M).
- **Data rules.** Normalize, deduplicate, and split by content hash before any
  modeling. Seed **42**. Split **80 / 10 / 10** train / validation / test.
- **Selection metric.** Validation **macro-F1** only. The test split stays
  sealed until a finalist is declared and is scored once.
- **Starting model.** `xlm-roberta-base`, full fine-tune and LoRA ladders on a
  Kaggle Tesla T4 (batch size 32, max length 64).
- **Experiment discipline.** One controlled change per run. Every run is logged
  in `experiments/registry.csv`. Metrics are measured, not invented.
- **Finalist rule.** When the ladder finishes, train once more from
  `xlm-roberta-base` using only settings that beat the kept validation score
  (see `docs/decisions/ADR-001-final-finetune-uses-winning-settings.md`).

## Current kept recipe (Stage 1, Track B)

Full fine-tune of `xlm-roberta-base`:

| Setting | Value |
| --- | --- |
| Schedule | Linear warmup 10% of 20-epoch cosine steps, half-cosine decay to ~0, then constant **2e-6** |
| Peak learning rate | **2e-5** |
| Weight decay | **0.01** |
| Loss | Unweighted cross-entropy (no label smoothing) |
| Batch size / max length / seed | **32** / **64** / **42** |
| Best validation macro-F1 | **0.9481** (epoch 46) |
| Checkpoint | private Hub `HammadHARahim/urdu-headline-stage1-cosine` |

Label smoothing 0.1, weight decay 0.1, and class-weighted loss were tried as
sole changes and did **not** beat the kept score, so they stay out of the
recipe.

## Results report (validation macro-F1)

Test metrics are empty on purpose. The test split has not been read for
selection.

### Baselines

| Run | Validation macro-F1 |
| --- | ---: |
| Majority (train majority = sports) | 0.1437 |
| TF-IDF + logistic regression | 0.9230 |

### Track B ladder (selected scores)

| Run | Strategy | Val macro-F1 | Notes |
| --- | --- | ---: | --- |
| Frozen encoder | classifier only | 0.8538 | T4, 2 epochs |
| Full fine-tune (A100, short) | full | 0.9257 | 2 epochs |
| LoRA rank 4 / 8 / 16 | LoRA | 0.9021 / 0.9042 / 0.9120 | 2-epoch ladder rows |
| Full T4, longer LR sweep | full | 0.9449 | best early constant-LR peak |
| Warmup + linear decay | full | 0.9466 | epoch 19 |
| **Cosine + 2e-6 continuation** | **full** | **0.9481** | **kept best, epoch 46** |
| Cosine + label smoothing 0.1 | full | 0.9466 | did not beat 0.9481 |
| LoRA r4 + smoothing (long) | LoRA | 0.9395 | best LoRA so far; below full model |

Full run log: [`experiments/registry.csv`](experiments/registry.csv).  
Per-run text reports: `data/processed/stage1_*.txt`.

### Status

- Track B Stage 1 ladder is active. Next unused skill-list items after the
  current cosine+smoothing resume: **input max length** and **model comparison**.
- Track A (from scratch) has not started.
- No finalist declared. **No test scores.**

## Data

- Source: [Urdu News Dataset 1M](https://www.kaggle.com/datasets/saurabhshahane/urdu-news-dataset)
  (Mendeley v3, DOI [10.17632/834vsxnb99.3](https://data.mendeley.com/datasets/834vsxnb99/3)).
- License note: [`data/licenses/urdu_news_dataset_1m.md`](data/licenses/urdu_news_dataset_1m.md)
  — non-commercial research; credit news sources.
- Raw CSV and split CSVs are **not** in this repo (see `.gitignore`). Rebuild
  with `notebooks/stage1_prepare.py` after placing the raw file under
  `data/raw/`.

Stage 1 audit (from the local raw file): 111,860 kept headlines —
sports 44,829, entertainment 34,901, business 24,131, technology 7,999.

## Repo layout

| Path | Role |
| --- | --- |
| `SPEC.md` | Binding stage-1 specification |
| `notebooks/` | Training and prep scripts |
| `kaggle/` | Private Kaggle GPU launchers |
| `experiments/registry.csv` | Every recorded run |
| `data/processed/*.txt` | Score reports and audits |
| `docs/decisions/` | ADRs |
| `tasks/plan.md` | Ladder checklist |
| `.cursor/skills/` | Project agent skills |

## Reproduction notes

1. Place the raw corpus at `data/raw/urdu-news-dataset-1M.csv`.
2. Run `notebooks/stage1_prepare.py`, then baselines, then a ladder script under
   `notebooks/`.
3. Kaggle T4 jobs use the launchers in `kaggle/` plus a private dataset that
   holds slim train/validation CSVs and a Hub token (never commit the token).
4. Select on validation macro-F1 only. Do not open the test split until a
   finalist is declared.

## Project docs

- Spec: [`SPEC.md`](SPEC.md)
- Plan: [`tasks/plan.md`](tasks/plan.md)
- Final-finetune ADR: [`docs/decisions/ADR-001-final-finetune-uses-winning-settings.md`](docs/decisions/ADR-001-final-finetune-uses-winning-settings.md)
