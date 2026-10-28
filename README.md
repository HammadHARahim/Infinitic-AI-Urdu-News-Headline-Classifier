# Infinitic AI Urdu News Headline Classifier

Infinitic AI research project for classifying Urdu news headlines.

**Stage 1 Track B is wrapped for the 20 October 2026 deadline.** The sealed
test split was scored once on the declared finalist.

## Deadline delivery (20 October 2026)

| Item | Result |
| --- | --- |
| Finalist | Kept cosine full fine-tune of `xlm-roberta-base` |
| Checkpoint | `HammadHARahim/urdu-headline-stage1-cosine` (best validation epoch **46**) |
| Validation macro-F1 | **0.9481** |
| **Test macro-F1 (once)** | **0.9384** |
| Test accuracy | 0.9579 |
| Test size | 11,252 headlines |
| Fresh retrain from base | Skipped — Kaggle GPU quota was 0.00h on the deadline |
| Track A (from scratch) | Not started (out of Stage 1 deadline scope) |

Test report: [`data/processed/stage1_finalist_test.txt`](data/processed/stage1_finalist_test.txt).  
Decision note: [`docs/decisions/ADR-001-final-finetune-uses-winning-settings.md`](docs/decisions/ADR-001-final-finetune-uses-winning-settings.md).

### Per-label test F1

| Label | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| sports | 0.9778 | 0.9810 | 0.9794 | 4531 |
| business | 0.9618 | 0.9754 | 0.9686 | 2401 |
| entertainment | 0.9533 | 0.9360 | 0.9446 | 3531 |
| technology | 0.8532 | 0.8695 | 0.8613 | 789 |

Technology is the smallest class and the weakest label. Sports is strongest.

## Key points

- **Two tracks.** Track B (delivered): fine-tune a pretrained multilingual
  transformer. Track A (later): Urdu tokenizer and transformer from scratch.
- **Stage 1 labels.** `sports`, `business`, `entertainment`, `technology`
  (mapped from Urdu News Dataset 1M).
- **Data rules.** Normalize, deduplicate, hash-split before modeling. Seed
  **42**. Split **80 / 10 / 10**.
- **Selection metric.** Validation **macro-F1** only during the ladder.
- **Test rule.** Test stayed sealed until the finalist was declared; scored
  **once** on 20 October 2026.
- **Starting model.** `xlm-roberta-base` on a Kaggle Tesla T4 (batch 32,
  max length 64 for training).
- **Experiment discipline.** One controlled change per run. Logged in
  `experiments/registry.csv`. Metrics are measured, not invented.

## Finalist recipe

| Setting | Value |
| --- | --- |
| Model | `xlm-roberta-base` full fine-tune |
| Schedule | Warmup 10% of a **20-epoch** half-cosine, then constant **2e-6** from epoch 21 |
| Peak learning rate | **2e-5** |
| Weight decay | **0.01** |
| Loss | Unweighted cross-entropy (**no** label smoothing) |
| Batch size / max length / seed | **32** / **64** / **42** |
| Training epochs scored | **48** of planned 50 |
| Best validation epoch | **46** → **0.9481** |
| Test macro-F1 | **0.9384** |

Tried and **rejected** (did not beat the kept validation score): weight decay
0.1, class-weighted loss, label smoothing 0.1, LoRA ranks 4/8/16.

## Results report

### Baselines (validation macro-F1)

| Run | Val macro-F1 |
| --- | ---: |
| Majority (sports) | 0.1437 |
| TF-IDF + logistic regression | 0.9230 |

### Epoch log (main Track B runs)

| Run | Epochs scored | Best epoch | Val macro-F1 | Notes |
| --- | ---: | ---: | ---: | --- |
| Frozen encoder (T4) | 2 | 1 | 0.8538 | Encoder frozen |
| Full fine-tune (A100) | 2 | 2 | 0.9257 | Short baseline |
| LoRA rank 4 / 8 / 16 | 2 / 2 / 2 | 2 | 0.9021 / 0.9042 / 0.9120 | Ladder rows |
| Full T4 early long runs | 5 | 4 | 0.9417 | Stopped mid-run |
| Full T4, LR 5e-6 | 9 | 4 | 0.9449 | Early constant-LR peak |
| Warmup + linear decay | **20** | **19** | **0.9466** | Previous kept best |
| LoRA r4 + smoothing (long) | **100** | **56** | **0.9395** | Best LoRA |
| **Cosine + 2e-6 (finalist)** | **48** | **46** | **0.9481** | **Val select; test 0.9384** |
| Cosine + label smoothing 0.1 | **60** | **36** | **0.9466** | Completed 60; did not beat |

### Headline comparison

| | Warmup + linear | Cosine (finalist) | Cosine + smoothing |
| --- | ---: | ---: | ---: |
| Epochs scored | 20 | **48** | **60** |
| Best epoch | 19 | **46** | 36 |
| Val macro-F1 | 0.9466 | **0.9481** | 0.9466 |
| Test macro-F1 | — | **0.9384** | — |

Full log: [`experiments/registry.csv`](experiments/registry.csv).  
Per-run reports: `data/processed/stage1_*.txt`.

## What was left unfinished (honest)

- Ladder items still unused: **max length** and **model comparison**.
- Preferred ADR-001 step (fresh fine-tune from base with only winning
  settings) was **not** run — GPU quota exhausted on the deadline.
- Track A (from-scratch Urdu transformer) was **not** started.
- Stage 2 labels (politics, health, world news) need a second licensed corpus.

## Data

- Source: [Urdu News Dataset 1M](https://www.kaggle.com/datasets/saurabhshahane/urdu-news-dataset)
  (Mendeley v3, DOI [10.17632/834vsxnb99.3](https://data.mendeley.com/datasets/834vsxnb99/3)).
- License: [`data/licenses/urdu_news_dataset_1m.md`](data/licenses/urdu_news_dataset_1m.md)
  — non-commercial research; credit news sources.
- Raw CSV and split CSVs are **not** in this repo (see `.gitignore`).

Stage 1 audit: 111,860 kept headlines — sports 44,829, entertainment 34,901,
business 24,131, technology 7,999.

## Repo layout

| Path | Role |
| --- | --- |
| `SPEC.md` | Binding stage-1 specification |
| `notebooks/` | Training, prep, and finalist test eval |
| `kaggle/` | Private Kaggle GPU launchers |
| `experiments/registry.csv` | Every recorded run |
| `data/processed/*.txt` | Score reports and audits |
| `docs/decisions/` | ADRs |
| `tasks/plan.md` | Ladder checklist |

## Reproduction notes

1. Place the raw corpus at `data/raw/urdu-news-dataset-1M.csv`.
2. Run `notebooks/stage1_prepare.py`, then baselines, then ladder scripts.
3. Finalist test (already run once): `notebooks/stage1_finalist_test_eval.py`
   — do **not** re-run unless you intentionally open a new finalist.
4. Select on validation macro-F1 only during development.

## Project docs

- Spec: [`SPEC.md`](SPEC.md)
- Plan: [`tasks/plan.md`](tasks/plan.md)
- Finalist ADR: [`docs/decisions/ADR-001-final-finetune-uses-winning-settings.md`](docs/decisions/ADR-001-final-finetune-uses-winning-settings.md)

## Hugging Face

| Resource | URL |
| --- | --- |
| Live demo (Static Space) | https://huggingface.co/spaces/HammadHARahim/infinitic-urdu-headline-classifier |
| Finalist model | https://huggingface.co/HammadHARahim/urdu-headline-stage1-cosine |

Space UI files live in `docs/hf_space_static/` and sync to the Space on push
to `main` via `.github/workflows/sync-hf-space.yml` (uploads UI only; does not
delete Space-hosted `/models`).
