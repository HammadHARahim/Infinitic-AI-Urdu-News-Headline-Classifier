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

Full fine-tune of `xlm-roberta-base` — the overall best validation run so far:

| Setting | Value |
| --- | --- |
| Schedule | Warmup 10% of a **20-epoch** half-cosine, then constant **2e-6** from epoch 21 |
| Peak learning rate | **2e-5** |
| Weight decay | **0.01** |
| Loss | Unweighted cross-entropy (no label smoothing) |
| Batch size / max length / seed | **32** / **64** / **42** |
| Planned epochs | **50** |
| Epochs completed | **48** (session stopped before 50) |
| Best epoch | **46** |
| Best validation macro-F1 | **0.9481** |
| Checkpoint | Hub `HammadHARahim/urdu-headline-stage1-cosine` (private weights) |

Label smoothing 0.1, weight decay 0.1, and class-weighted loss were tried as
sole changes and did **not** beat 0.9481, so they stay out of the recipe.

## Results report (validation macro-F1)

Test metrics are empty on purpose. The test split has not been read for
selection.

### Baselines

| Run | Validation macro-F1 |
| --- | ---: |
| Majority (train majority = sports) | 0.1437 |
| TF-IDF + logistic regression | 0.9230 |

### Epoch log (main Track B runs)

`Epochs scored` = how many full epochs finished and wrote a validation score.
`Best epoch` = epoch with the highest validation macro-F1 in that run.

| Run | Epochs scored | Best epoch | Val macro-F1 | Notes |
| --- | ---: | ---: | ---: | --- |
| Frozen encoder (T4) | 2 | 1 | 0.8538 | Encoder frozen |
| Full fine-tune (A100) | 2 | 2 | 0.9257 | Short baseline |
| LoRA rank 4 | 2 | 2 | 0.9021 | Ladder row |
| LoRA rank 8 | 2 | 2 | 0.9042 | Ladder row |
| LoRA rank 16 | 2 | 2 | 0.9120 | Ladder row |
| Full T4 (early long runs) | 5 | 4 | 0.9417 | Planned 9; stopped mid-run |
| Full T4 (20-epoch attempt) | 5 | 3 | 0.9416 | Session closed early |
| Full T4, LR 5e-6 | 9 | 4 | 0.9449 | Early constant-LR peak |
| Full T4, weight decay 0.1 | 15 | 19* | 0.9407 | Did not beat 0.9449 |
| Full T4, class weights | 5 | 5 | 0.9430 | Did not beat 0.9449 |
| Warmup + linear decay | **20** | **19** | **0.9466** | Previous kept best |
| LoRA r4 + label smoothing | 37 | 30 | 0.9377 | Early stop after 7 flat epochs |
| Full model, LR 1e-4 + smoothing | 5 | — | 0.1437 | Collapsed to majority |
| LoRA r4 + smoothing (long) | **100** | **56** | **0.9395** | Best LoRA; still below full model |
| LoRA rank 8 (30 epochs) | 30 | 26 | 0.9377 | Below best LoRA |
| **Cosine + 2e-6 continuation** | **48** | **46** | **0.9481** | **Kept overall best** (planned 50) |
| Cosine + label smoothing 0.1 | **60** | **36** | **0.9466** | Completed planned 60; did not beat 0.9481 |

\* Weight-decay run notes refer to resumed epoch numbering from an earlier
checkpoint; see `data/processed/stage1_full_finetune_t4_wd0p1.txt`.

### Headline comparison

| | Warmup + linear | Cosine (kept) | Cosine + smoothing |
| --- | ---: | ---: | ---: |
| Epochs scored | 20 | **48** | **60** |
| Best epoch | 19 | **46** | 36 |
| Val macro-F1 | 0.9466 | **0.9481** | 0.9466 |
| Beat previous kept? | yes (vs 0.9449) | **yes (vs 0.9466)** | no |

Full run log: [`experiments/registry.csv`](experiments/registry.csv).  
Per-run text reports: `data/processed/stage1_*.txt`.

### Status

- Track B Stage 1 ladder is active. Cosine+smoothing finished all **60**
  epochs; best remains **0.9466** (below kept **0.9481**).
- Next unused skill-list items: **input max length** and **model comparison**.
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
