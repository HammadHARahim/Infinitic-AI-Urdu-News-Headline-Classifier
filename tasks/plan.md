# Implementation plan: stage 1 data

## Overview

Stage 1 prepares Urdu News Dataset 1M for a later four-class fine-tune. The download, audit, deduplication, and split come first. Fine-tuning stays behind the data gate, the majority and TF-IDF baselines, and the smoke test.

## Architecture decisions

- The source is the Kaggle copy of Urdu News Dataset 1M, which cites Mendeley version 3, DOI 10.17632/834vsxnb99.3. The license note is `data/licenses/urdu_news_dataset_1m.md`.
- Labels in this stage: sports, business, entertainment, technology.
- The public card shows a story field named News Text. The audit records whether a headline field exists. Training uses the headline field when the file has one. Otherwise it uses the story text and the run notes say so.
- Split by the hash of the normalized text. Same text stays in one split. Sizes: 80 percent train, 10 percent validation, 10 percent test. Seed 42. Test labels stay unread.
- The seven news sites stay blocked.

## Risks

- Kaggle may require a signed-in download. The file is not in `data/raw/` yet.
- The release is story text. A headline-only model needs that difference written on the run.
- Stage 2 labels are not in this file. They wait for a separate licensed corpus.

## Task list

### Data gate

- [x] Task 1: Download the Kaggle copy into `data/raw/`
- [x] Task 2: Audit columns, label counts, and empty text
- [x] Task 3: Normalize, deduplicate, split, and check leakage

### Checkpoint: data gate

- [x] Manifest names the Kaggle URL, the Mendeley citation, the download date, the license file, and the code version
- [x] Validation and test files exist, and test labels are not used to choose a model

### Later gates

- [x] Task 4: Majority and TF-IDF baselines on validation macro-F1
- [x] Tokenizer check on validation headlines (`data/processed/stage1_tokenizer.txt`)
- [x] Task 5 smoke test: one training batch overfit, then save and resume (`data/processed/stage1_smoke_test.txt`)
- [x] Task 5 ladder, first run: frozen encoder on a T4, validation macro-F1 0.8538 (`data/processed/stage1_frozen_encoder_t4.txt`)
- [x] Task 5 ladder, full fine-tune on an A100, validation macro-F1 0.9257 (`data/processed/stage1_full_finetune_a100.txt`)
- [ ] Task 5 ladder, full fine-tune on a T4: weights are on the private Hub repo; epoch 1 validation macro-F1 0.9293; the selected score was not copied back (`data/processed/stage1_full_finetune_t4.txt`)
- [x] Task 5 ladder, LoRA rank 4 on a T4, validation macro-F1 0.9021 (`data/processed/stage1_lora_r4_t4.txt`)
- [ ] Task 5 ladder, full fine-tune on a T4 for 9 epochs: 5 epochs scored, best validation macro-F1 0.9417 at epoch 4; stopped during epoch 6; weights were not uploaded (`data/processed/stage1_full_finetune_t4_9epochs.txt`)
- [ ] Task 5 ladder, full fine-tune on a T4 for 20 epochs: 5 epochs scored, best validation macro-F1 0.9416 at epoch 3; Google closed the session during epoch 6 (`data/processed/stage1_full_finetune_t4_20epochs.txt`)
- [ ] Task 5 ladder, full fine-tune on a T4 at learning rate 5e-6: 9 epochs scored, best validation macro-F1 0.9449 at epoch 4; stopped during epoch 13 (`data/processed/stage1_full_finetune_t4_lr5e6.txt`)
- [ ] Task 5 ladder, same checkpoint with weight decay 0.1: 15 epochs scored, best validation macro-F1 0.9407 at epoch 19; did not beat 0.9449; resume upload hit the private storage limit (`data/processed/stage1_full_finetune_t4_wd0p1.txt`)
- [ ] Task 5 ladder, same checkpoint with class-weighted loss: epochs 5 to 9 scored, best validation macro-F1 0.9430 at epoch 5; did not beat 0.9449; stopped during epoch 10 (`data/processed/stage1_full_finetune_t4_classweight.txt`)
- [ ] Task 5 ladder, fresh fine-tune from xlm-roberta-base with linear warmup for 10% of steps then linear decay to zero: 20 epochs scored, best validation macro-F1 0.9466 at epoch 19; kept checkpoint (`data/processed/stage1_full_finetune_t4_warmup.txt`)
- [ ] Task 5 ladder: rerun LoRA rank 4 with label smoothing 0.1: 37 epochs scored, best validation macro-F1 0.9377 at epoch 30; early stopping fired after epoch 37; did not beat 0.9466 (`data/processed/stage1_lora_r4_smooth.txt`)
- [ ] Task 5 ladder: full fine-tune from xlm-roberta-base at learning rate 1e-4 and label smoothing 0.1: 5 epochs scored at the majority baseline 0.1437; stopped during epoch 6; weight file deleted (`data/processed/stage1_full_finetune_t4_smooth60_1e4.txt`)
- [ ] Task 5 ladder: fresh LoRA rank 4 through epoch 100: 100 epochs scored, best validation macro-F1 0.9395 at epoch 56; epoch 100 scored 0.9369; did not beat 0.9466 (`data/processed/stage1_lora_r4_50.txt`)
- [ ] Task 5 ladder: LoRA rank 8, same settings as the rank-4 ladder run: 2 epochs scored, best validation macro-F1 0.9042 at epoch 2; beat the rank-4 ladder score 0.9021; did not beat 0.9466 (`data/processed/stage1_lora_r8_t4.txt`)
- [ ] Task 5 ladder: LoRA rank 8 for 30 epochs, same settings, new adapter: 30 epochs scored, best validation macro-F1 0.9377 at epoch 26; did not beat 0.9395 or 0.9466 (`data/processed/stage1_lora_r8_30.txt`)
- [ ] Task 5 ladder: LoRA rank 16, same settings as the rank-4 ladder run: 2 epochs scored, best validation macro-F1 0.9120 at epoch 2; beat rank 4 at 0.9021 and rank 8 at 0.9042; did not beat 0.9466 (`data/processed/stage1_lora_r16_t4.txt`)
- [ ] Task 5 ladder: full fine-tune with half-cosine decay then constant 2e-6: 48 epochs scored, best validation macro-F1 0.9481 at epoch 46; beat linear warmup 0.9466; session stopped after epoch 48 (`data/processed/stage1_full_finetune_t4_cosine.txt`)
- [ ] Task 5 ladder: kept cosine recipe with label smoothing 0.1 as the one change: 60 epochs scored, best validation macro-F1 0.9466 at epoch 36; did not beat 0.9481 (`data/processed/stage1_full_finetune_t4_cosine_smooth.txt`)

Fine-tuning is Task 5. It does not start in the data phase.

### Final training run

The ladder changes one setting at a time. A setting joins the final recipe only when its own run beats the kept validation macro-F1.

### Deadline wrap-up (20 October 2026)

- [x] Declare Stage 1 Track B finalist: kept cosine checkpoint (validation macro-F1 0.9481, epoch 46)
- [x] Score the sealed test split once: test macro-F1 0.9384 (`data/processed/stage1_finalist_test.txt`)
- [x] Update README and registry with the test score

Fresh fine-tune from `xlm-roberta-base` with only winning settings was the preferred finalist job (ADR-001). It was skipped on the deadline because Kaggle GPU quota was 0.00h. The kept cosine Hub checkpoint is the delivered finalist.

The kept full-model recipe is the cosine run: full fine-tune, linear warmup for 10% of steps then half-cosine decay for 20 epochs, then constant learning rate 2e-6, weight decay 0.01, unweighted cross-entropy, batch size 32, max length 64, seed 42. Best epoch 46, validation macro-F1 0.9481, checkpoint `HammadHARahim/urdu-headline-stage1-cosine`. Weight decay 0.1, class-weighted loss, and label smoothing 0.1 did not beat the scores they were tested against, so they stay out.
