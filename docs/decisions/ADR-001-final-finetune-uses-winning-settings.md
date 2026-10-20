# ADR-001: One final fine-tune uses the winning experiment settings

## Status
Accepted (deadline amendment 20 October 2026)

## Date
2026-10-16

## Context
Stage 1 experiments change one setting at a time and select on validation macro-F1. The test split stays sealed until a finalist is declared. The kept full-model checkpoint is the cosine schedule run, epoch 46, validation macro-F1 0.9481 on HammadHARahim/urdu-headline-stage1-cosine. The earlier linear warmup run peaked at 0.9466.

The project deadline is 20 October 2026. On that date Kaggle GPU quota was exhausted (0.00h remaining until 24 October 2026), so a fresh fine-tune from `xlm-roberta-base` could not be started.

## Decision
Keep the ladder as single-change experiments. In the normal process, after that ladder is finished, fine-tune once from xlm-roberta-base using only the settings whose own runs beat the kept validation macro-F1, then score the test split once.

**Deadline amendment:** for the 20 October 2026 Stage 1 delivery, declare the kept cosine checkpoint itself as the finalist and score the test split once on that checkpoint. Document that the preferred fresh retrain was skipped only because of GPU quota. Settings that lost their ladder comparisons (weight decay 0.1, class-weighted loss, label smoothing 0.1, LoRA) stay out of the recipe.

## Consequences
A setting that loses its comparison stays out of the final recipe. Under the deadline amendment, the delivered finalist is the validation-selected Hub checkpoint, not a new training job. Test macro-F1 is written once to the registry and the finalist test report.
