# ADR-001: One final fine-tune uses the winning experiment settings

## Status
Accepted

## Date
2026-10-16

## Context
Stage 1 experiments change one setting at a time and select on validation macro-F1. The test split stays sealed until a finalist is declared. The current kept full-model checkpoint is the cosine schedule run, epoch 46, validation macro-F1 0.9481 on HammadHARahim/urdu-headline-stage1-cosine. The earlier linear warmup run peaked at 0.9466.

## Decision
Keep the ladder as single-change experiments. After that ladder is finished, fine-tune once from xlm-roberta-base using only the settings whose own runs beat the kept validation macro-F1. That combined run is the finalist candidate. Score the test split once, after the finalist is declared.

## Consequences
A setting that loses its comparison stays out of the final recipe. The final run is a fresh training job, not a continuation of an earlier checkpoint. LoRA and other strategies enter that recipe only if their validation macro-F1 beats the kept full-model score.
