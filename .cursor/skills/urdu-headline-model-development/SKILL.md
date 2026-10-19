---
name: urdu-headline-model-development
description: Guides the phased development and evaluation of Infinitic AI's Urdu news headline classifiers, including a transformer trained from scratch and a pretrained model fine-tuned for classification. Use when planning, implementing, training, evaluating, or documenting either model track in this project.
---

# Urdu Headline Model Development

Build reproducible evidence that Infinitic AI can both train a compact Urdu
transformer from scratch and fine-tune a pretrained multilingual transformer.

## Non-negotiable experiment rules

- Use seed 42 unless a run is explicitly part of a multi-seed study.
- Split by normalized-content hash before fitting tokenizers or models.
- Keep near-duplicates and repeated URLs in one split.
- Select models and hyperparameters using validation macro-F1 only.
- Do not inspect test labels during development. Evaluate each finalist once.
- Record every completed run, including failures and regressions.
- Compare models on the same immutable test split and label mapping.
- Save provenance: source URL, collection date, license/terms notes, and code version.
- Never claim a model was trained "from scratch" if its tokenizer or weights were pretrained.

## Two distinct model tracks

### Track A: Urdu transformer from scratch

Train an Urdu tokenizer only on the permitted training corpus plus clearly
documented unlabeled Urdu pretraining text. Initialize all transformer weights
randomly, pretrain with masked-language modeling, then train a headline
classification head.

Always report:

- tokenizer algorithm, vocabulary size, corpus size, and tokens per Urdu word;
- architecture, parameter count, context length, objective, and compute;
- pretraining and downstream learning curves;
- downstream validation/test metrics and comparison with the fine-tuned model.

Use a compact architecture that the available compute can train credibly.
Do not imitate XLM-R-base scale without the data and compute to support it.

### Track B: pretrained model fine-tuning

Start with XLM-R-base unless an experiment justifies another model. Establish:

1. frozen-encoder baseline;
2. full fine-tuning baseline;
3. LoRA ranks 4, 8, and 16;
4. regularization, optimization, balancing, input-length, and model comparisons.

Report trainable and total parameters for every strategy.

## Phase gates

Do not advance a track until its current gate passes.

1. **Specification** — labels, success metrics, compute budget, corpora,
   and acceptance criteria are written down.
2. **Data** — collection is reproducible; normalization, audit, deduplication,
   immutable splits, and leakage checks pass.
3. **Baselines** — majority and TF-IDF/linear baselines are measured before
   transformer claims are made.
4. **Tokenization** — fragmentation and coverage are measured on held-out Urdu.
5. **Model smoke test** — one batch can overfit and a short run can save/resume.
6. **Training ladder** — only one controlled variable changes per experiment.
7. **Final evaluation** — finalist chosen on validation data; test metrics,
   confusion matrix, calibration, latency, and error analysis are produced.
8. **Packaging** — configs, seeds, environment, checkpoints/adapters, model card,
   dataset card, report, and runnable inference example are complete.

## Run record

Append one row per run to the experiment registry with:

`run_id, track, model, tokenizer, data_version, strategy, seed, trainable_params,
total_params, learning_rate, batch_size, epochs_or_steps, val_macro_f1,
test_macro_f1, runtime, peak_memory, checkpoint, notes`

Leave `test_macro_f1` empty except for declared finalist evaluations.

## Evidence standard

For every claim, link it to an artifact: dataset manifest, config, log, plot,
metric file, checkpoint, or report section. Label planned or unrun work
explicitly; never invent metrics.
