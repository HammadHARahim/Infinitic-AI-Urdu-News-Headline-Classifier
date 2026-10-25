# Comparison notes for client / LinkedIn use

Use these claims carefully. Different papers use different datasets, label sets, and metrics.

## Infinitic AI Stage 1 (measured)

| Metric | Value |
| --- | --- |
| Task | 4-class Urdu news category (sports, business, entertainment, technology) |
| Corpus | Urdu News Dataset 1M (research / non-commercial) |
| Validation macro-F1 | **0.9481** |
| Test macro-F1 | **0.9384** |
| Test accuracy | **0.9579** |
| Test size | 11,252 headlines (scored once) |

### Same-split baselines we beat

| System | Macro-F1 |
| --- | ---: |
| Majority | 0.1437 |
| Frozen encoder | 0.8538 |
| TF-IDF + logistic regression | 0.9230 |
| Best LoRA (validation) | 0.9395 |
| Infinitic finalist (validation) | **0.9481** |
| Infinitic finalist (test) | **0.9384** |

## External public numbers (context only)

| Source | Task | Reported score | Why not identical |
| --- | --- | --- | --- |
| Spark NLP `classifierdl_bert_news` (Urdu) | 7 news-like classes | macro-F1 ~0.75, acc ~0.82 | Different dataset and **more classes** |
| IMTIC 2021 BERT on iNLTK Urdu headlines | Headline classification | ~95% accuracy | Different corpus; accuracy ≠ macro-F1 |
| Ax-to-Grind / fake-news XLM-R papers | Fake vs real | F1 ~0.90–0.95 | **Different task** (misinformation, not category) |
| EnML (2023) multi-label Urdu news | Multi-label | F1 ~0.92 (ensemble) | Multi-label setup, different labels |

## Safe LinkedIn wording

- Prefer: “On our sealed 4-class Urdu News Dataset 1M split, we reach test macro-F1 **0.9384** and beat TF-IDF (**0.9230**) and LoRA.”
- Prefer: “Public 7-class Urdu news classifiers often report lower macro-F1 (e.g. Spark NLP ~0.75); our Stage 1 path is a focused 4-class production setup.”
- Avoid: “We beat every Urdu model ever published” or comparing to fake-news F1 as if it were category classification.
