# Urdu News Headline Classification with Parameter-Efficient Fine-Tuning

LLM-6XX Assignment 1 | Group project document | Deadline: October 20, 2026

Group members: Member A: \_\_\_\_\_\_ | Member B: \_\_\_\_\_\_ | Member C: \_\_\_\_\_\_

## 1. Project Description

**Problem.** Given an Urdu news headline, predict its news category (sports, business, entertainment, politics, technology, health, world). Urdu is a low-resource language written in a right-to-left Perso-Arabic script, and tokenizers built mostly on English text split Urdu words into many small pieces. This project measures how well multilingual pretrained transformers handle that, and how much of their quality can be kept while training only a small fraction of their parameters.

**What we build.** A complete data-to-model pipeline: (1) scrape about 2,000-3,000 labelled Urdu headlines from public news sites, (2) clean, deduplicate and split them, (3) fine-tune pretrained models with several strategies (frozen encoder, full fine-tuning, LoRA at several ranks), (4) run a documented ladder of experiments, and (5) evaluate on a held-out test set with macro-F1 and a confusion matrix.

**Research questions.**

- RQ1: How close does LoRA get to full fine-tuning, and with how many trainable parameters?
- RQ2: How does the tokenizer affect Urdu text (tokens per word) and downstream F1?
- RQ3: Does a zero-shot LLM prompt beat a small fine-tuned encoder on this task?
- RQ4: Do regularization and data-side fixes (class balancing, deduplication) help?

**Rubric coverage.** Dataset (10), preprocessing and visualization (10), model and architecture (10), training and monitoring (10), experiments (20), testing and metrics (10), report (10), viva (20, individual).

## 2. End-to-End Pipeline

| Stage | Input | Output | Owner |
| --- | --- | --- | --- |
| 1. Scrape | News section URLs | raw.csv (text, label, site, url, date) | A |
| 2. Clean and dedupe | raw.csv | clean.csv | A |
| 3. Split | clean.csv | train.csv, val.csv, test.csv, label2id.json | A |
| 4. Visualize | splits | class, length and token plots | A |
| 5. Model setup | splits | architecture printout, trainable parameter counts | B |
| 6. Baselines and fine-tuning | splits | trained models, per-epoch logs, curves | B |
| 7. Experiments (regularization, data-side, tokenizer, other model) | splits | results table (results.csv) | C |
| 8. Final test, error analysis | best model, test.csv | metrics, confusion matrix, correct and wrong samples | C |
| 9. Report and packaging | all outputs | 2-page PDF, executed notebook, weights, dataset | C (with A and B reviewing) |

**Shared contract (so the three parts plug together).**

- Files: train.csv, val.csv, test.csv with columns text, label, site, url. Labels are strings; label2id.json maps them to integers.
- Random seed 42 everywhere. Max sequence length 64 tokens for headline-only input.
- Every experiment appends one row to results.csv: name, model, strategy, trainable\_params, lr, epochs, val\_macro\_f1, test\_macro\_f1 (test filled only for the final model, see rule below), notes.
- **Test-set rule:** the test set is used once per finalist model, after all tuning is done. All comparison between experiments uses validation macro-F1. This is the no-leakage rule the rubric checks.

**Assembling one notebook.** The assignment requires one notebook with sections in this order: Scraping, Preprocessing, Visualization, Model development, Training, Testing. Each member writes their cells in a separate notebook with the same section headers, then Member C merges them in that order and re-runs top to bottom on a fresh Colab runtime so outputs are retained and the notebook runs without errors.

**Setup cell (everyone).**

```python
!pip -q install transformers peft datasets evaluate accelerate scikit-learn beautifulsoup4 lxml
import random, numpy as np, torch
random.seed(42); np.random.seed(42); torch.manual_seed(42)
```

## Section A (Member A): Data - Scraping, Preprocessing, Visualization

**Owns notebook sections 1, 2, 3. Rubric: 10 + 10 marks.**

### A1. Collection

- Sources: use 2-3 Urdu news sites (for example BBC Urdu, Express Urdu, Geo Urdu, Jang) so the model does not learn one site's style. Check each site's robots.txt and terms of use, and record the result in the documentation.
- Labels come from the site section the article is listed under (sports, business, and so on). This is automatic labelling; document it as the labelling method.
- Collect headline, first paragraph if available (used later for the headline-only vs. headline+lead experiment), URL, site, date.
- Target 300+ headlines per class, 2,000-3,000 total. If a class has fewer than about 150, merge it into a neighbour class or drop it.
- Wait 1-2 seconds between requests and send a normal User-Agent header.
- If a site blocks scraping, supplement with a public Urdu news dataset from Hugging Face. Cite its name, source and licence in the notebook.

```python
import requests, time, pandas as pd
from bs4 import BeautifulSoup

def scrape(url, label, site, selector):
    html = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15).text
    soup = BeautifulSoup(html, 'lxml')
    rows = [{'text': a.get_text(strip=True), 'url': a.get('href'), 'label': label, 'site': site}
            for a in soup.select(selector) if a.get_text(strip=True)]
    time.sleep(1.5)
    return rows
```

The CSS selector differs per site: open the section page, inspect a headline element, and write one selector per site. Loop over (section URL, label) pairs and paginate where the site allows.

### A2. Preprocessing

```python
import re
MAP = str.maketrans({'ي': 'ی', 'ى': 'ی', 'ك': 'ک', 'ۃ': 'ہ'})
def clean(t):
    t = re.sub(r'https?://\S+|@\w+', ' ', t)
    t = t.translate(MAP)
    t = re.sub(r'[\u064B-\u065F\u0670]', '', t)   # Arabic diacritics
    return re.sub(r'\s+', ' ', t).strip()

df['text'] = df.text.map(clean)
df = df[df.text.str.split().str.len() >= 3].drop_duplicates('text')
```

- Show before and after samples, row counts after each step, and how many duplicates were removed.
- Manual label audit: the team reads 50 random headlines and counts how many carry the wrong section label. Report this as the label-noise estimate; it is a strong point for the report and viva.

### A3. Split (before any model sees the data)

```python
from sklearn.model_selection import train_test_split
train, tmp = train_test_split(df, test_size=0.2, stratify=df.label, random_state=42)
val, test = train_test_split(tmp, test_size=0.5, stratify=tmp.label, random_state=42)
for n, d in [('train', train), ('val', val), ('test', test)]:
    d.to_csv(f'{n}.csv', index=False)
```

Check that no headline appears in more than one split.

### A4. Visualization (all outputs must stay visible in the saved notebook)

- Class distribution per split (bar chart) and per source site.
- Headline length in words and in tokens (histograms).
- Tokenizer comparison: average tokens per word for xlm-roberta-base, bert-base-multilingual-cased, and bert-base-uncased on the training headlines. Show one headline tokenized by each.
- Sample rows (head of each class) and the most frequent words per class.

### A5. Documentation block (markdown cell)

Sources and URLs, date of collection, labelling method, label-noise estimate, split sizes, licensing and privacy note (public headlines only, no personal data, used for non-commercial coursework).

**Definition of done:** train.csv, val.csv, test.csv and label2id.json exist; the scraper cell runs; documentation block is complete; plots are rendered.

**Viva questions Member A must answer:** Why label by URL section and what noise does that add? Why split before preprocessing statistics? What does deduplication prevent? Why did you normalize those specific characters? What does the tokenizer comparison show?

## Section B (Member B): Model Development and Training

**Owns notebook sections 4 and 5. Rubric: 10 + 10 marks, plus the core of the 20 experiment marks.**

### B1. Model choices and why

- **Main model: xlm-roberta-base** (about 278M parameters). Pretrained on 100 languages including Urdu, a SentencePiece vocabulary of 250k tokens, and a standard encoder that is easy to fine-tune and explain.
- **Second model for comparison: bert-base-multilingual-cased** (or MuRIL if the team prefers).
- **LLM baseline: Qwen2.5-1.5B-Instruct**, prompted zero-shot and few-shot with the label list. Expect it to be weaker on Urdu; that is a result worth reporting, not a failure.

### B2. Architecture

XLM-R encoder (12 layers, hidden size 768) plus a classification head over the first token with one output per class. Print the structure with print(model) and print the trainable and total parameter counts for every strategy.

### B3. One reusable training function

```python
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, EarlyStoppingCallback)
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def metrics(p):
    y = p.predictions.argmax(-1)
    return {'accuracy': accuracy_score(p.label_ids, y),
            'macro_f1': f1_score(p.label_ids, y, average='macro')}

def build(model_name, lora_r=None, freeze=False):
    m = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(label2id), id2label=id2label, label2id=label2id)
    if freeze:
        for p in m.base_model.parameters(): p.requires_grad = False
    if lora_r:
        m = get_peft_model(m, LoraConfig(task_type='SEQ_CLS', r=lora_r, lora_alpha=2*lora_r,
                                         lora_dropout=0.1, target_modules=['query', 'value']))
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    total = sum(p.numel() for p in m.parameters())
    print(f'trainable {trainable:,} / {total:,} ({100*trainable/total:.2f}%)')
    return m

def train(name, model_name='xlm-roberta-base', lr=2e-5, epochs=5, wd=0.0, ls=0.0, **kw):
    tok = AutoTokenizer.from_pretrained(model_name)
    # tokenize train_ds / val_ds with max_length=64, truncation=True (done once per model)
    args = TrainingArguments(output_dir=name, learning_rate=lr, num_train_epochs=epochs,
        per_device_train_batch_size=32, per_device_eval_batch_size=64, weight_decay=wd,
        label_smoothing_factor=ls, warmup_ratio=0.1, lr_scheduler_type='linear',
        eval_strategy='epoch', save_strategy='epoch', logging_strategy='epoch',
        load_best_model_at_end=True, metric_for_best_model='macro_f1', fp16=True, seed=42,
        report_to='none')
    tr = Trainer(model=build(model_name, **kw), args=args, train_dataset=train_ds,
                 eval_dataset=val_ds, compute_metrics=metrics,
                 callbacks=[EarlyStoppingCallback(2)])
    tr.train()
    return tr
```

The LoRA learning rate should be higher (about 2e-4) than for full fine-tuning (2e-5). Keep these values in the hyperparameter table.

### B4. Training outputs required by the rubric

- Per-epoch training loss, validation loss, validation accuracy and macro-F1 printed by the Trainer.
- Training accuracy per epoch: evaluate on a fixed 500-sample subset of the training set at the end of each epoch (a small callback calling trainer.evaluate with metric\_key\_prefix='train').
- Learning-curve plots (loss and accuracy, train vs. validation) drawn from trainer.state.log\_history.

### B5. Experiments owned by B (all rows go into results.csv)

| ID | Setup |
| --- | --- |
| E1 | Zero-shot and few-shot prompting with Qwen2.5-1.5B-Instruct (evaluate on val) |
| E2 | XLM-R-base, frozen encoder, train head only |
| E3 | XLM-R-base, full fine-tuning (reference baseline) |
| E4a-c | XLM-R-base + LoRA, r = 4, 8, 16 |

For LoRA, record trainable parameters next to macro-F1 so the trade-off is visible. If the GPU runs out of memory, lower the batch size to 16 and keep max length 64.

**Definition of done:** architecture printed for each strategy; curves plotted for E3 and the best LoRA run; results.csv has rows E1-E4c; all outputs are visible.

**Viva questions Member B must answer:** What does LoRA do and what are r and alpha? Why target the query and value matrices? Which layers are trainable in each strategy? Why is the LoRA learning rate higher? What would early stopping at epoch 2 mean in your curves? Why did the zero-shot LLM score as it did?

## Section C (Member C): Experiments, Testing, Report and Submission

**Owns notebook section 6 (Testing), the extra experiments, and the report. Rubric: 10 + 10 marks, plus the improvement half of the 20 experiment marks.**

### C1. Improvement experiments (use the best setup from B as the starting point)

| ID | Technique | What it tests |
| --- | --- | --- |
| E5 | Weight decay 0.01 and label smoothing 0.1 | Regularization |
| E6 | Learning-rate schedule: linear vs. cosine, warmup 0 vs. 0.1; batch size 16 vs. 32 | Optimization |
| E7 | Class balancing (class weights or oversampling), only if classes are imbalanced | Data-side |
| E8 | Headline only vs. headline + first paragraph (max length 128) | Input length |
| E9 | Different pretrained model (mBERT or MuRIL) with the best strategy | Model comparison |
| E10 | Tokenizer effect: English-only model (bert-base-uncased) on the same data | Low-resource script effect |

Keep every run in the notebook, including weak ones, with one sentence on why it behaved that way. Optionally repeat the final two or three configurations with three seeds and report mean and standard deviation.

### C2. Final testing (test set, once)

- Choose the final model by validation macro-F1 only, then evaluate on test.csv.
- Report accuracy, macro-F1, per-class precision, recall and F1 (classification\_report), and a confusion matrix plot.
- Show 5 correct and 5 incorrect predictions with the true and predicted label, and explain the errors (expected confusions: politics vs. world, business vs. politics, entertainment vs. sports for celebrity or cricket-league stories).

```python
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
pred = trainer.predict(test_ds)
y = pred.predictions.argmax(-1)
print(classification_report(pred.label_ids, y, target_names=labels))
ConfusionMatrixDisplay.from_predictions(pred.label_ids, y, display_labels=labels, xticks_rotation=45)
```

### C3. Two-page report (PDF, must not exceed two pages)

Use small fonts and a compact table; one figure at most.

1. **Dataset:** sources, size, classes, split sizes, preprocessing, label-noise estimate.
2. **Model:** base model, classification head, LoRA configuration, why chosen.
3. **Hyperparameters:** optimizer (AdamW), learning rates, batch size, epochs, scheduler and warmup, regularization, max length, seed, hardware (Colab T4).
4. **Results table:** all experiments E1-E10 with trainable parameters, validation macro-F1, and test result for the final model; highlight techniques that helped.
5. **Observations:** what worked, what did not, limitations (label noise from section labelling, small dataset, source bias), future work (larger dataset, Roman Urdu, bigger multilingual models).

### C4. Packaging checklist

- Executed notebook (.ipynb) with all outputs kept, run top to bottom on a fresh runtime without errors.
- Report PDF, two pages.
- Dataset files or a download link plus the scraper; model weights (LoRA adapters are small) or a link.
- File names include the project title and all group member IDs; one submission per group.

**Definition of done:** all experiments in results.csv; final test evaluation complete; notebook merged and re-run; report is exactly two pages.

**Viva questions Member C must answer:** Why is macro-F1 better than accuracy here? Why was the test set used only once? Which experiment helped most and why? What does the confusion matrix tell you about the data? How did the tokenizer change results?

## 3. Timeline (October 11-20)

| Dates | Member A | Member B | Member C |
| --- | --- | --- | --- |
| Oct 11-13 | Scrape, clean, split, plots | Set up training function on a small sample; write the LLM prompt baseline | Draft report skeleton; prepare results.csv and evaluation cells |
| Oct 14-15 | Label audit, documentation block, fix data issues | E1-E3, curves | Review A's outputs; prepare E5-E10 code |
| Oct 16-17 | Help run E7 and E10 | E4a-c, architecture printouts | E5-E10 |
| Oct 18 | Check leakage and split integrity | Final training of the chosen model | Final test evaluation, error analysis |
| Oct 19 | Review notebook | Review notebook | Merge, re-run, fill report table, export PDF, package |
| Oct 20 | Submit |  |  |

Submit with margin: aim to finish packaging on October 19.

## 4. Viva Preparation (20 marks, individual)

Every member is graded on the whole pipeline, and agent-generated code must also be explainable. Before submission:

- Each member presents their own section for five minutes to the other two, who ask the viva questions above.
- Each member re-runs the notebook once and traces one example from raw headline to token IDs to prediction.
- Everyone can state: dataset size and class counts, the final model, trainable parameter count, best hyperparameters, final test macro-F1, and the main limitation.

## 5. Risks and Fallbacks

| Risk | Fallback |
| --- | --- |
| Site blocks scraping or changes layout | Use another section or site; add a public Hugging Face Urdu news dataset and cite it |
| Classes too small or imbalanced | Merge sparse classes; run E7 (class balancing) |
| Colab out-of-memory or session timeout | Batch size 16, max length 64, fp16, save checkpoints to Drive each epoch |
| Zero-shot LLM unusable on Urdu | Report it honestly as the baseline result; try few-shot with 2 examples per class |
| High label noise from section labels | Report the audit result; optionally relabel the 50 audited headlines and discuss |
| Report exceeds two pages | Cut the figure, shrink the table font, merge hyperparameters into the results table |
