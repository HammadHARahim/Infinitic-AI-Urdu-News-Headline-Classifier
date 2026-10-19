---
name: intermediate-programmer-code
description: >-
  Writes Python and Jupyter notebook code that an intermediate programmer can
  read without asking for help. Use when writing, editing, or explaining
  notebooks, data preparation, training, or evaluation code in this project,
  or when the user asks for understandable, intermediate-level, or
  intermediary-programmer code.
---

# Intermediate programmer code

The code must be understandable by an intermediary programmer.

The reader can follow Python functions, `if` and `for`, dictionaries, lists, and basic pandas (`read_csv`, column access, `value_counts`). They can read a notebook from top to bottom. They are still learning Hugging Face training code. Write for that reader.

## How to write

1. Put one action on a line. Loading, cleaning, and counting stay in separate steps.
2. Name the value. Use `headline_text`, `label_name`, and `validation_macro_f1`.
3. A function does one job. Pass in every value it uses. Do not read notebook variables from inside the function.
4. Import only what that notebook section uses, at the top of the section.
5. Use a normal `for` loop or `if` when a one-line comprehension would be hard to read aloud.
6. Call libraries with named arguments. Leave Trainer setup as a direct call with those arguments. Add a custom class only when the notebook needs behavior the direct call cannot express.
7. Comment the reason for a step when the next line is not obvious, such as a hash split or a sealed test set. Do not comment `i = i + 1`.
8. Start each notebook section with a markdown heading: the gate name, and the file that section writes.
9. Show the check that proves the section worked, such as row counts or missing labels, and print a sentence that says what the number means.

## Examples

Label audit:

```python
import pandas as pd

headlines = pd.read_csv("data/raw/headlines.csv")
label_counts = headlines["label"].value_counts()
print("Rows per label:")
print(label_counts)
```

One cleaning step:

```python
def normalize_headline(headline_text):
    """Return the headline with one space between words and no edge spaces."""
    words = headline_text.split()
    cleaned_text = " ".join(words)
    return cleaned_text.strip()
```

A count written as a loop:

```python
def count_empty_labels(headlines):
    empty_label_count = 0
    for label_name in headlines["label"]:
        label_is_missing = label_name == "" or label_name is None
        if label_is_missing:
            empty_label_count = empty_label_count + 1
    return empty_label_count
```

A comment that explains a project rule:

```python
# Same normalized headline must stay in one split.
# Hash the cleaned text before any tokenizer or model is fit.
headline_hash = hash_normalized_headline(cleaned_headline)
```

A training call an intermediate programmer can scan:

```python
from transformers import TrainingArguments

training_arguments = TrainingArguments(
    output_dir="outputs/xlm-roberta-frozen",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    seed=42,
    eval_strategy="epoch",
)
```
