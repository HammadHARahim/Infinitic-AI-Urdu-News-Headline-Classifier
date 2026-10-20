"""Score the Stage 1 finalist once on the sealed test split.

Deadline wrap-up (20 October 2026): the kept cosine checkpoint is the
finalist. A fresh fine-tune from base was not run because Kaggle GPU quota
was exhausted. This script reads the test split only for this declared
finalist evaluation.
"""

import time

import pandas as pd
import torch
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

SEED = 42
HUB_REPO = "HammadHARahim/urdu-headline-stage1-cosine"
TEST_PATH = "data/processed/stage1_test.csv"
REPORT_PATH = "data/processed/stage1_finalist_test.txt"
LABELS = ["sports", "business", "entertainment", "technology"]
BATCH_SIZE = 16
MAX_LENGTH = 64


def write_lines(path, lines):
    """Write a text report."""
    with open(path, "w", encoding="utf-8") as report_file:
        report_file.write("\n".join(lines))
        report_file.write("\n")


def predict_labels(model, tokenizer, headlines, device):
    """Predict a label name for every headline."""
    model.eval()
    predicted_names = []
    start = 0
    while start < len(headlines):
        batch_headlines = headlines[start:start + BATCH_SIZE]
        start = start + BATCH_SIZE
        encoded = tokenizer(
            batch_headlines,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        token_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        with torch.no_grad():
            outputs = model(
                input_ids=token_ids,
                attention_mask=attention_mask,
            )
        predicted_ids = torch.argmax(outputs.logits, dim=1).tolist()
        for predicted_id in predicted_ids:
            predicted_names.append(LABELS[predicted_id])
        if start % (BATCH_SIZE * 50) == 0 or start >= len(headlines):
            print(f"scored {min(start, len(headlines))} of {len(headlines)}", flush=True)
    return predicted_names


def main():
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
    else:
        device = torch.device("cpu")
        device_name = "cpu"
    print(f"Device: {device_name}", flush=True)
    print(
        "Finalist: kept cosine checkpoint. Test split opened once for this run.",
        flush=True,
    )

    test_rows = pd.read_csv(TEST_PATH, usecols=["normalized_headline", "label"])
    test_headlines = test_rows["normalized_headline"].tolist()
    true_labels = test_rows["label"].tolist()
    print(f"Test headlines: {len(test_headlines)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(HUB_REPO)
    model = AutoModelForSequenceClassification.from_pretrained(HUB_REPO)
    model.to(device)

    started_at = time.perf_counter()
    predicted_labels = predict_labels(model, tokenizer, test_headlines, device)
    elapsed_seconds = time.perf_counter() - started_at

    test_macro_f1 = float(
        f1_score(
            true_labels,
            predicted_labels,
            average="macro",
            labels=LABELS,
        )
    )
    report_text = classification_report(
        true_labels,
        predicted_labels,
        labels=LABELS,
        digits=4,
    )
    matrix = confusion_matrix(true_labels, predicted_labels, labels=LABELS)

    lines = [
        "Stage 1 Track B finalist test evaluation. Deadline wrap-up 20 October 2026.",
        "Finalist checkpoint: HammadHARahim/urdu-headline-stage1-cosine",
        "This is the kept cosine run, best validation epoch 46, validation macro-F1 0.9481.",
        "A fresh fine-tune from xlm-roberta-base was not run: Kaggle GPU quota was 0.00h.",
        "The test split was read only for this declared finalist evaluation.",
        f"Device: {device_name}",
        f"Test headlines: {len(test_headlines)}",
        f"Batch size: {BATCH_SIZE}",
        f"Max length: {MAX_LENGTH}",
        f"Test macro-F1: {test_macro_f1:.4f}",
        f"Seconds: {elapsed_seconds:.1f}",
        "",
        "Per-label report:",
        report_text.rstrip(),
        "",
        "Confusion matrix rows=true labels, columns=predicted labels.",
        "Label order: " + ", ".join(LABELS),
    ]
    for row in matrix:
        lines.append(" ".join(str(value) for value in row.tolist()))

    write_lines(REPORT_PATH, lines)
    print("\n".join(lines), flush=True)
    print(f"Wrote {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
