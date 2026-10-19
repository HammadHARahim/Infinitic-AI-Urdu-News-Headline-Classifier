"""Stage 1 frozen-encoder baseline.

The pretrained encoder stays fixed. Only the new classifier is updated.
The score is validation macro-F1. The test split is not read.
This run is for a T4. It is a separate run from any later A100 result.
"""

import os
import time

import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

SEED = 42
MODEL_NAME = "xlm-roberta-base"
TRAIN_PATH = "data/processed/stage1_train.csv"
VALIDATION_PATH = "data/processed/stage1_validation.csv"
CHECKPOINT_PATH = "outputs/stage1-frozen-t4/classifier.pt"
REPORT_PATH = "data/processed/stage1_frozen_encoder_t4.txt"
LABELS = ["sports", "business", "entertainment", "technology"]
BATCH_SIZE = 32
MAX_LENGTH = 64
MAX_EPOCHS = 2
LEARNING_RATE = 1e-3
TRAINING_BUDGET_SECONDS = 35 * 60


def write_lines(path, lines):
    """Write a text report."""
    with open(path, "w", encoding="utf-8") as report_file:
        report_file.write("\n".join(lines))
        report_file.write("\n")


def freeze_encoder(model):
    """Keep the pretrained encoder fixed."""
    for parameter in model.roberta.parameters():
        parameter.requires_grad = False


def count_parameters(model):
    """Return trainable and total parameter counts."""
    trainable_count = 0
    total_count = 0
    for parameter in model.parameters():
        parameter_count = parameter.numel()
        total_count = total_count + parameter_count
        if parameter.requires_grad:
            trainable_count = trainable_count + parameter_count
    return trainable_count, total_count


def label_tensor(label_names):
    """Turn label names into integer ids."""
    label_ids = []
    for label_name in label_names:
        label_ids.append(LABELS.index(label_name))
    return torch.tensor(label_ids)


def training_batches(headlines, label_ids, generator):
    """Yield shuffled batches for one epoch."""
    order = torch.randperm(len(headlines), generator=generator)
    start = 0
    while start < len(headlines):
        chosen = order[start:start + BATCH_SIZE]
        batch_headlines = []
        for index in chosen.tolist():
            batch_headlines.append(headlines[index])
        batch_labels = label_ids[chosen]
        start = start + BATCH_SIZE
        yield batch_headlines, batch_labels


def training_step(model, optimizer, tokenizer, headlines, label_ids, device):
    """Update the classifier on one batch."""
    model.train()
    model.roberta.eval()
    encoded = tokenizer(
        headlines,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    token_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    batch_labels = label_ids.to(device)
    optimizer.zero_grad()
    outputs = model(
        input_ids=token_ids,
        attention_mask=attention_mask,
        labels=batch_labels,
    )
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    return float(loss.detach().item())


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
    return predicted_names


def validation_macro_f1(model, tokenizer, headlines, label_names, device):
    """Score the validation headlines. This does not read the test split."""
    predicted_names = predict_labels(model, tokenizer, headlines, device)
    score = f1_score(
        label_names,
        predicted_names,
        average="macro",
        labels=LABELS,
    )
    return float(score)


def main():
    torch.manual_seed(SEED)
    if not torch.cuda.is_available():
        raise SystemExit("This run needs the T4. CUDA is not available.")
    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(0)
    print(f"Device: {device_name}", flush=True)

    train_rows = pd.read_csv(TRAIN_PATH, usecols=["normalized_headline", "label"])
    validation_rows = pd.read_csv(
        VALIDATION_PATH,
        usecols=["normalized_headline", "label"],
    )
    train_headlines = train_rows["normalized_headline"].tolist()
    train_label_ids = label_tensor(train_rows["label"].tolist())
    validation_headlines = validation_rows["normalized_headline"].tolist()
    validation_label_names = validation_rows["label"].tolist()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label={index: name for index, name in enumerate(LABELS)},
        label2id={name: index for index, name in enumerate(LABELS)},
    )
    model.to(device)
    freeze_encoder(model)
    trainable_count, total_count = count_parameters(model)
    trainable_parameters = []
    for parameter in model.parameters():
        if parameter.requires_grad:
            trainable_parameters.append(parameter)
    optimizer = AdamW(trainable_parameters, lr=LEARNING_RATE)
    generator = torch.Generator()
    generator.manual_seed(SEED)
    print(
        f"Trainable parameters: {trainable_count}. Total parameters: {total_count}.",
        flush=True,
    )

    started_at = time.perf_counter()
    best_score = -1.0
    best_epoch = 0
    best_classifier_state = None
    completed_epochs = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        elapsed = time.perf_counter() - started_at
        if elapsed > TRAINING_BUDGET_SECONDS:
            print("Stopped before another epoch to stay inside the T4 time budget.", flush=True)
            break
        batch_number = 0
        epoch_loss = 0.0
        for batch_headlines, batch_labels in training_batches(
            train_headlines,
            train_label_ids,
            generator,
        ):
            batch_loss = training_step(
                model,
                optimizer,
                tokenizer,
                batch_headlines,
                batch_labels,
                device,
            )
            epoch_loss = epoch_loss + batch_loss
            batch_number = batch_number + 1
            if batch_number % 100 == 0:
                print(
                    f"epoch {epoch} batch {batch_number}: loss {batch_loss:.4f}",
                    flush=True,
                )
        mean_loss = epoch_loss / batch_number
        score = validation_macro_f1(
            model,
            tokenizer,
            validation_headlines,
            validation_label_names,
            device,
        )
        completed_epochs = epoch
        print(
            f"epoch {epoch}: mean loss {mean_loss:.4f}, validation macro-F1 {score:.4f}",
            flush=True,
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_classifier_state = {
                name: value.detach().cpu().clone()
                for name, value in model.classifier.state_dict().items()
            }

    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    torch.save(
        {
            "model_name": MODEL_NAME,
            "labels": LABELS,
            "classifier_state": best_classifier_state,
            "epoch": best_epoch,
            "val_macro_f1": best_score,
        },
        CHECKPOINT_PATH,
    )
    elapsed_seconds = time.perf_counter() - started_at
    peak_gpu_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    report_lines = [
        "Stage 1 frozen-encoder baseline. The test split was not read.",
        f"Device: {device_name}",
        "Encoder: frozen. Only the new classifier was updated.",
        f"Trainable parameters: {trainable_count}",
        f"Total parameters: {total_count}",
        f"Learning rate: {LEARNING_RATE}",
        f"Batch size: {BATCH_SIZE}",
        f"Epochs completed: {completed_epochs}",
        f"Best epoch: {best_epoch}",
        f"Validation macro-F1: {best_score:.4f}",
        f"Seconds: {elapsed_seconds:.1f}",
        f"Peak memory: {peak_gpu_mb:.1f}MB GPU",
        f"Checkpoint: {CHECKPOINT_PATH}",
        "The checkpoint stores the trained classifier. The encoder is xlm-roberta-base.",
    ]
    write_lines(REPORT_PATH, report_lines)
    print("\n".join(report_lines), flush=True)


if __name__ == "__main__":
    main()
