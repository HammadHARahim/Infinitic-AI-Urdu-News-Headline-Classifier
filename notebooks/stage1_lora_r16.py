"""Stage 1 LoRA rank 16.

The pretrained weights stay frozen. Trainable pieces are the rank-16 LoRA
updates on the query and value matrices, plus the new classifier.
lora_alpha equals the rank, so the LoRA scale stays 1.
Learning rate, batch size, and epoch count match the rank-4 ladder run.
The score is validation macro-F1. The test split is not read.
"""

import os
import time

import pandas as pd
import torch
from huggingface_hub import HfApi
from peft import LoraConfig
from peft import TaskType
from peft import get_peft_model
from sklearn.metrics import f1_score
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

SEED = 42
MODEL_NAME = "xlm-roberta-base"
TRAIN_PATH = "data/processed/stage1_train.csv"
VALIDATION_PATH = "data/processed/stage1_validation.csv"
CHECKPOINT_DIR = "outputs/stage1-lora-r16"
REPORT_PATH = "data/processed/stage1_lora_r16_t4.txt"
HUB_REPO = "HammadHARahim/urdu-headline-stage1-lora-r16"
LABELS = ["sports", "business", "entertainment", "technology"]
BATCH_SIZE = 32
MAX_LENGTH = 64
MAX_EPOCHS = 2
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
LORA_RANK = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
TRAINING_BUDGET_SECONDS = 40 * 60


def write_lines(path, lines):
    """Write a text report."""
    with open(path, "w", encoding="utf-8") as report_file:
        report_file.write("\n".join(lines))
        report_file.write("\n")


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
    """Update the LoRA weights and classifier on one batch."""
    model.train()
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


def report_lines(device_name, trainable_count, total_count, epoch_rows, best_epoch, best_score, elapsed_seconds, peak_gpu_mb):
    """Build the text record for this run."""
    lines = [
        "Stage 1 LoRA rank 16. The test split was not read.",
        "The rank-4 ladder run scored 0.9021. The rank-8 ladder run scored 0.9042.",
        "The only change is the rank. Alpha equals the rank.",
        f"Device: {device_name}",
        f"Runtime: {os.environ.get('TRAINING_RUNTIME', 'not set')}",
        "Encoder: frozen. LoRA updates query and value. The classifier is trained.",
        f"LoRA rank: {LORA_RANK}",
        f"LoRA alpha: {LORA_ALPHA}",
        f"LoRA dropout: {LORA_DROPOUT}",
        f"Trainable parameters: {trainable_count}",
        f"Total parameters: {total_count}",
        f"Learning rate: {LEARNING_RATE}",
        f"Weight decay: {WEIGHT_DECAY}",
        f"Batch size: {BATCH_SIZE}",
        f"Epochs completed: {len(epoch_rows)}",
        f"Best epoch: {best_epoch}",
        f"Validation macro-F1: {best_score:.4f}",
    ]
    for epoch, mean_loss, score in epoch_rows:
        lines.append(
            f"Epoch {epoch}: mean loss {mean_loss:.4f}, validation macro-F1 {score:.4f}"
        )
    lines.append(f"Seconds: {elapsed_seconds:.1f}")
    lines.append(f"Peak memory: {peak_gpu_mb:.1f}MB GPU")
    lines.append(f"Checkpoint: {CHECKPOINT_DIR}")
    lines.append(f"Hugging Face repo: {HUB_REPO}")
    lines.append("The Hub repo is private. The training corpus is non-commercial research.")
    return lines


def publish_score_log(lines):
    """Upload the score log so a stalled connection does not lose it."""
    text = "\n".join(lines) + "\n"
    HfApi().upload_file(
        path_or_fileobj=text.encode("utf-8"),
        path_in_repo="epoch_scores.txt",
        repo_id=HUB_REPO,
        commit_message="Save epoch validation scores",
    )


def publish_best_adapter(model, tokenizer, best_epoch, best_score):
    """Upload the best LoRA adapter. The frozen base model is not uploaded."""
    model.push_to_hub(
        HUB_REPO,
        private=True,
        commit_message=f"Best epoch {best_epoch} validation macro-F1 {best_score:.4f}",
    )
    tokenizer.push_to_hub(HUB_REPO, private=True)


def main():
    torch.manual_seed(SEED)
    if not torch.cuda.is_available():
        raise SystemExit("This run needs the T4. CUDA is not available.")
    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(0)
    print(f"Device: {device_name}", flush=True)
    HfApi().create_repo(repo_id=HUB_REPO, private=True, exist_ok=True)

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
    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label={index: name for index, name in enumerate(LABELS)},
        label2id={name: index for index, name in enumerate(LABELS)},
    )
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=["query", "value"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        modules_to_save=["classifier"],
    )
    model = get_peft_model(base_model, lora_config)
    model.to(device)
    trainable_count, total_count = count_parameters(model)
    trainable_parameters = []
    for parameter in model.parameters():
        if parameter.requires_grad:
            trainable_parameters.append(parameter)
    optimizer = AdamW(
        trainable_parameters,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    generator = torch.Generator()
    generator.manual_seed(SEED)
    print(
        f"Trainable parameters: {trainable_count}. Total parameters: {total_count}.",
        flush=True,
    )
    print(f"Next epoch: 1 of {MAX_EPOCHS}.", flush=True)

    started_at = time.perf_counter()
    best_score = -1.0
    best_epoch = 0
    epoch_rows = []
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
        epoch_rows.append((epoch, mean_loss, score))
        print(
            f"epoch {epoch}: mean loss {mean_loss:.4f}, validation macro-F1 {score:.4f}",
            flush=True,
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            publish_best_adapter(model, tokenizer, best_epoch, best_score)
            print(f"epoch {epoch} adapter uploaded", flush=True)
        elapsed_seconds = time.perf_counter() - started_at
        peak_gpu_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        lines = report_lines(
            device_name,
            trainable_count,
            total_count,
            epoch_rows,
            best_epoch,
            best_score,
            elapsed_seconds,
            peak_gpu_mb,
        )
        os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
        write_lines(REPORT_PATH, lines)
        publish_score_log(lines)
        print(f"epoch {epoch} score record uploaded", flush=True)

    elapsed_seconds = time.perf_counter() - started_at
    peak_gpu_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    lines = report_lines(
        device_name,
        trainable_count,
        total_count,
        epoch_rows,
        best_epoch,
        best_score,
        elapsed_seconds,
        peak_gpu_mb,
    )
    write_lines(REPORT_PATH, lines)
    publish_score_log(lines)
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
