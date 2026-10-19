"""Fine-tune xlm-roberta-base with a warmup and decay learning-rate schedule.

The constant learning-rate run at 2e-5 peaked at validation macro-F1 0.9416.
This run uses the same checkpoint, batch size, weight decay, and loss.
The one change is the schedule: linear warmup for 10% of the planned steps,
then linear decay to zero. That is the Hugging Face linear schedule.
The kept overall best is still epoch 4 at 0.9449 on the 5e-6 repo.
The test split is not read.
Every epoch uploads the score log only.
The weight file is uploaded only when this run sets a new best score.
Optimizer state is not uploaded.
"""

import json
import math
import os
import time

import pandas as pd
import torch
from huggingface_hub import HfApi
from huggingface_hub import hf_hub_download
from sklearn.metrics import f1_score
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

SEED = 42
MODEL_NAME = "xlm-roberta-base"
TRAIN_PATH = "data/processed/stage1_train.csv"
VALIDATION_PATH = "data/processed/stage1_validation.csv"
CHECKPOINT_DIR = "outputs/stage1-full-t4-warmup"
REPORT_PATH = "data/processed/stage1_full_finetune_t4_warmup.txt"
HUB_REPO = "HammadHARahim/urdu-headline-stage1-warmup"
LABELS = ["sports", "business", "entertainment", "technology"]
BATCH_SIZE = 32
MAX_LENGTH = 64
MAX_EPOCHS = 20
PEAK_LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
TRAINING_BUDGET_SECONDS = 230 * 60


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


def batches_in_one_epoch(headline_count):
    """Count batches, including a shorter final batch."""
    batch_count = headline_count // BATCH_SIZE
    leftover = headline_count % BATCH_SIZE
    if leftover > 0:
        batch_count = batch_count + 1
    return batch_count


def learning_rate_for_step(step_index, total_steps, warmup_steps):
    """Return the linear warmup-then-decay rate. step_index starts at 0."""
    if step_index < warmup_steps:
        scale = float(step_index) / float(max(1, warmup_steps))
        return PEAK_LEARNING_RATE * scale
    steps_after_warmup = max(1, total_steps - warmup_steps)
    steps_remaining = total_steps - step_index
    scale = float(steps_remaining) / float(steps_after_warmup)
    if scale < 0.0:
        scale = 0.0
    return PEAK_LEARNING_RATE * scale


def set_learning_rate(optimizer, learning_rate):
    """Apply one learning rate to every parameter group."""
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


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
    """Update the full model on one batch with unweighted cross-entropy."""
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
    )
    loss_function = torch.nn.CrossEntropyLoss()
    loss = loss_function(outputs.logits, batch_labels)
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


def report_lines(device_name, trainable_count, total_count, epoch_rows, best_epoch, best_score, elapsed_seconds, peak_gpu_mb, total_steps, warmup_steps):
    """Build the text record for this run."""
    lines = [
        "Stage 1 full fine-tune on a T4 from xlm-roberta-base.",
        "The test split was not read.",
        "The one change from the constant 2e-5 run is the learning-rate schedule.",
        "Schedule: linear warmup for 10% of planned steps, then linear decay to zero.",
        "The constant 2e-5 run peaked at validation macro-F1 0.9416.",
        "The kept overall best is epoch 4 at 0.9449 on HammadHARahim/urdu-headline-stage1-lr5e6.",
        f"Device: {device_name}",
        f"Runtime: {os.environ.get('TRAINING_RUNTIME', 'not set')}",
        "Encoder: trained. The full model was updated.",
        f"Trainable parameters: {trainable_count}",
        f"Total parameters: {total_count}",
        f"Peak learning rate: {PEAK_LEARNING_RATE}",
        f"Weight decay: {WEIGHT_DECAY}",
        "Loss: unweighted cross-entropy",
        f"Batch size: {BATCH_SIZE}",
        f"Planned epochs: {MAX_EPOCHS}",
        f"Planned steps: {total_steps}",
        f"Warmup steps: {warmup_steps}",
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


def publish_best_checkpoint(model, tokenizer, best_epoch, best_score):
    """Upload the best weights, then drop older copies of those files."""
    model.push_to_hub(
        HUB_REPO,
        private=True,
        commit_message=f"Best epoch {best_epoch} validation macro-F1 {best_score:.4f}",
    )
    tokenizer.push_to_hub(HUB_REPO, private=True)
    HfApi().super_squash_history(
        repo_id=HUB_REPO,
        commit_message="Keep the latest score log and best weights.",
    )


def progress_from_score_log(score_text):
    """Read completed epochs from the score log when no resume file exists yet."""
    completed_epochs = 0
    best_epoch = 0
    best_score = -1.0
    epoch_rows = []
    for line in score_text.splitlines():
        if line.startswith("Epochs completed:"):
            completed_epochs = int(line.split(": ", 1)[1])
        if line.startswith("Best epoch:"):
            best_epoch = int(line.split(": ", 1)[1])
        if line.startswith("Validation macro-F1:"):
            best_score = float(line.split(": ", 1)[1])
        if line.startswith("Epoch ") and "mean loss" in line:
            epoch_text, score_part = line.split(", validation macro-F1 ")
            epoch_label, loss_text = epoch_text.split(": mean loss ")
            epoch_number = int(epoch_label.replace("Epoch ", ""))
            mean_loss = float(loss_text)
            score = float(score_part)
            epoch_rows.append([epoch_number, mean_loss, score])
    progress = {
        "completed_epochs": completed_epochs,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "epoch_rows": epoch_rows,
    }
    return progress


def find_saved_progress():
    """Return saved epoch progress, or None when training has not started."""
    filenames = HfApi().list_repo_files(repo_id=HUB_REPO)
    if "resume/progress.json" in filenames:
        progress_path = hf_hub_download(
            repo_id=HUB_REPO,
            filename="resume/progress.json",
        )
        with open(progress_path, encoding="utf-8") as progress_file:
            progress = json.load(progress_file)
        progress["has_best_weights"] = "model.safetensors" in filenames
        return progress
    if "epoch_scores.txt" in filenames:
        score_path = hf_hub_download(repo_id=HUB_REPO, filename="epoch_scores.txt")
        with open(score_path, encoding="utf-8") as score_file:
            progress = progress_from_score_log(score_file.read())
        progress["has_best_weights"] = "model.safetensors" in filenames
        return progress
    return None


def load_saved_weights(model):
    """Load this run's best weights from its own Hub repo."""
    saved_model = AutoModelForSequenceClassification.from_pretrained(HUB_REPO)
    model.load_state_dict(saved_model.state_dict())
    del saved_model


def save_progress(completed_epochs, best_epoch, best_score, epoch_rows):
    """Upload the small epoch record. This does not upload weights."""
    progress = {
        "completed_epochs": completed_epochs,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "epoch_rows": epoch_rows,
    }
    HfApi().upload_file(
        path_or_fileobj=json.dumps(progress).encode("utf-8"),
        path_in_repo="resume/progress.json",
        repo_id=HUB_REPO,
        commit_message=f"Save progress after epoch {completed_epochs}",
    )


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

    batches_per_epoch = batches_in_one_epoch(len(train_headlines))
    total_steps = MAX_EPOCHS * batches_per_epoch
    warmup_steps = math.ceil(total_steps * WARMUP_RATIO)
    print(
        f"Planned steps: {total_steps}. Warmup steps: {warmup_steps}.",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label={index: name for index, name in enumerate(LABELS)},
        label2id={name: index for index, name in enumerate(LABELS)},
    )
    model.to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=PEAK_LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    generator = torch.Generator()
    generator.manual_seed(SEED)
    progress = find_saved_progress()
    best_score = -1.0
    best_epoch = 0
    epoch_rows = []
    completed_epochs = 0
    if progress is not None and progress["has_best_weights"]:
        load_saved_weights(model)
        completed_epochs = progress["completed_epochs"]
        best_epoch = progress["best_epoch"]
        best_score = progress["best_score"]
        epoch_rows = progress["epoch_rows"]
        print(
            f"Resumed scores through epoch {completed_epochs}. "
            f"Weights are best epoch {best_epoch}. Adam starts fresh.",
            flush=True,
        )
    else:
        print(
            "Loaded xlm-roberta-base. The classifier head starts untrained.",
            flush=True,
        )
    trainable_count, total_count = count_parameters(model)
    print(
        f"Trainable parameters: {trainable_count}. Total parameters: {total_count}.",
        flush=True,
    )
    print(
        f"Next epoch: {completed_epochs + 1} of {MAX_EPOCHS}.",
        flush=True,
    )

    # The budget belongs to this T4 session. Time spent on a closed session does not count.
    started_at = time.perf_counter()
    for epoch in range(completed_epochs + 1, MAX_EPOCHS + 1):
        elapsed = time.perf_counter() - started_at
        if elapsed > TRAINING_BUDGET_SECONDS:
            print("Stopped before another epoch to stay inside the T4 time budget.", flush=True)
            break
        batch_number = 0
        epoch_loss = 0.0
        # A killed epoch is repeated. Its first step uses this epoch's place in the schedule.
        global_step = (epoch - 1) * batches_per_epoch
        for batch_headlines, batch_labels in training_batches(
            train_headlines,
            train_label_ids,
            generator,
        ):
            current_lr = learning_rate_for_step(global_step, total_steps, warmup_steps)
            set_learning_rate(optimizer, current_lr)
            if batch_number == 0:
                print(
                    f"epoch {epoch} learning rate {current_lr:.8f}",
                    flush=True,
                )
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
            global_step = global_step + 1
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
            publish_best_checkpoint(model, tokenizer, best_epoch, best_score)
            print(f"epoch {epoch} weights uploaded", flush=True)
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
            total_steps,
            warmup_steps,
        )
        os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
        write_lines(REPORT_PATH, lines)
        publish_score_log(lines)
        save_progress(
            epoch,
            best_epoch,
            best_score,
            epoch_rows,
        )
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
        total_steps,
        warmup_steps,
    )
    write_lines(REPORT_PATH, lines)
    publish_score_log(lines)
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
