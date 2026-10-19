"""Continue from the epoch-4 checkpoint with stronger weight decay.

Epoch 4 is the best 5e-6 checkpoint, validation macro-F1 0.9449.
That run used AdamW's default weight decay of 0.01.
This run keeps learning rate 5e-6 and changes weight decay to 0.1.
The saved weights are the epoch with the highest validation macro-F1.
The test split is not read.
After every new epoch the script uploads a resume checkpoint.
"""

import json
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
CHECKPOINT_DIR = "outputs/stage1-full-t4-wd0p1"
RESUME_DIR = "outputs/stage1-full-t4-wd0p1-resume"
REPORT_PATH = "data/processed/stage1_full_finetune_t4_wd0p1.txt"
HUB_REPO = "HammadHARahim/urdu-headline-stage1-wd0p1"
SOURCE_REPO = "HammadHARahim/urdu-headline-stage1-lr5e6"
LABELS = ["sports", "business", "entertainment", "technology"]
BATCH_SIZE = 32
MAX_LENGTH = 64
MAX_EPOCHS = 20
LEARNING_RATE = 5e-6
# The one change from the 5e-6 run. That run used AdamW's default of 0.01.
WEIGHT_DECAY = 0.1
# Measured on the 5e-6 run. A new epoch must beat this score to replace those weights.
EPOCH4_SCORE = 0.9449
EPOCH4_ROWS = [
    [1, 0.2700, 0.9293],
    [2, 0.1675, 0.9299],
    [3, 0.1343, 0.9416],
    [4, 0.0721, 0.9449],
]
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
    """Update the full model on one batch."""
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


def copy_state_to_cpu(model):
    """Keep the best weights in memory after the epoch that scored highest."""
    copied_state = {}
    for name, value in model.state_dict().items():
        copied_state[name] = value.detach().cpu().clone()
    return copied_state


def report_lines(device_name, trainable_count, total_count, epoch_rows, best_epoch, best_score, elapsed_seconds, peak_gpu_mb):
    """Build the text record for this run."""
    lines = [
        "Stage 1 full fine-tune on a T4. Epoch 5 onward uses weight decay 0.1.",
        "The test split was not read.",
        "Epochs 1 to 3 were trained at learning rate 2e-5 and weight decay 0.01.",
        "Epoch 4 was trained at learning rate 5e-6 and weight decay 0.01.",
        f"Device: {device_name}",
        f"Runtime: {os.environ.get('TRAINING_RUNTIME', 'not set')}",
        "Encoder: trained. The full model was updated.",
        f"Trainable parameters: {trainable_count}",
        f"Total parameters: {total_count}",
        f"Learning rate from epoch 5: {LEARNING_RATE}",
        f"Weight decay from epoch 5: {WEIGHT_DECAY}",
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


def publish_best_checkpoint(model, tokenizer, best_epoch, best_score):
    """Upload the best weights before the next epoch starts."""
    model.push_to_hub(
        HUB_REPO,
        private=True,
        commit_message=f"Best epoch {best_epoch} validation macro-F1 {best_score:.4f}",
    )
    tokenizer.push_to_hub(HUB_REPO, private=True)


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
        progress["has_latest_weights"] = "resume/latest/model.safetensors" in filenames
        progress["has_training_state"] = "resume/training.pt" in filenames
        return progress
    if "epoch_scores.txt" in filenames and "model.safetensors" in filenames:
        score_path = hf_hub_download(repo_id=HUB_REPO, filename="epoch_scores.txt")
        with open(score_path, encoding="utf-8") as score_file:
            progress = progress_from_score_log(score_file.read())
        progress["has_latest_weights"] = False
        progress["has_training_state"] = False
        return progress
    return None


def load_saved_weights(model, progress):
    """Load the last finished epoch into the model."""
    if progress["has_latest_weights"]:
        saved_model = AutoModelForSequenceClassification.from_pretrained(
            HUB_REPO,
            subfolder="resume/latest",
        )
    else:
        saved_model = AutoModelForSequenceClassification.from_pretrained(HUB_REPO)
    model.load_state_dict(saved_model.state_dict())
    del saved_model


def load_training_state(optimizer, generator, device):
    """Load Adam and the shuffle generator when the resume file has them."""
    training_path = hf_hub_download(repo_id=HUB_REPO, filename="resume/training.pt")
    training_state = torch.load(
        training_path,
        map_location=device,
        weights_only=False,
    )
    optimizer.load_state_dict(training_state["optimizer"])
    generator_state = training_state["generator"]
    generator.set_state(generator_state.cpu())


def save_resume(model, tokenizer, optimizer, generator, completed_epochs, best_epoch, best_score, epoch_rows):
    """Upload the last finished epoch so a new T4 can continue."""
    latest_dir = os.path.join(RESUME_DIR, "latest")
    os.makedirs(latest_dir, exist_ok=True)
    model.save_pretrained(latest_dir)
    tokenizer.save_pretrained(latest_dir)
    progress = {
        "completed_epochs": completed_epochs,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "epoch_rows": epoch_rows,
    }
    progress_path = os.path.join(RESUME_DIR, "progress.json")
    with open(progress_path, "w", encoding="utf-8") as progress_file:
        json.dump(progress, progress_file)
    training_path = os.path.join(RESUME_DIR, "training.pt")
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "generator": generator.get_state(),
        },
        training_path,
    )
    HfApi().upload_folder(
        folder_path=RESUME_DIR,
        path_in_repo="resume",
        repo_id=HUB_REPO,
        commit_message=f"Resume checkpoint after epoch {completed_epochs}",
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
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    generator = torch.Generator()
    generator.manual_seed(SEED)
    progress = find_saved_progress()
    best_score = -1.0
    best_epoch = 0
    best_state = None
    epoch_rows = []
    completed_epochs = 0
    if progress is not None:
        load_saved_weights(model, progress)
        completed_epochs = progress["completed_epochs"]
        best_epoch = progress["best_epoch"]
        best_score = progress["best_score"]
        epoch_rows = progress["epoch_rows"]
        if progress["has_training_state"]:
            load_training_state(optimizer, generator, device)
            print(
                f"Resumed from epoch {completed_epochs} with the saved Adam state.",
                flush=True,
            )
        else:
            print(
                f"Resumed weights from epoch {completed_epochs}. Adam starts fresh.",
                flush=True,
            )
        if best_epoch == completed_epochs:
            best_state = copy_state_to_cpu(model)
    else:
        saved_model = AutoModelForSequenceClassification.from_pretrained(SOURCE_REPO)
        model.load_state_dict(saved_model.state_dict())
        del saved_model
        completed_epochs = 4
        best_epoch = 4
        best_score = EPOCH4_SCORE
        epoch_rows = []
        for row in EPOCH4_ROWS:
            epoch_rows.append(row)
        best_state = copy_state_to_cpu(model)
        print(
            "Loaded epoch 4 weights. Weight decay is 0.1 from epoch 5 onward.",
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
            best_state = copy_state_to_cpu(model)
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
        )
        os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
        write_lines(REPORT_PATH, lines)
        publish_score_log(lines)
        save_resume(
            model,
            tokenizer,
            optimizer,
            generator,
            epoch,
            best_epoch,
            best_score,
            epoch_rows,
        )
        print(f"epoch {epoch} resume checkpoint uploaded", flush=True)

    if best_state is None:
        print("The Hub already has the best weights.", flush=True)
        return
    model.load_state_dict(best_state)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    model.save_pretrained(CHECKPOINT_DIR)
    tokenizer.save_pretrained(CHECKPOINT_DIR)
    model.push_to_hub(
        HUB_REPO,
        private=True,
        commit_message=f"Best epoch {best_epoch} validation macro-F1 {best_score:.4f}",
    )
    tokenizer.push_to_hub(HUB_REPO, private=True)
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
