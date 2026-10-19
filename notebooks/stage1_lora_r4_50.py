"""Train LoRA rank 4 through epoch 100.

The settings match the rank-4 run that reached validation macro-F1 0.9377
at epoch 30. Epochs 1 to 50 are already scored. The best of those is
epoch 50 at 0.9394. This launch continues that saved adapter.
The encoder stays frozen. LoRA updates query and value. The classifier trains.
Learning rate 1e-4. Weight decay 0.01. Label smoothing 0.1. Batch size 32.
After every epoch the adapter, optimizer, and shuffle state are uploaded.
The test split is not read.
Optimizer state for this adapter is small, so it is saved. The full base
model is not uploaded.
"""

import json
import os
import shutil
import time

import pandas as pd
import torch
from huggingface_hub import HfApi
from huggingface_hub import hf_hub_download
from peft import LoraConfig
from peft import PeftModel
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
CHECKPOINT_DIR = "outputs/stage1-lora-r4-50"
RESUME_DIR = "outputs/stage1-lora-r4-50-resume"
REPORT_PATH = "data/processed/stage1_lora_r4_50.txt"
HUB_REPO = "HammadHARahim/urdu-headline-stage1-lora-r4-50"
LABELS = ["sports", "business", "entertainment", "technology"]
BATCH_SIZE = 32
MAX_LENGTH = 64
MAX_EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
LORA_RANK = 4
LORA_ALPHA = 4
LORA_DROPOUT = 0.1
LABEL_SMOOTHING = 0.1
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
    """Update the LoRA weights and classifier with smoothed cross-entropy."""
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
    loss_function = torch.nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
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


def build_fresh_model():
    """Load xlm-roberta-base and attach a new rank-4 adapter."""
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
    return model


def classification_model():
    """Load the base classifier before an adapter is attached."""
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label={index: name for index, name in enumerate(LABELS)},
        label2id={name: index for index, name in enumerate(LABELS)},
    )
    return model


def report_lines(device_name, trainable_count, total_count, epoch_rows, best_epoch, best_score, elapsed_seconds, peak_gpu_mb):
    """Build the text record for this run."""
    shown_score = best_score
    if shown_score < 0.0:
        shown_score = 0.0
    lines = [
        "Stage 1 LoRA rank 4 through epoch 100. The test split was not read.",
        "The settings match the run that scored 0.9377 at epoch 30.",
        "Epochs 1 to 50 scored 0.9394 at epoch 50. Later epochs continue that adapter.",
        f"Device: {device_name}",
        f"Runtime: {os.environ.get('TRAINING_RUNTIME', 'not set')}",
        "Encoder: frozen. LoRA updates query and value. The classifier is trained.",
        f"LoRA rank: {LORA_RANK}",
        f"LoRA alpha: {LORA_ALPHA}",
        f"LoRA dropout: {LORA_DROPOUT}",
        f"Label smoothing: {LABEL_SMOOTHING}",
        f"Trainable parameters: {trainable_count}",
        f"Total parameters: {total_count}",
        f"Learning rate: {LEARNING_RATE}",
        f"Weight decay: {WEIGHT_DECAY}",
        f"Batch size: {BATCH_SIZE}",
        f"Epoch cap: {MAX_EPOCHS}",
        f"Epochs completed: {len(epoch_rows)}",
        f"Best epoch: {best_epoch}",
        f"Validation macro-F1: {shown_score:.4f}",
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
    """Upload the best adapter to the repo root. Later epochs do not replace it."""
    model.push_to_hub(
        HUB_REPO,
        private=True,
        commit_message=f"Best epoch {best_epoch} validation macro-F1 {best_score:.4f}",
    )
    tokenizer.push_to_hub(HUB_REPO, private=True)


def save_resume(model, optimizer, generator, completed_epochs, best_epoch, best_score, epoch_rows):
    """Save the latest epoch, including optimizer state, in one commit."""
    if os.path.isdir(RESUME_DIR):
        shutil.rmtree(RESUME_DIR)
    os.makedirs(RESUME_DIR, exist_ok=True)
    model.save_pretrained(RESUME_DIR)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "generator": generator.get_state(),
        },
        os.path.join(RESUME_DIR, "optimizer.pt"),
    )
    progress = {
        "completed_epochs": completed_epochs,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "epoch_rows": epoch_rows,
    }
    progress_path = os.path.join(RESUME_DIR, "progress.json")
    with open(progress_path, "w", encoding="utf-8") as progress_file:
        json.dump(progress, progress_file)
    HfApi().upload_folder(
        folder_path=RESUME_DIR,
        path_in_repo="resume",
        repo_id=HUB_REPO,
        commit_message=f"Save resume after epoch {completed_epochs}",
    )


def squash_history():
    """Keep one copy of the current files. Old copies fill private storage."""
    try:
        HfApi().super_squash_history(
            repo_id=HUB_REPO,
            commit_message="Keep the latest resume files and the best adapter.",
        )
    except Exception as error:
        print(f"History squash failed: {error}", flush=True)


def adapter_weight_filename(filenames):
    """Return the saved adapter weight name, or the safetensors name when none exists."""
    if "resume/adapter_model.safetensors" in filenames:
        return "resume/adapter_model.safetensors"
    if "resume/adapter_model.bin" in filenames:
        return "resume/adapter_model.bin"
    return "resume/adapter_model.safetensors"


def download_resume():
    """Copy a complete resume. Stop when a previous epoch was saved only in part."""
    filenames = HfApi().list_repo_files(repo_id=HUB_REPO)
    required = [
        adapter_weight_filename(filenames),
        "resume/adapter_config.json",
        "resume/optimizer.pt",
        "resume/progress.json",
    ]
    missing = []
    for filename in required:
        if filename not in filenames:
            missing.append(filename)
    training_has_started = "epoch_scores.txt" in filenames or "adapter_model.safetensors" in filenames
    for filename in filenames:
        if filename.startswith("resume/"):
            training_has_started = True
    if len(missing) == 0:
        if os.path.isdir(RESUME_DIR):
            shutil.rmtree(RESUME_DIR)
        os.makedirs(RESUME_DIR, exist_ok=True)
        for filename in required:
            downloaded = hf_hub_download(repo_id=HUB_REPO, filename=filename)
            target_name = filename.split("/", 1)[1]
            shutil.copyfile(downloaded, os.path.join(RESUME_DIR, target_name))
        with open(os.path.join(RESUME_DIR, "progress.json"), encoding="utf-8") as progress_file:
            progress = json.load(progress_file)
        return progress
    if training_has_started:
        missing_text = ", ".join(missing)
        raise SystemExit(
            "Resume files are incomplete. Training was not started again. Missing: "
            + missing_text
        )
    return None


def trainable_parameters_of(model):
    """Return the parameters Adam is allowed to update."""
    chosen = []
    for parameter in model.parameters():
        if parameter.requires_grad:
            chosen.append(parameter)
    return chosen


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
    progress = download_resume()
    generator = torch.Generator()
    best_score = -1.0
    best_epoch = 0
    epoch_rows = []
    completed_epochs = 0
    if progress is None:
        model = build_fresh_model()
        generator.manual_seed(SEED)
        print("Loaded a new rank-4 adapter on xlm-roberta-base.", flush=True)
        saved_optimizer = None
    else:
        model = PeftModel.from_pretrained(
            classification_model(),
            RESUME_DIR,
            is_trainable=True,
        )
        completed_epochs = progress["completed_epochs"]
        best_epoch = progress["best_epoch"]
        best_score = progress["best_score"]
        epoch_rows = progress["epoch_rows"]
        saved_optimizer = torch.load(
            os.path.join(RESUME_DIR, "optimizer.pt"),
            map_location="cpu",
            weights_only=False,
        )
        print(
            f"Resumed from epoch {completed_epochs}. "
            f"Best epoch so far is {best_epoch}.",
            flush=True,
        )
    model.to(device)
    optimizer = AdamW(
        trainable_parameters_of(model),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    if saved_optimizer is not None:
        optimizer.load_state_dict(saved_optimizer["optimizer"])
        generator.set_state(saved_optimizer["generator"])
    trainable_count, total_count = count_parameters(model)
    print(
        f"Trainable parameters: {trainable_count}. Total parameters: {total_count}.",
        flush=True,
    )
    print(
        f"Learning rate: {LEARNING_RATE}. Label smoothing: {LABEL_SMOOTHING}.",
        flush=True,
    )
    print(
        f"Next epoch: {completed_epochs + 1} of {MAX_EPOCHS}.",
        flush=True,
    )

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
        epoch_rows.append([epoch, mean_loss, score])
        print(
            f"epoch {epoch}: mean loss {mean_loss:.4f}, validation macro-F1 {score:.4f}",
            flush=True,
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            publish_best_adapter(model, tokenizer, best_epoch, best_score)
            print(f"epoch {epoch} best adapter uploaded", flush=True)
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
            optimizer,
            generator,
            epoch,
            best_epoch,
            best_score,
            epoch_rows,
        )
        squash_history()
        print(f"epoch {epoch} resume uploaded", flush=True)

    if len(epoch_rows) == 0:
        print("No epoch finished, so no score was uploaded.", flush=True)
        return
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
