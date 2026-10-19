"""Smoke test for stage 1. One training batch must overfit, then save and resume.

The validation and test splits are not read.
On a GPU the full model is updated. On CPU the encoder stays frozen, because
this computer has 7.4 GB of memory.
"""

import os
import resource
import time

import pandas as pd
import torch
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

SEED = 42
MODEL_NAME = "xlm-roberta-base"
TRAIN_PATH = "data/processed/stage1_train.csv"
CHECKPOINT_PATH = "outputs/stage1-smoke/checkpoint.pt"
REPORT_PATH = "data/processed/stage1_smoke_test.txt"
REGISTRY_PATH = "experiments/registry.csv"
LABELS = ["sports", "business", "entertainment", "technology"]
BATCH_SIZE = 4
MAX_STEPS = 40
FULL_MODEL_LEARNING_RATE = 2e-4
HEAD_LEARNING_RATE = 1e-2
REGISTRY_HEADER = (
    "run_id,track,model,tokenizer,data_version,strategy,seed,trainable_params,"
    "total_params,learning_rate,batch_size,epochs_or_steps,val_macro_f1,"
    "test_macro_f1,runtime,peak_memory,checkpoint,notes"
)


def write_lines(path, lines):
    """Write a text report."""
    with open(path, "w", encoding="utf-8") as report_file:
        report_file.write("\n".join(lines))
        report_file.write("\n")


def append_registry_row(row):
    """Append one smoke-test run. val_macro_f1 and test_macro_f1 stay empty."""
    registry_file_exists = False
    try:
        with open(REGISTRY_PATH, encoding="utf-8") as registry_file:
            registry_file_exists = True
    except FileNotFoundError:
        registry_file_exists = False

    with open(REGISTRY_PATH, "a", encoding="utf-8") as registry_file:
        if not registry_file_exists:
            registry_file.write(REGISTRY_HEADER + "\n")
        registry_file.write(row + "\n")


def freeze_encoder(model):
    """Keep the pretrained encoder fixed so this check fits in memory."""
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


def make_optimizer(model, learning_rate):
    """Update only parameters that are allowed to change."""
    trainable_parameters = []
    for parameter in model.parameters():
        if parameter.requires_grad:
            trainable_parameters.append(parameter)
    optimizer = AdamW(trainable_parameters, lr=learning_rate)
    return optimizer


def choose_device():
    """Use the GPU when Colab provides one."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        return device, device_name
    device = torch.device("cpu")
    return device, "cpu"


def batch_accuracy(logits, label_ids):
    """Return how many headlines in this batch got the right label."""
    predicted_ids = torch.argmax(logits, dim=1)
    correct_count = int((predicted_ids == label_ids).sum().item())
    return correct_count / len(label_ids)


def training_step(model, optimizer, token_ids, attention_mask, label_ids):
    """Run one update on the same batch."""
    model.train()
    model.roberta.eval()
    optimizer.zero_grad()
    outputs = model(
        input_ids=token_ids,
        attention_mask=attention_mask,
        labels=label_ids,
    )
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    accuracy = batch_accuracy(outputs.logits.detach(), label_ids)
    return float(loss.detach().item()), accuracy


def main():
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    train_rows = pd.read_csv(
        TRAIN_PATH,
        usecols=["normalized_headline", "label"],
    )
    batch_rows = train_rows.sample(n=BATCH_SIZE, random_state=SEED)
    headlines = batch_rows["normalized_headline"].tolist()
    label_names = batch_rows["label"].tolist()
    label_ids = torch.tensor([LABELS.index(name) for name in label_names])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    encoded = tokenizer(
        headlines,
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt",
    )
    token_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    device, device_name = choose_device()
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label={index: name for index, name in enumerate(LABELS)},
        label2id={name: index for index, name in enumerate(LABELS)},
    )
    model.to(device)
    token_ids = token_ids.to(device)
    attention_mask = attention_mask.to(device)
    label_ids = label_ids.to(device)
    if device.type == "cuda":
        learning_rate = FULL_MODEL_LEARNING_RATE
        encoder_note = "trained. The full model was updated."
        run_id = "stage1-smoke-t4"
        strategy = "full-model-smoke"
        notes = f"{device_name}; batch accuracy is not validation macro-F1"
    else:
        # A full AdamW update does not fit in 7.4 GB of CPU memory.
        freeze_encoder(model)
        learning_rate = HEAD_LEARNING_RATE
        encoder_note = "frozen. Only the new classifier was updated."
        run_id = "stage1-smoke-cpu"
        strategy = "frozen-encoder-smoke"
        notes = "CPU only; encoder frozen; batch accuracy is not validation macro-F1"
    trainable_count, total_count = count_parameters(model)
    optimizer = make_optimizer(model, learning_rate)
    print(
        f"Trainable parameters: {trainable_count}. "
        f"Total parameters: {total_count}.",
        flush=True,
    )

    started_at = time.perf_counter()
    last_loss = None
    last_accuracy = 0.0
    finished_step = 0
    for step in range(1, MAX_STEPS + 1):
        last_loss, last_accuracy = training_step(
            model,
            optimizer,
            token_ids,
            attention_mask,
            label_ids,
        )
        finished_step = step
        print(
            f"step {step}: loss {last_loss:.4f}, batch accuracy {last_accuracy:.2f}",
            flush=True,
        )
        if last_accuracy == 1.0 and last_loss < 0.1:
            break

    overfit_passed = last_accuracy == 1.0 and last_loss < 0.1
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    torch.save(
        {
            "step": finished_step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        CHECKPOINT_PATH,
    )
    del model
    del optimizer

    resumed_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
    )
    resumed_model.to(device)
    if device.type != "cuda":
        freeze_encoder(resumed_model)
    resumed_optimizer = make_optimizer(resumed_model, learning_rate)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    resumed_model.load_state_dict(checkpoint["model_state"])
    resumed_optimizer.load_state_dict(checkpoint["optimizer_state"])
    resumed_loss, resumed_accuracy = training_step(
        resumed_model,
        resumed_optimizer,
        token_ids,
        attention_mask,
        label_ids,
    )
    resume_passed = resumed_accuracy == 1.0
    elapsed_seconds = time.perf_counter() - started_at
    peak_memory_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    if device.type == "cuda":
        peak_gpu_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        peak_memory_text = f"{peak_gpu_mb:.1f}MB GPU"
    else:
        peak_memory_text = f"{peak_memory_mb:.1f}MB"

    report_lines = [
        "Stage 1 smoke test on one training batch. Validation and test were not read.",
        f"Device: {device_name}",
        f"Encoder: {encoder_note}",
        f"Trainable parameters: {trainable_count}",
        f"Total parameters: {total_count}",
        f"Learning rate: {learning_rate}",
        f"Batch size: {BATCH_SIZE}",
        f"Steps before save: {finished_step}",
        f"Loss before save: {last_loss:.4f}",
        f"Batch accuracy before save: {last_accuracy:.2f}",
        f"One batch overfit: {overfit_passed}",
        f"Loss after resume step: {resumed_loss:.4f}",
        f"Batch accuracy after resume: {resumed_accuracy:.2f}",
        f"Save and resume: {resume_passed}",
        f"Seconds: {elapsed_seconds:.1f}",
        f"Peak memory: {peak_memory_text}",
        f"Checkpoint: {CHECKPOINT_PATH}",
    ]
    write_lines(REPORT_PATH, report_lines)
    print("\n".join(report_lines), flush=True)
    append_registry_row(
        f"{run_id},B,xlm-roberta-base,xlm-roberta-base,kaggle-urdu-news-1m-v3,"
        f"{strategy},"
        f"{SEED},{trainable_count},{total_count},{learning_rate},{BATCH_SIZE},"
        f"{finished_step},,,{elapsed_seconds:.1f}s,{peak_memory_text},"
        f"{CHECKPOINT_PATH},"
        f"{notes}"
    )
    if not overfit_passed or not resume_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
