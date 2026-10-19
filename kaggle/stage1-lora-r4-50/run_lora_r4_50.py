"""Start a fresh 50-epoch LoRA rank-4 run on a private Kaggle GPU notebook.

The notebook uses one GPU. Kaggle's T4 shape attaches two T4 chips, and this
script leaves the second chip idle so the batch size stays 32.
"""

import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path


def find_bundle_dir():
    """Return the folder that holds the script, headlines, and token file."""
    roots = [
        Path(__file__).resolve().parent,
        Path("/kaggle/working"),
        Path("/kaggle/input"),
    ]
    searched_folders = []
    for root in roots:
        if not root.is_dir():
            print(f"Not a directory: {root}", flush=True)
            continue
        folders = [root]
        for path in root.rglob("*"):
            if path.is_dir():
                folders.append(path)
        for folder in folders:
            searched_folders.append(folder)
            token_path = folder / "hf_token"
            train_path = folder / "stage1_train.csv"
            if token_path.is_file() and train_path.is_file():
                print(f"Found training files in {folder}", flush=True)
                return folder
    print("Searched these folders:", flush=True)
    for folder in searched_folders:
        print(folder, flush=True)
    input_root = Path("/kaggle/input")
    if input_root.is_dir():
        print("Files under /kaggle/input:", flush=True)
        for path in input_root.rglob("*"):
            if path.is_file():
                print(path, flush=True)
    raise SystemExit("The Kaggle bundle is missing the headlines or token file.")


def install_token(bundle_dir):
    """Put the Hub token where huggingface_hub expects it."""
    token_dir = Path.home() / ".cache" / "huggingface"
    token_dir.mkdir(parents=True, exist_ok=True)
    token_path = token_dir / "token"
    shutil.copyfile(bundle_dir / "hf_token", token_path)
    os.chmod(token_path, 0o600)


def install_headlines(bundle_dir, work_dir):
    """Copy the slim headline files to the paths the training script reads."""
    processed_dir = work_dir / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(bundle_dir / "stage1_train.csv", processed_dir / "stage1_train.csv")
    shutil.copyfile(
        bundle_dir / "stage1_validation.csv",
        processed_dir / "stage1_validation.csv",
    )


def install_peft():
    """Install peft. Kaggle's torchao 0.10 is too old for the peft already on the image."""
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "peft"])


def main():
    """Prepare files, then run the 50-epoch LoRA training script."""
    bundle_dir = find_bundle_dir()
    work_dir = Path("/kaggle/working")
    if not work_dir.is_dir():
        work_dir = bundle_dir
    install_token(bundle_dir)
    install_headlines(bundle_dir, work_dir)
    # Kaggle attaches two T4 chips. Keep one visible so each update stays 32 headlines.
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["TRAINING_RUNTIME"] = "Kaggle"
    os.chdir(work_dir)
    install_peft()
    training_script = bundle_dir / "stage1_lora_r4_50.py"
    print("Continuing LoRA rank 4 from the saved resume through epoch 100.", flush=True)
    runpy.run_path(str(training_script), run_name="__main__")


if __name__ == "__main__":
    main()
