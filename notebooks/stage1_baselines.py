"""Score stage 1 baselines on validation macro-F1. The test split is not read."""

import time

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

SEED = 42
TRAIN_PATH = "data/processed/stage1_train.csv"
VALIDATION_PATH = "data/processed/stage1_validation.csv"
REPORT_PATH = "data/processed/stage1_baselines.txt"
REGISTRY_PATH = "experiments/registry.csv"

REGISTRY_HEADER = (
    "run_id,track,model,tokenizer,data_version,strategy,seed,"
    "trainable_params,total_params,learning_rate,batch_size,epochs_or_steps,"
    "val_macro_f1,test_macro_f1,runtime,peak_memory,checkpoint,notes"
)


def macro_f1(true_labels, predicted_labels):
    """Return macro-F1 for the validation labels."""
    score = f1_score(
        true_labels,
        predicted_labels,
        average="macro",
    )
    return float(score)


def majority_label(train_labels):
    """Return the label that appears most often in the training rows."""
    label_counts = train_labels.value_counts()
    most_common_label = label_counts.index[0]
    return most_common_label


def write_lines(path, lines):
    """Write a text report."""
    with open(path, "w", encoding="utf-8") as report_file:
        report_file.write("\n".join(lines))
        report_file.write("\n")


def append_registry_row(row):
    """Append one baseline run. test_macro_f1 stays empty."""
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


def main():
    train_rows = pd.read_csv(TRAIN_PATH)
    validation_rows = pd.read_csv(VALIDATION_PATH)
    train_text = train_rows["normalized_headline"]
    validation_text = validation_rows["normalized_headline"]
    train_labels = train_rows["label"]
    validation_labels = validation_rows["label"]

    common_label = majority_label(train_labels)
    majority_predictions = [common_label] * len(validation_labels)
    majority_score = macro_f1(validation_labels, majority_predictions)

    started_at = time.perf_counter()
    vectorizer = TfidfVectorizer()
    train_vectors = vectorizer.fit_transform(train_text)
    validation_vectors = vectorizer.transform(validation_text)
    classifier = LogisticRegression(
        random_state=SEED,
        max_iter=1000,
    )
    classifier.fit(train_vectors, train_labels)
    tfidf_predictions = classifier.predict(validation_vectors)
    tfidf_seconds = time.perf_counter() - started_at
    tfidf_score = macro_f1(validation_labels, tfidf_predictions)

    report_lines = [
        "Stage 1 baselines. Scores are validation macro-F1. The test split was not read.",
        f"Training majority label: {common_label}",
        f"Majority baseline validation macro-F1: {majority_score:.4f}",
        f"TF-IDF plus logistic regression validation macro-F1: {tfidf_score:.4f}",
        f"TF-IDF fit and score seconds: {tfidf_seconds:.1f}",
    ]
    write_lines(REPORT_PATH, report_lines)
    print("\n".join(report_lines))

    append_registry_row(
        "stage1-majority,B,majority,none,kaggle-urdu-news-1m-v3,majority,"
        f"{SEED},0,0,,,,{majority_score:.4f},,,,"
        "predict the training majority label"
    )
    append_registry_row(
        "stage1-tfidf-logistic,B,logistic-regression,tfidf-word,kaggle-urdu-news-1m-v3,tfidf-linear,"
        f"{SEED},,,,{tfidf_seconds:.1f}s,{tfidf_score:.4f},,,,"
        "TfidfVectorizer plus LogisticRegression on normalized headlines"
    )


if __name__ == "__main__":
    main()
