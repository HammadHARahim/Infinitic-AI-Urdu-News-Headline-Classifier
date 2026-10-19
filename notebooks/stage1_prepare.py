"""Audit Urdu News Dataset 1M and write the sealed stage 1 split."""

import hashlib
import io

import pandas as pd

SEED = 42
RAW_PATH = "data/raw/urdu-news-dataset-1M.csv"
AUDIT_PATH = "data/processed/stage1_audit.txt"
LEAKAGE_PATH = "data/processed/stage1_leakage.txt"
TRAIN_PATH = "data/processed/stage1_train.csv"
VALIDATION_PATH = "data/processed/stage1_validation.csv"
TEST_PATH = "data/processed/stage1_test.csv"

LABEL_BY_CATEGORY = {
    "Sports": "sports",
    "Business & Economics": "business",
    "Entertainment": "entertainment",
    "Science & Technology": "technology",
}


def normalize_headline(headline_text):
    """Return the headline with one space between words and no edge spaces."""
    if headline_text is None or pd.isna(headline_text):
        return ""
    words = str(headline_text).split()
    cleaned_text = " ".join(words)
    return cleaned_text.strip()


def map_category(category_name):
    """Return the project label for a source category, or an empty string."""
    if category_name is None or pd.isna(category_name):
        return ""
    cleaned_category = str(category_name).strip()
    if cleaned_category not in LABEL_BY_CATEGORY:
        return ""
    return LABEL_BY_CATEGORY[cleaned_category]


def find_root(parent, key):
    """Return the group root for a headline or URL key."""
    root = key
    while parent[root] != root:
        root = parent[root]
    current_key = key
    while parent[current_key] != root:
        next_key = parent[current_key]
        parent[current_key] = root
        current_key = next_key
    return root


def union_keys(parent, left_key, right_key):
    """Put two keys in the same split group."""
    if left_key not in parent:
        parent[left_key] = left_key
    if right_key not in parent:
        parent[right_key] = right_key
    left_root = find_root(parent, left_key)
    right_root = find_root(parent, right_key)
    if left_root != right_root:
        parent[right_root] = left_root


def split_name_for_group(group_key):
    """Assign one group to train, validation, or test.

    The same group key always gets the same split because the hash includes
    seed 42 and that key.
    """
    payload = f"{SEED}\n{group_key}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    bucket = int(digest[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


def write_lines(path, lines):
    """Write a text report."""
    with open(path, "w", encoding="utf-8") as report_file:
        report_file.write("\n".join(lines))
        report_file.write("\n")


def load_headlines(path):
    """Load the CSV. The published file ends in the middle of the last row."""
    with open(path, "rb") as raw_file:
        file_bytes = raw_file.read()
    file_text = file_bytes[:-1].decode("utf-8-sig")
    last_newline = file_text.rfind("\n")
    file_text = file_text[: last_newline + 1]
    headlines = pd.read_csv(io.StringIO(file_text))
    return headlines


def main():
    headlines = load_headlines(RAW_PATH)
    headlines.columns = [column_name.replace("\ufeff", "") for column_name in headlines.columns]

    headline_column_exists = "Headline" in headlines.columns
    row_count = len(headlines)
    category_counts = headlines["Category"].value_counts(dropna=False)

    headlines["normalized_headline"] = headlines["Headline"].map(normalize_headline)
    headlines["label"] = headlines["Category"].map(map_category)

    empty_headline_count = int((headlines["normalized_headline"] == "").sum())
    unknown_category_count = int((headlines["label"] == "").sum())
    kept_headlines = headlines[
        (headlines["normalized_headline"] != "") & (headlines["label"] != "")
    ].copy()

    label_counts = kept_headlines["label"].value_counts()

    audit_lines = [
        "Stage 1 audit of data/raw/urdu-news-dataset-1M.csv",
        "The published file ends in the middle of the last row. That incomplete row was left out.",
        f"Headline column exists: {headline_column_exists}",
        f"Rows in the file: {row_count}",
        f"Rows with an empty headline: {empty_headline_count}",
        f"Rows with an unmapped category: {unknown_category_count}",
        f"Rows kept for the split: {len(kept_headlines)}",
        "",
        "Source category counts:",
    ]
    for category_name, category_count in category_counts.items():
        audit_lines.append(f"{category_name}: {category_count}")
    audit_lines.append("")
    audit_lines.append("Project label counts for kept rows:")
    for label_name, label_count in label_counts.items():
        audit_lines.append(f"{label_name}: {label_count}")
    write_lines(AUDIT_PATH, audit_lines)
    print("\n".join(audit_lines))

    parent = {}
    for normalized_headline, source_url in zip(
        kept_headlines["normalized_headline"], kept_headlines["URL"]
    ):
        text_key = "text:" + normalized_headline
        if text_key not in parent:
            parent[text_key] = text_key
        url_text = ""
        if source_url is not None and not pd.isna(source_url):
            url_text = str(source_url).strip()
        if url_text != "":
            url_key = "url:" + url_text
            union_keys(parent, text_key, url_key)

    group_key_by_root = {}
    for key in list(parent.keys()):
        if not key.startswith("text:"):
            continue
        root = find_root(parent, key)
        normalized_headline = key[len("text:") :]
        current_group_key = group_key_by_root.get(root)
        if current_group_key is None or normalized_headline < current_group_key:
            group_key_by_root[root] = normalized_headline

    split_names = []
    for normalized_headline in kept_headlines["normalized_headline"]:
        text_key = "text:" + normalized_headline
        root = find_root(parent, text_key)
        group_key = group_key_by_root[root]
        split_names.append(split_name_for_group(group_key))
    kept_headlines["split"] = split_names

    split_frames = {}
    for split_name, output_path in (
        ("train", TRAIN_PATH),
        ("validation", VALIDATION_PATH),
        ("test", TEST_PATH),
    ):
        split_frame = kept_headlines[kept_headlines["split"] == split_name]
        split_frames[split_name] = split_frame
        split_frame.to_csv(output_path, index=False)

    train_texts = set(split_frames["train"]["normalized_headline"])
    validation_texts = set(split_frames["validation"]["normalized_headline"])
    test_texts = set(split_frames["test"]["normalized_headline"])
    text_overlap_count = len(train_texts & validation_texts)
    text_overlap_count = text_overlap_count + len(train_texts & test_texts)
    text_overlap_count = text_overlap_count + len(validation_texts & test_texts)

    def url_set(split_frame):
        urls = set()
        for source_url in split_frame["URL"]:
            if source_url is None or pd.isna(source_url):
                continue
            url_text = str(source_url).strip()
            if url_text != "":
                urls.add(url_text)
        return urls

    train_urls = url_set(split_frames["train"])
    validation_urls = url_set(split_frames["validation"])
    test_urls = url_set(split_frames["test"])
    url_overlap_count = len(train_urls & validation_urls)
    url_overlap_count = url_overlap_count + len(train_urls & test_urls)
    url_overlap_count = url_overlap_count + len(validation_urls & test_urls)

    leakage_lines = [
        "Stage 1 leakage check. Test labels are stored in stage1_test.csv and are not listed here.",
        f"Train rows: {len(split_frames['train'])}",
        f"Validation rows: {len(split_frames['validation'])}",
        f"Test rows: {len(split_frames['test'])}",
        f"Normalized headlines that appear in more than one split: {text_overlap_count}",
        f"URLs that appear in more than one split: {url_overlap_count}",
    ]
    validation_label_counts = split_frames["validation"]["label"].value_counts()
    leakage_lines.append("")
    leakage_lines.append("Validation rows per label:")
    for label_name, label_count in validation_label_counts.items():
        leakage_lines.append(f"{label_name}: {label_count}")
    write_lines(LEAKAGE_PATH, leakage_lines)
    print("\n".join(leakage_lines))


if __name__ == "__main__":
    main()
