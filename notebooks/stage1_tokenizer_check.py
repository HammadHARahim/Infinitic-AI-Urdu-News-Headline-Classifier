"""Measure how xlm-roberta-base splits the validation headlines.

The test split is not read.
"""

from transformers import AutoTokenizer

import pandas as pd

MODEL_NAME = "xlm-roberta-base"
VALIDATION_PATH = "data/processed/stage1_validation.csv"
REPORT_PATH = "data/processed/stage1_tokenizer.txt"


def write_lines(path, lines):
    """Write a text report."""
    with open(path, "w", encoding="utf-8") as report_file:
        report_file.write("\n".join(lines))
        report_file.write("\n")


def headline_token_ids(tokenizer, headline_text):
    """Return subword ids for one headline, without the special start and end tokens."""
    token_ids = tokenizer.encode(
        headline_text,
        add_special_tokens=False,
    )
    return token_ids


def main():
    validation_rows = pd.read_csv(VALIDATION_PATH)
    headlines = validation_rows["normalized_headline"].tolist()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    unknown_token_id = tokenizer.unk_token_id

    headline_count = 0
    unknown_headline_count = 0
    token_count = 0
    word_count = 0

    for headline_text in headlines:
        words = str(headline_text).split()
        if len(words) == 0:
            continue
        token_ids = headline_token_ids(tokenizer, str(headline_text))
        headline_count = headline_count + 1
        token_count = token_count + len(token_ids)
        word_count = word_count + len(words)
        headline_has_unknown = unknown_token_id in token_ids
        if headline_has_unknown:
            unknown_headline_count = unknown_headline_count + 1

    unknown_share = unknown_headline_count / headline_count
    tokens_per_word = token_count / word_count
    tokens_per_headline = token_count / headline_count

    report_lines = [
        "Stage 1 tokenizer check on validation headlines only. The test split was not read.",
        f"Tokenizer: {MODEL_NAME}",
        f"Validation headlines measured: {headline_count}",
        f"Share of headlines containing an unknown token: {unknown_share:.4f}",
        f"Mean subword tokens per word: {tokens_per_word:.4f}",
        f"Mean subword tokens per headline: {tokens_per_headline:.4f}",
    ]
    write_lines(REPORT_PATH, report_lines)
    print("\n".join(report_lines))


if __name__ == "__main__":
    main()
