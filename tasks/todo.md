# Tasks: stage 1 data

## Task 1: Download Urdu News Dataset 1M

**Description:** Save the Kaggle copy of Urdu News Dataset 1M under `data/raw/` and record the file name and download date.

**Acceptance criteria:**
- [x] The file is the Kaggle copy that cites Mendeley version 3, DOI 10.17632/834vsxnb99.3
- [x] `data/licenses/urdu_news_dataset_1m.md` is the license note for that file

**Verification:**
- [x] Manual check: the raw path, file size, and download date are written in the manifest

**Dependencies:** None

**Files likely touched:**
- `data/raw/`
- `data/manifest.csv`

**Estimated scope:** Small

## Task 2: Audit the release

**Description:** Count rows, list columns, map the four source categories to sports, business, entertainment, and technology, and count empty text.

**Acceptance criteria:**
- [x] The audit states whether a headline column exists
- [x] Each of the four labels has a row count

**Verification:**
- [x] Manual check: `data/processed/stage1_audit.txt` contains those counts

**Dependencies:** Task 1

**Files likely touched:**
- `notebooks/01_stage1_audit.ipynb`
- `data/processed/stage1_audit.txt`

**Estimated scope:** Medium

## Task 3: Normalize, deduplicate, and split

**Description:** Normalize the text, hash it, keep duplicate text in one split, and write train, validation, and test files at 80 percent, 10 percent, and 10 percent with seed 42.

**Acceptance criteria:**
- [x] No normalized text appears in more than one split
- [x] Test labels are stored and not printed for model selection

**Verification:**
- [x] Manual check: a leakage report shows zero overlapping hashes

**Dependencies:** Task 2

**Files likely touched:**
- `notebooks/02_stage1_split.ipynb`
- `data/processed/`

**Estimated scope:** Medium

## Checkpoint: after Tasks 1-3

- [ ] The data gate artifacts exist
- [ ] Review with the project owner before baselines or fine-tuning
