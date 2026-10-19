# Spec: Urdu news headline classifier (Track B)

## Objective

Infinitic AI will fine-tune a pretrained multilingual transformer to classify
Urdu news text. The full label list is seven names. Stage 1 trains on four
labels from one released research corpus. Stage 2 adds politics, health, and
world news only after a second corpus has its own license file, and that
stage is a new fine-tune. This document is the specification gate. A source
stays closed until its license block below is cleared.

The reader of the notebook code is an intermediate programmer.

## Decisions already made

- Track: B, fine-tuning. The from-scratch transformer is a later track.
- Working surface: one VS Code Jupyter notebook, run section by section on a
  Google Colab runtime.
- Final labels: sports, business, entertainment, politics, technology, health,
  world news.
- Stage 1 labels, mapped from Urdu News Dataset 1M: Sports to sports,
  Business & Economics to business, Entertainment to entertainment, and
  Science & Technology to technology. Politics, health, and world news are
  absent in stage 1.
- Stage 1 text is the released story field. A separate headline column was
  not on the public dataset card. If the downloaded file has a headline
  field, the audit uses that field. If it does not, stage 1 trains on the
  story text, and the run notes say so.
- Stage 2 is a separate fine-tune after a licensed corpus adds politics,
  health, and world news. The stage 1 model is not the finalist.
- Stage 1 keeps the dataset's category label. The page-heading check applies
  to a row collected from a news site.
- Stage 1 split: 80 percent train, 10 percent validation, 10 percent test,
  after the normalized-text hash. Seed 42. The test split stays sealed.
- Selection metric: validation macro-F1 only.
- Seed: 42.
- Starting checkpoint: `xlm-roberta-base`. Its model card must list Urdu.
  Another checkpoint is allowed only after the same validation-headline
  measurements justify the change: share of headlines with an unknown token,
  mean subword tokens per word, and mean subword tokens per headline.
- Test split stays sealed until a finalist is declared.
- Target size, from the project README: about 2,000–3,000 labeled Urdu
  headlines.
- Code rule: the code must be understandable by an intermediary programmer.
  Follow `.cursor/skills/intermediate-programmer-code/SKILL.md`.

## Tech stack

- Python notebooks in VS Code, executed with the Google Colab extension.
- Colab accelerators used by this project: T4 and A100.
- L4 and G4 are outside this budget.
- Model library for the later training sections: Hugging Face Transformers.
  Versions get pinned in the notebook when that section starts.

### GPU configuration

Record the accelerator on every run.

| Job | Accelerator | Why this one |
| --- | --- | --- |
| Data audit, headline check, smoke test | T4 | These steps fit in 16 GB |
| Fine-tuning ladder | A100 | The chosen training GPU |

One runtime at a time. Stop it when the section finishes. A run that used a
T4 and a run that used an A100 are different runs. Do not compare them as if
the GPU were the same.

## Labels

The site category is the candidate label. A sports-page headline starts as
sports. The row is accepted only after a person checks the article heading.

The check confirms three things:

1. The heading is a news headline. Menus, ads, and section titles are rejected.
2. The heading is Urdu.
3. The heading belongs to the candidate category. If it does not, the row is
   rejected or given the corrected label, and the decision is stored.

```python
ALLOWED_LABELS = [
    "sports",
    "business",
    "entertainment",
    "politics",
    "technology",
    "health",
    "world news",
]


def accept_headline(headline_text, category_label, heading_was_checked):
    """Keep the row only after a person has checked the heading."""
    if heading_was_checked is not True:
        return False
    if category_label not in ALLOWED_LABELS:
        return False
    if headline_text is None or headline_text.strip() == "":
        return False
    return True
```

## Corpora and licenses

Stage 1 uses the released research corpus below. The news sites in the
following table stay blocked. Their pages do not grant collection, and a
dataset row from those sites still needs that publisher’s written grant in
`data/licenses/`.

| Stage 1 source | Record | Use condition | Status |
| --- | --- | --- | --- |
| Urdu News Dataset 1M. The download is the Kaggle copy at https://www.kaggle.com/datasets/saurabhshahane/urdu-news-dataset. That page cites Mendeley version 3, DOI 10.17632/834vsxnb99.3, not version 4. | https://data.mendeley.com/datasets/834vsxnb99/3 and `data/licenses/urdu_news_dataset_1m.md` | The authors write that the stories may be used for non-commercial research if the news source is credited. Checked 13 October 2026. | Accepted for stage 1 |

| Candidate | URL checked | What the page says | Status |
| --- | --- | --- | --- |
| ARY News Urdu | https://urdu.arynews.tv/ and https://arynews.tv/llms.txt | Model training, text and data mining, and large-scale ingestion need separate written authorization. The terms also restrict copying, downloading, and archiving. | Blocked |
| Express News | https://www.express.pk/copyright | Publication of Express material needs written permission. A reprint license is the offered route. | Blocked |
| Express Tribune | https://tribune.com.pk/copyrights and https://tribune.com.pk/privacy-policy | Content may not be published, rewritten, redistributed, or derived from without a license. The phones terms also forbid scraping and data mining. Urdu headlines only; English pages stay out. | Blocked |
| Dawn | https://www.dawn.com/terms/ | Copying, downloading, and other use beyond personal non-commercial use need prior written permission. A December 2025 report says the Urdu site DawnNews.tv closed, so it is not a live source until that is rechecked. | Blocked |
| Geo News Urdu | https://urdu.geo.tv/ | The privacy page found does not grant headline collection. | Blocked |
| Geo Sports | https://geosuper.pk/terms-and-conditions/ | Geo Super forbids copying and downloading site content without prior written permission. Confirm this is the sports site you mean. | Blocked |
| Aaj News | https://english.aaj.tv/contact-us/ | A copyright contact exists. No page was found that permits collection. | Blocked |

Permission, when it exists, is a file under `data/licenses/` containing the
publisher, the date, the Urdu URL, and the written grant. “We can scrape
them” is not that file.

## Commands

No train command exists yet. Stage 1 data work starts by downloading the
Kaggle copy of Urdu News Dataset 1M into `data/raw/`, then auditing,
deduplicating, and splitting that file. Fine-tuning waits until the data
gate, the baselines, and the smoke test pass.

## Project structure

```
SPEC.md                         this specification
.cursor/skills/                 project workflow skills
data/raw/                       collected rows, after a license file exists
data/licenses/                  written publisher grants
data/processed/                 normalized, deduplicated, split tables
notebooks/                      section-wise Colab notebooks
outputs/                        later run logs and checkpoints
```

## Code style

Names say what the value is. One action per line. A function does one job.
A comment explains a project rule when the next line is not obvious.

## Testing strategy

This gate is reviewed by reading `SPEC.md`. Later gates produce files:

- Data: a manifest with source URL, collection date, license file, and code
  version. Normalization, audit, deduplication, a hash split, and a leakage
  check. Same normalized headline and repeated URLs stay in one split.
  Split sizes are 80 percent train, 10 percent validation, 10 percent test.
  Seed 42. The test labels stay unread.
- Proposed floor: at least 200 accepted headlines in each of the seven
  labels, inside the 2,000–3,000 total. This floor waits for approval.
- Baselines, before any transformer claim: majority class and TF-IDF plus a
  linear model, scored with validation macro-F1.
- Training runs append one row to the experiment registry. `test_macro_f1`
  stays empty until a finalist is declared.

## Boundaries

- Always: seed 42, check the heading before accepting a label, record the
  GPU, keep the test split sealed, follow the intermediate-programmer skill.
- Ask first: adding a source, changing the GPU budget, opening the test set,
  or replacing `xlm-roberta-base`.
- Never: collect or train on a source that lacks written permission, bypass a
  login wall or bot block, invent a metric, or fit a tokenizer or model
  before the split is frozen.

## Approval

Approved by the project owner on 12 October 2026.

On 13 October 2026 the project owner added the two-stage corpus: stage 1 is
Urdu News Dataset 1M, and stage 2 waits for a licensed corpus that supplies
politics, health, and world news. The same day the project owner accepted
the stage 1 split: 80 percent train, 10 percent validation, 10 percent test.

This approval does not clear the Blocked news-site rows. No row from those
sites may be collected until that publisher’s written grant is in
`data/licenses/`.

## Success criteria

This specification gate passes when:

- This file is approved, including the 13 October 2026 two-stage addition.
- Every source used for collection has a written grant in `data/licenses/`.
  Stage 1 uses `data/licenses/urdu_news_dataset_1m.md`.
- The GPU table is the budget: T4 for audit and smoke tests, A100 for
  fine-tuning.
- The label rule is the category plus a stored heading check.

## Open questions

1. The seven news sites stay blocked until each publisher’s grant is filed,
   or that site is dropped. Stage 1 does not use them.
2. Is Geo Sports the Geo Super site, or a sports section on urdu.geo.tv?
   This matters only if Geo Sports is collected later.
3. The floor of 200 headlines per label applies to a label when that label
   enters the dataset. Stage 1 uses the released corpus counts for its four
   labels.
