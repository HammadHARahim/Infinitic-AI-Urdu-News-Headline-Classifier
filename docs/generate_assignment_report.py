"""Generate the Stage 1 coursework PDF report with a cover page."""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

OUT_PATH = Path(__file__).resolve().parents[1] / "docs" / "Urdu_News_Headline_Classification_Assignment_Report.pdf"

INK = colors.HexColor("#1a2332")
MUTED = colors.HexColor("#5a6577")
RULE = colors.HexColor("#c5ccd6")
BAND = colors.HexColor("#eef2f6")
ACCENT = colors.HexColor("#1f4b7a")
WHITE = colors.white


def styles():
    base = getSampleStyleSheet()
    out = {
        "cover_course": ParagraphStyle(
            "cover_course",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=11,
            textColor=MUTED,
            alignment=TA_CENTER,
            spaceAfter=6,
        ),
        "cover_title": ParagraphStyle(
            "cover_title",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=28,
            textColor=INK,
            alignment=TA_CENTER,
            spaceAfter=10,
        ),
        "cover_sub": ParagraphStyle(
            "cover_sub",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=12,
            leading=16,
            textColor=MUTED,
            alignment=TA_CENTER,
            spaceAfter=8,
        ),
        "cover_meta": ParagraphStyle(
            "cover_meta",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=11,
            leading=16,
            textColor=INK,
            alignment=TA_CENTER,
            spaceAfter=4,
        ),
        "h1": ParagraphStyle(
            "h1",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=14,
            textColor=ACCENT,
            spaceBefore=14,
            spaceAfter=8,
            borderPadding=0,
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11.5,
            textColor=INK,
            spaceBefore=10,
            spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13,
            textColor=INK,
            alignment=TA_JUSTIFY,
            spaceAfter=6,
        ),
        "bullet": ParagraphStyle(
            "bullet",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13,
            textColor=INK,
            leftIndent=12,
            spaceAfter=3,
        ),
        "table_cell": ParagraphStyle(
            "table_cell",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            textColor=INK,
        ),
        "table_head": ParagraphStyle(
            "table_head",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=10,
            textColor=INK,
        ),
        "footer": ParagraphStyle(
            "footer",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=8,
            textColor=MUTED,
            alignment=TA_CENTER,
        ),
        "caption": ParagraphStyle(
            "caption",
            parent=base["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=8,
            textColor=MUTED,
            spaceBefore=2,
            spaceAfter=8,
        ),
    }
    return out


def p(text, style):
    return Paragraph(text, style)


def make_table(rows, col_widths, header=True):
    styled = []
    for i, row in enumerate(rows):
        cell_style = "table_head" if header and i == 0 else "table_cell"
        styled.append([p(str(c), styles()[cell_style]) for c in row])
    table = Table(styled, colWidths=col_widths, repeatRows=1 if header else 0)
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), BAND) if header else ("SPAN", (0, 0), (0, 0)),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold") if header else ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.4, RULE),
        ("BOX", (0, 0), (-1, -1), 0.8, RULE),
    ]
    if not header:
        commands = [c for c in commands if c[0] not in {"BACKGROUND"}]
    table.setStyle(TableStyle(commands))
    return table


def cover(story, s):
    story.append(Spacer(1, 3.2 * cm))
    story.append(p("LLM-6XX &nbsp;|&nbsp; Assignment 1", s["cover_course"]))
    story.append(Spacer(1, 0.4 * cm))
    story.append(
        p(
            "Urdu News Headline Classification<br/>with Parameter-Efficient Fine-Tuning",
            s["cover_title"],
        )
    )
    story.append(
        p(
            "Stage 1 Track B Delivery Report<br/>Infinitic AI coursework project",
            s["cover_sub"],
        )
    )
    story.append(Spacer(1, 1.4 * cm))
    story.append(p("<b>Group members</b>", s["cover_meta"]))
    story.append(p("Hammad Rahim", s["cover_meta"]))
    story.append(p("Sultan Tufail", s["cover_meta"]))
    story.append(p("Adil Abdul Haq", s["cover_meta"]))
    story.append(Spacer(1, 1.6 * cm))
    story.append(p("<b>Finalist model</b>", s["cover_meta"]))
    story.append(p("xlm-roberta-base &nbsp;·&nbsp; full fine-tune &nbsp;·&nbsp; cosine schedule", s["cover_meta"]))
    story.append(Spacer(1, 0.6 * cm))
    story.append(p("<b>Primary results</b>", s["cover_meta"]))
    story.append(p("Validation macro-F1: <b>0.9481</b> &nbsp;|&nbsp; Test macro-F1: <b>0.9384</b>", s["cover_meta"]))
    story.append(Spacer(1, 2.0 * cm))
    story.append(p("Deadline: 20 October 2026", s["cover_course"]))
    story.append(p("Repository: github.com/HammadHARahim/Infinitic-AI-Urdu-News-Headline-Classifier", s["cover_course"]))
    story.append(PageBreak())


def body(story, s):
    story.append(p("1. Project description", s["h1"]))
    story.append(
        p(
            "Given an Urdu news headline, predict its news category. Urdu is a low-resource "
            "language written in a right-to-left Perso-Arabic script. Tokenizers built mostly "
            "on English often split Urdu words into many subword pieces. This project measures "
            "how well a multilingual pretrained transformer handles that setting, how close "
            "LoRA gets to full fine-tuning, and which regularization and schedule choices "
            "improve validation macro-F1.",
            s["body"],
        )
    )
    story.append(p("<b>Research questions addressed in Stage 1</b>", s["h2"]))
    story.append(p("• RQ1: How close does LoRA get to full fine-tuning, and with how many trainable parameters?", s["bullet"]))
    story.append(p("• RQ2 (partial): How does the XLM-R tokenizer behave on Urdu headlines?", s["bullet"]))
    story.append(p("• RQ4 (partial): Do class weights, weight decay, and label smoothing help?", s["bullet"]))
    story.append(
        p(
            "RQ3 (zero-shot / few-shot LLM with Qwen2.5) and the full tokenizer / other-model "
            "comparisons (E8–E10) remain for follow-up work after the deadline.",
            s["body"],
        )
    )

    story.append(p("2. Dataset and preprocessing", s["h1"]))
    story.append(
        p(
            "Stage 1 uses the public <b>Urdu News Dataset 1M</b> (Kaggle copy of Mendeley v3, "
            "DOI 10.17632/834vsxnb99.3). The authors allow non-commercial research use with "
            "credit to news sources. Direct scraping of several Urdu news sites was blocked by "
            "terms of use, so the released research corpus was used instead.",
            s["body"],
        )
    )
    story.append(
        make_table(
            [
                ["Item", "Value"],
                ["Rows kept after audit", "111,860"],
                ["Stage 1 labels", "sports, business, entertainment, technology"],
                ["sports / entertainment / business / technology", "44,829 / 34,901 / 24,131 / 7,999"],
                ["Split", "80% train / 10% validation / 10% test"],
                ["Seed", "42"],
                ["Split method", "Normalized-content hash; near-duplicates stay in one split"],
                ["Test headlines (sealed until finalist)", "11,252"],
            ],
            [6.5 * cm, 10.0 * cm],
        )
    )
    story.append(p("Table 1. Stage 1 corpus summary.", s["caption"]))
    story.append(
        p(
            "Headlines were normalized (whitespace and character variants), audited for empty "
            "or unmapped labels, and split before any tokenizer or model was fit. The test "
            "split stayed sealed until the finalist was declared on 20 October 2026.",
            s["body"],
        )
    )

    story.append(p("3. Model and training setup", s["h1"]))
    story.append(
        p(
            "Main model: <b>xlm-roberta-base</b> (~278M parameters) with a four-class "
            "classification head. Hardware: Kaggle Tesla T4 (one chip visible; batch size 32). "
            "Max sequence length 64. Optimizer AdamW. Selection metric: validation macro-F1 only.",
            s["body"],
        )
    )
    story.append(p("<b>Finalist recipe (kept after the ladder)</b>", s["h2"]))
    story.append(
        make_table(
            [
                ["Setting", "Value"],
                ["Strategy", "Full fine-tune (not LoRA)"],
                ["Peak learning rate", "2e-5"],
                ["Schedule", "10% warmup, half-cosine over 20 epochs, then constant 2e-6"],
                ["Weight decay", "0.01"],
                ["Loss", "Unweighted cross-entropy (no label smoothing)"],
                ["Batch size / max length / seed", "32 / 64 / 42"],
                ["Epochs scored / best epoch", "48 of planned 50 / epoch 46"],
                ["Validation macro-F1", "0.9481"],
                ["Test macro-F1 (once)", "0.9384"],
                ["Checkpoint", "HammadHARahim/urdu-headline-stage1-cosine"],
            ],
            [5.5 * cm, 11.0 * cm],
        )
    )
    story.append(p("Table 2. Declared Stage 1 finalist configuration.", s["caption"]))

    story.append(p("4. Experiment ladder", s["h1"]))
    story.append(
        p(
            "Each run changed one controlled variable. A setting joined the recipe only if it "
            "beat the kept validation macro-F1. Full run log: experiments/registry.csv.",
            s["body"],
        )
    )
    story.append(
        make_table(
            [
                ["ID", "Experiment", "Status", "Val macro-F1", "Notes"],
                ["—", "Majority / TF-IDF+LR", "Done", "0.1437 / 0.9230", "Baselines"],
                ["E2", "Frozen encoder", "Done", "0.8538", "2 epochs"],
                ["E3", "Full fine-tune", "Done", "0.9481", "Cosine finalist"],
                ["E4a", "LoRA r=4", "Done", "0.9021 → 0.9395", "Long run best LoRA"],
                ["E4b", "LoRA r=8", "Done", "0.9042 → 0.9377", "30-epoch run"],
                ["E4c", "LoRA r=16", "Done", "0.9120", "2-epoch ladder"],
                ["E5", "WD / label smoothing", "Partly", "≤0.9466", "Did not beat 0.9481"],
                ["E6", "Warmup / cosine schedule", "Done", "0.9466 → 0.9481", "Cosine won"],
                ["E7", "Class weights", "Done", "0.9430", "Did not beat then-best"],
                ["E1", "Qwen zero/few-shot LLM", "Remaining", "—", "RQ3"],
                ["E8", "Max length / headline+lead", "Remaining", "—", "Input length"],
                ["E9", "mBERT / MuRIL", "Remaining", "—", "Model comparison"],
                ["E10", "English-only BERT", "Remaining", "—", "Tokenizer effect"],
            ],
            [1.4 * cm, 4.2 * cm, 1.8 * cm, 2.4 * cm, 6.2 * cm],
        )
    )
    story.append(p("Table 3. Assignment ladder coverage (E1–E10).", s["caption"]))

    story.append(p("5. Main results", s["h1"]))
    story.append(
        make_table(
            [
                ["Run", "Epochs scored", "Best epoch", "Val macro-F1"],
                ["Warmup + linear decay", "20", "19", "0.9466"],
                ["Cosine + 2e-6 continuation (finalist)", "48", "46", "0.9481"],
                ["Cosine + label smoothing 0.1", "60", "36", "0.9466"],
                ["Best LoRA (r=4 + smoothing)", "100", "56", "0.9395"],
            ],
            [7.5 * cm, 3.0 * cm, 2.5 * cm, 3.0 * cm],
        )
    )
    story.append(p("Table 4. Headline full-model and best LoRA scores.", s["caption"]))
    story.append(p("<b>Observations</b>", s["h2"]))
    story.append(p("• Full fine-tuning clearly beat LoRA on validation macro-F1 (0.9481 vs 0.9395).", s["bullet"]))
    story.append(p("• Warmup helped; switching decay from linear to cosine + low-LR continuation gained another +0.0015.", s["bullet"]))
    story.append(p("• Label smoothing, stronger weight decay, and class weights did not beat the kept full-model score.", s["bullet"]))
    story.append(p("• Technology is the smallest class and remains the hardest label on the test set.", s["bullet"]))

    story.append(p("6. Final test evaluation (once)", s["h1"]))
    story.append(
        p(
            "On 20 October 2026 the kept cosine checkpoint was declared the Stage 1 finalist. "
            "A fresh retrain from xlm-roberta-base with only winning settings was preferred by "
            "project ADR-001 but was skipped because Kaggle GPU quota was exhausted (0.00h). "
            "The sealed test split was opened once on that checkpoint.",
            s["body"],
        )
    )
    story.append(
        make_table(
            [
                ["Metric", "Score"],
                ["Test macro-F1", "0.9384"],
                ["Test accuracy", "0.9579"],
                ["Test size", "11,252 headlines"],
                ["Eval device", "CPU"],
            ],
            [6.0 * cm, 10.5 * cm],
        )
    )
    story.append(p("Table 5. Finalist test summary.", s["caption"]))
    story.append(
        make_table(
            [
                ["Label", "Precision", "Recall", "F1", "Support"],
                ["sports", "0.9778", "0.9810", "0.9794", "4531"],
                ["business", "0.9618", "0.9754", "0.9686", "2401"],
                ["entertainment", "0.9533", "0.9360", "0.9446", "3531"],
                ["technology", "0.8532", "0.8695", "0.8613", "789"],
            ],
            [3.5 * cm, 3.0 * cm, 3.0 * cm, 3.0 * cm, 3.0 * cm],
        )
    )
    story.append(p("Table 6. Per-label test metrics.", s["caption"]))
    story.append(
        make_table(
            [
                ["", "pred sports", "pred business", "pred entertainment", "pred technology"],
                ["true sports", "4445", "11", "70", "5"],
                ["true business", "10", "2342", "27", "22"],
                ["true entertainment", "83", "52", "3305", "91"],
                ["true technology", "8", "30", "65", "686"],
            ],
            [3.4 * cm, 3.0 * cm, 3.2 * cm, 3.6 * cm, 3.2 * cm],
        )
    )
    story.append(p("Table 7. Confusion matrix (rows = true, columns = predicted).", s["caption"]))

    story.append(p("7. Limitations and remaining work", s["h1"]))
    story.append(p("• Stage 1 uses four labels only; politics, health, and world news need a second licensed corpus.", s["bullet"]))
    story.append(p("• E1 (LLM prompts), E8 (input length), E9 (other multilingual model), and E10 (English-only model) were not completed before the deadline.", s["bullet"]))
    story.append(p("• The preferred fresh finalist retrain from base was not run (GPU quota).", s["bullet"]))
    story.append(p("• Label noise from section-based labelling in the source corpus was not re-audited with a 50-headline manual sample in this delivery.", s["bullet"]))
    story.append(p("• Track A (Urdu tokenizer and transformer trained from scratch) is future work.", s["bullet"]))

    story.append(p("8. Conclusion", s["h1"]))
    story.append(
        p(
            "For Stage 1 Track B we delivered a reproducible fine-tuning ladder on "
            "xlm-roberta-base, selected the cosine-schedule full fine-tune on validation "
            "macro-F1 (0.9481), and scored the sealed test once (macro-F1 0.9384). LoRA "
            "reduced trainable parameters but did not match full fine-tuning. Schedule "
            "choice mattered more than label smoothing or class weights on this corpus. "
            "Code, experiment registry, and reports are published at the project repository.",
            s["body"],
        )
    )

    story.append(p("9. Roles and packaging", s["h1"]))
    story.append(
        make_table(
            [
                ["Member", "Primary focus for this delivery"],
                ["Hammad Rahim", "Training ladder, Kaggle runs, finalist selection, repository"],
                ["Sultan Tufail", "Data audit / split support, experiment logging, review"],
                ["Adil Abdul Haq", "Evaluation, report packaging, remaining-experiment notes"],
            ],
            [4.0 * cm, 12.5 * cm],
        )
    )
    story.append(p("Table 8. Group roles for Stage 1 delivery.", s["caption"]))
    story.append(
        p(
            "Artifacts: GitHub repository (public), experiments/registry.csv, "
            "data/processed/*.txt score reports, private Hub checkpoint "
            "HammadHARahim/urdu-headline-stage1-cosine, and notebooks under notebooks/.",
            s["body"],
        )
    )


def add_page_number(canvas, doc):
    canvas.saveState()
    page = canvas.getPageNumber()
    if page > 1:
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.5)
        canvas.line(2 * cm, 1.5 * cm, A4[0] - 2 * cm, 1.5 * cm)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(MUTED)
        canvas.drawCentredString(
            A4[0] / 2,
            1.0 * cm,
            f"Urdu News Headline Classification  ·  LLM-6XX Assignment 1  ·  Page {page - 1}",
        )
    canvas.restoreState()


def main():
    s = styles()
    doc = SimpleDocTemplate(
        str(OUT_PATH),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=1.8 * cm,
        bottomMargin=2.2 * cm,
        title="Urdu News Headline Classification — Assignment Report",
        author="Hammad Rahim, Sultan Tufail, Adil Abdul Haq",
    )
    story = []
    cover(story, s)
    body(story, s)
    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
