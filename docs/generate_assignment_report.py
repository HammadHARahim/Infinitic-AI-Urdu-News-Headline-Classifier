"""Generate the Stage 1 coursework PDF report with cover page and diagrams."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "docs" / "Urdu_News_Headline_Classification_Assignment_Report.pdf"
FIG_DIR = ROOT / "docs" / "figures"

INK = colors.HexColor("#1a2332")
MUTED = colors.HexColor("#5a6577")
RULE = colors.HexColor("#c5ccd6")
BAND = colors.HexColor("#eef2f6")
ACCENT = colors.HexColor("#1f4b7a")

# Matplotlib palette (matches report ink)
C_INK = "#1a2332"
C_MUTED = "#5a6577"
C_ACCENT = "#1f4b7a"
C_BAND = "#eef2f6"
C_LINE = "#c5ccd6"
C_GOOD = "#2f6b4f"
C_WARN = "#8a5a2b"


def styles():
    base = getSampleStyleSheet()
    return {
        "cover_course": ParagraphStyle(
            "cover_course", parent=base["Normal"], fontName="Helvetica",
            fontSize=11, textColor=MUTED, alignment=TA_CENTER, spaceAfter=6,
        ),
        "cover_title": ParagraphStyle(
            "cover_title", parent=base["Normal"], fontName="Helvetica-Bold",
            fontSize=22, leading=28, textColor=INK, alignment=TA_CENTER, spaceAfter=10,
        ),
        "cover_sub": ParagraphStyle(
            "cover_sub", parent=base["Normal"], fontName="Helvetica",
            fontSize=12, leading=16, textColor=MUTED, alignment=TA_CENTER, spaceAfter=8,
        ),
        "cover_meta": ParagraphStyle(
            "cover_meta", parent=base["Normal"], fontName="Helvetica",
            fontSize=11, leading=16, textColor=INK, alignment=TA_CENTER, spaceAfter=4,
        ),
        "h1": ParagraphStyle(
            "h1", parent=base["Heading1"], fontName="Helvetica-Bold",
            fontSize=14, textColor=ACCENT, spaceBefore=12, spaceAfter=7,
        ),
        "h2": ParagraphStyle(
            "h2", parent=base["Heading2"], fontName="Helvetica-Bold",
            fontSize=11.5, textColor=INK, spaceBefore=9, spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "body", parent=base["Normal"], fontName="Helvetica",
            fontSize=9.5, leading=13, textColor=INK, alignment=TA_JUSTIFY, spaceAfter=6,
        ),
        "bullet": ParagraphStyle(
            "bullet", parent=base["Normal"], fontName="Helvetica",
            fontSize=9.5, leading=13, textColor=INK, leftIndent=12, spaceAfter=3,
        ),
        "table_cell": ParagraphStyle(
            "table_cell", parent=base["Normal"], fontName="Helvetica",
            fontSize=8, leading=10, textColor=INK,
        ),
        "table_head": ParagraphStyle(
            "table_head", parent=base["Normal"], fontName="Helvetica-Bold",
            fontSize=8, leading=10, textColor=INK,
        ),
        "caption": ParagraphStyle(
            "caption", parent=base["Normal"], fontName="Helvetica-Oblique",
            fontSize=8, textColor=MUTED, spaceBefore=2, spaceAfter=8, alignment=TA_CENTER,
        ),
    }


def p(text, style):
    return Paragraph(text, style)


def make_table(rows, col_widths, header=True):
    s = styles()
    styled = []
    for i, row in enumerate(rows):
        cell_style = "table_head" if header and i == 0 else "table_cell"
        styled.append([p(str(c), s[cell_style]) for c in row])
    table = Table(styled, colWidths=col_widths, repeatRows=1 if header else 0)
    commands = [
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.4, RULE),
        ("BOX", (0, 0), (-1, -1), 0.8, RULE),
    ]
    if header:
        commands.append(("BACKGROUND", (0, 0), (-1, 0), BAND))
    table.setStyle(TableStyle(commands))
    return table


def fig_image(path, width_cm=16.5):
    # Keep aspect ratio from PNG size.
    from PIL import Image as PILImage
    with PILImage.open(path) as im:
        w, h = im.size
    height_cm = width_cm * (h / w)
    return Image(str(path), width=width_cm * cm, height=height_cm * cm)


def draw_rounded_box(ax, xy, width, height, text, facecolor=C_BAND, edge=C_ACCENT):
    box = FancyBboxPatch(
        xy, width, height,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.2, edgecolor=edge, facecolor=facecolor,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2, xy[1] + height / 2, text,
        ha="center", va="center", fontsize=8.5, color=C_INK, wrap=True,
    )


def make_pipeline_figure(path):
    fig, ax = plt.subplots(figsize=(10.5, 2.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3)
    ax.axis("off")
    steps = [
        (0.2, "Corpus\naudit"),
        (2.2, "Normalize\n& dedupe"),
        (4.2, "Hash split\n80/10/10"),
        (6.2, "Baselines\n& ladder"),
        (8.2, "Select on\nval macro-F1"),
        (10.2, "Test once\nfinalist"),
    ]
    for x, label in steps:
        draw_rounded_box(ax, (x, 0.85), 1.6, 1.3, label)
    for x in [1.85, 3.85, 5.85, 7.85, 9.85]:
        ax.annotate(
            "", xy=(x + 0.3, 1.5), xytext=(x, 1.5),
            arrowprops=dict(arrowstyle="->", color=C_ACCENT, lw=1.4),
        )
    ax.set_title("Figure 1. Stage 1 end-to-end pipeline", fontsize=10, color=C_INK, pad=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_architecture_figure(path):
    fig, ax = plt.subplots(figsize=(10.5, 3.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")
    draw_rounded_box(ax, (0.4, 1.4), 2.2, 1.4, "Urdu headline\nmax length 64", C_BAND, C_MUTED)
    draw_rounded_box(ax, (3.2, 1.4), 3.0, 1.4, "XLM-RoBERTa encoder\n12 layers · hidden 768\n~278M params", "#e7eef6", C_ACCENT)
    draw_rounded_box(ax, (6.8, 1.4), 2.2, 1.4, "CLS / pooled\nrepresentation", C_BAND, C_MUTED)
    draw_rounded_box(ax, (9.4, 1.4), 2.2, 1.4, "Classification head\n4 labels", "#e6f0ea", C_GOOD)
    for x0, x1 in [(2.6, 3.2), (6.2, 6.8), (9.0, 9.4)]:
        ax.annotate(
            "", xy=(x1, 2.1), xytext=(x0, 2.1),
            arrowprops=dict(arrowstyle="->", color=C_ACCENT, lw=1.4),
        )
    ax.text(4.7, 0.45, "Strategies compared: frozen encoder · full fine-tune · LoRA r=4/8/16",
            ha="center", fontsize=8, color=C_MUTED)
    ax.set_title("Figure 2. Model architecture for Stage 1 classification", fontsize=10, color=C_INK, pad=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_ladder_bar_figure(path):
    labels = [
        "Majority",
        "TF-IDF+LR",
        "Frozen",
        "LoRA r4",
        "LoRA r8",
        "LoRA r16",
        "Best LoRA\n(long)",
        "Warmup\nlinear",
        "Cosine\n+smooth",
        "Cosine\n(finalist)",
    ]
    values = [0.1437, 0.9230, 0.8538, 0.9021, 0.9042, 0.9120, 0.9395, 0.9466, 0.9466, 0.9481]
    bar_colors = [C_MUTED, C_MUTED, C_ACCENT, C_ACCENT, C_ACCENT, C_ACCENT, C_ACCENT, C_ACCENT, C_WARN, C_GOOD]
    fig, ax = plt.subplots(figsize=(10.5, 4.0))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=bar_colors, edgecolor=C_INK, linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5, color=C_INK)
    ax.set_ylabel("Validation macro-F1", fontsize=9, color=C_INK)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.9481, color=C_GOOD, linestyle="--", linewidth=1, alpha=0.8)
    ax.text(0.15, 0.958, "finalist 0.9481", fontsize=7.5, color=C_GOOD)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(C_LINE)
    ax.spines["bottom"].set_color(C_LINE)
    ax.tick_params(colors=C_MUTED)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.3f}",
                ha="center", va="bottom", fontsize=6.5, color=C_INK)
    ax.set_title("Figure 3. Validation macro-F1 across the Stage 1 ladder", fontsize=10, color=C_INK, pad=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_schedule_figure(path):
    # Conceptual LR schedule for the finalist (not a logged step curve).
    steps = np.linspace(0, 1, 400)
    warmup = 0.1
    schedule_end = 20 / 50
    lr = np.zeros_like(steps)
    for i, t in enumerate(steps):
        if t < warmup * schedule_end:
            lr[i] = (t / (warmup * schedule_end)) * 2e-5
        elif t < schedule_end:
            progress = (t - warmup * schedule_end) / (schedule_end - warmup * schedule_end)
            lr[i] = 2e-5 * 0.5 * (1 + np.cos(np.pi * progress))
        else:
            lr[i] = 2e-6
    fig, ax = plt.subplots(figsize=(10.5, 3.2))
    ax.plot(steps * 50, lr, color=C_ACCENT, linewidth=2)
    ax.axvline(20, color=C_MUTED, linestyle=":", linewidth=1)
    ax.axvline(46, color=C_GOOD, linestyle="--", linewidth=1)
    ax.text(20.3, 1.6e-5, "cosine ends\nepoch 20", fontsize=7.5, color=C_MUTED)
    ax.text(46.3, 4e-6, "best epoch 46", fontsize=7.5, color=C_GOOD)
    ax.set_xlabel("Epoch (planned)", fontsize=9, color=C_INK)
    ax.set_ylabel("Learning rate", fontsize=9, color=C_INK)
    ax.set_xlim(0, 50)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=C_MUTED)
    ax.set_title("Figure 4. Finalist learning-rate schedule (warmup → cosine → 2e-6)", fontsize=10, color=C_INK, pad=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_confusion_figure(path):
    labels = ["sports", "business", "entertainment", "technology"]
    matrix = np.array([
        [4445, 11, 70, 5],
        [10, 2342, 27, 22],
        [83, 52, 3305, 91],
        [8, 30, 65, 686],
    ], dtype=float)
    # Row-normalize for readability of error rates.
    row_sums = matrix.sum(axis=1, keepdims=True)
    norm = matrix / row_sums
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(labels, fontsize=8, color=C_INK)
    ax.set_yticklabels(labels, fontsize=8, color=C_INK)
    ax.set_xlabel("Predicted", fontsize=9, color=C_INK)
    ax.set_ylabel("True", fontsize=9, color=C_INK)
    for i in range(4):
        for j in range(4):
            count = int(matrix[i, j])
            rate = norm[i, j]
            text_color = "white" if rate > 0.55 else C_INK
            ax.text(j, i, f"{count}\n({rate:.1%})", ha="center", va="center",
                    fontsize=7.5, color=text_color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=7, colors=C_MUTED)
    cbar.set_label("Row-normalized rate", fontsize=8, color=C_MUTED)
    ax.set_title("Figure 5. Finalist test confusion matrix", fontsize=10, color=C_INK, pad=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_class_balance_figure(path):
    labels = ["sports", "entertainment", "business", "technology"]
    counts = [44829, 34901, 24131, 7999]
    fig, ax = plt.subplots(figsize=(10.5, 3.2))
    bars = ax.bar(labels, counts, color=[C_ACCENT, C_ACCENT, C_ACCENT, C_WARN], edgecolor=C_INK, linewidth=0.4)
    ax.set_ylabel("Headline count", fontsize=9, color=C_INK)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=C_MUTED)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, count + 600, f"{count:,}",
                ha="center", fontsize=8, color=C_INK)
    ax.set_title("Figure 6. Stage 1 label counts before the hash split", fontsize=10, color=C_INK, pad=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_figures():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "pipeline": FIG_DIR / "pipeline.png",
        "architecture": FIG_DIR / "architecture.png",
        "ladder": FIG_DIR / "ladder_bar.png",
        "schedule": FIG_DIR / "schedule.png",
        "confusion": FIG_DIR / "confusion.png",
        "classes": FIG_DIR / "class_balance.png",
    }
    make_pipeline_figure(paths["pipeline"])
    make_architecture_figure(paths["architecture"])
    make_ladder_bar_figure(paths["ladder"])
    make_schedule_figure(paths["schedule"])
    make_confusion_figure(paths["confusion"])
    make_class_balance_figure(paths["classes"])
    return paths


def cover(story, s):
    story.append(Spacer(1, 3.0 * cm))
    story.append(p("LLM-6XX &nbsp;|&nbsp; Assignment 1", s["cover_course"]))
    story.append(Spacer(1, 0.35 * cm))
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
    story.append(Spacer(1, 1.2 * cm))
    story.append(p("<b>Group members</b>", s["cover_meta"]))
    story.append(p("Hammad Rahim", s["cover_meta"]))
    story.append(p("Sultan Tufail", s["cover_meta"]))
    story.append(p("Adil Abdul Haq", s["cover_meta"]))
    story.append(Spacer(1, 1.3 * cm))
    story.append(p("<b>Finalist model</b>", s["cover_meta"]))
    story.append(p("xlm-roberta-base &nbsp;·&nbsp; full fine-tune &nbsp;·&nbsp; cosine schedule", s["cover_meta"]))
    story.append(Spacer(1, 0.5 * cm))
    story.append(p("<b>Primary results</b>", s["cover_meta"]))
    story.append(p("Validation macro-F1: <b>0.9481</b> &nbsp;|&nbsp; Test macro-F1: <b>0.9384</b>", s["cover_meta"]))
    story.append(Spacer(1, 1.8 * cm))
    story.append(p("Deadline: 20 October 2026", s["cover_course"]))
    story.append(p("Repository: github.com/HammadHARahim/Infinitic-AI-Urdu-News-Headline-Classifier", s["cover_course"]))
    story.append(PageBreak())


def body(story, s, figs):
    story.append(p("1. Project description", s["h1"]))
    story.append(
        p(
            "Given an Urdu news headline, predict its news category. Urdu is a low-resource "
            "language written in a right-to-left Perso-Arabic script. This project measures "
            "how well a multilingual pretrained transformer handles that setting, how close "
            "LoRA gets to full fine-tuning, and which schedule or regularization choices "
            "improve validation macro-F1.",
            s["body"],
        )
    )
    story.append(fig_image(figs["pipeline"], 16.2))
    story.append(p("Figure 1. Stage 1 end-to-end pipeline.", s["caption"]))
    story.append(p("<b>Research questions</b>", s["h2"]))
    story.append(p("• RQ1: How close does LoRA get to full fine-tuning, and with how many trainable parameters?", s["bullet"]))
    story.append(p("• RQ2 (partial): How does the XLM-R tokenizer behave on Urdu headlines?", s["bullet"]))
    story.append(p("• RQ4 (partial): Do class weights, weight decay, and label smoothing help?", s["bullet"]))
    story.append(
        p(
            "RQ3 (Qwen2.5 prompts) and E8–E10 (input length, other models, English-only BERT) "
            "remain after the deadline.",
            s["body"],
        )
    )

    story.append(p("2. Dataset and preprocessing", s["h1"]))
    story.append(
        p(
            "Stage 1 uses the public <b>Urdu News Dataset 1M</b> (Kaggle copy of Mendeley v3, "
            "DOI 10.17632/834vsxnb99.3) for non-commercial research. Direct scraping of several "
            "Urdu news sites was blocked by terms of use.",
            s["body"],
        )
    )
    story.append(fig_image(figs["classes"], 16.0))
    story.append(p("Figure 6. Stage 1 label counts (technology is the minority class).", s["caption"]))
    story.append(
        make_table(
            [
                ["Item", "Value"],
                ["Rows kept after audit", "111,860"],
                ["Stage 1 labels", "sports, business, entertainment, technology"],
                ["Split", "80% train / 10% validation / 10% test · seed 42"],
                ["Split method", "Normalized-content hash; near-duplicates stay together"],
                ["Test headlines", "11,252 (sealed until finalist)"],
            ],
            [5.5 * cm, 11.0 * cm],
        )
    )
    story.append(p("Table 1. Stage 1 corpus summary.", s["caption"]))

    story.append(p("3. Model and training setup", s["h1"]))
    story.append(fig_image(figs["architecture"], 16.2))
    story.append(p("Figure 2. Model architecture for Stage 1 classification.", s["caption"]))
    story.append(
        p(
            "Main model: <b>xlm-roberta-base</b> (~278M parameters). Hardware: Kaggle Tesla T4, "
            "batch size 32, max length 64. Optimizer AdamW. Selection metric: validation macro-F1 only.",
            s["body"],
        )
    )
    story.append(fig_image(figs["schedule"], 16.0))
    story.append(p("Figure 4. Finalist learning-rate schedule.", s["caption"]))
    story.append(
        make_table(
            [
                ["Setting", "Value"],
                ["Strategy", "Full fine-tune (not LoRA)"],
                ["Peak learning rate", "2e-5"],
                ["Schedule", "10% warmup, half-cosine over 20 epochs, then constant 2e-6"],
                ["Weight decay / loss", "0.01 / unweighted cross-entropy"],
                ["Epochs scored / best epoch", "48 of planned 50 / epoch 46"],
                ["Validation / test macro-F1", "0.9481 / 0.9384"],
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
            "beat the kept validation macro-F1.",
            s["body"],
        )
    )
    story.append(fig_image(figs["ladder"], 16.2))
    story.append(p("Figure 3. Validation macro-F1 across the Stage 1 ladder.", s["caption"]))
    story.append(
        make_table(
            [
                ["ID", "Experiment", "Status", "Val macro-F1", "Notes"],
                ["—", "Majority / TF-IDF+LR", "Done", "0.1437 / 0.9230", "Baselines"],
                ["E2", "Frozen encoder", "Done", "0.8538", "2 epochs"],
                ["E3", "Full fine-tune", "Done", "0.9481", "Cosine finalist"],
                ["E4a-c", "LoRA r=4/8/16", "Done", "≤0.9395", "Below full model"],
                ["E5", "WD / label smoothing", "Partly", "≤0.9466", "Did not beat 0.9481"],
                ["E6", "Warmup / cosine", "Done", "0.9466 → 0.9481", "Cosine won"],
                ["E7", "Class weights", "Done", "0.9430", "Did not beat then-best"],
                ["E1", "Qwen prompts", "Remaining", "—", "RQ3"],
                ["E8–E10", "Length / other models", "Remaining", "—", "After deadline"],
            ],
            [1.5 * cm, 4.0 * cm, 1.9 * cm, 2.6 * cm, 5.5 * cm],
        )
    )
    story.append(p("Table 3. Assignment ladder coverage.", s["caption"]))

    story.append(p("5. Main results", s["h1"]))
    story.append(
        make_table(
            [
                ["Run", "Epochs scored", "Best epoch", "Val macro-F1"],
                ["Warmup + linear decay", "20", "19", "0.9466"],
                ["Cosine + 2e-6 (finalist)", "48", "46", "0.9481"],
                ["Cosine + label smoothing", "60", "36", "0.9466"],
                ["Best LoRA (r=4 + smoothing)", "100", "56", "0.9395"],
            ],
            [7.5 * cm, 3.0 * cm, 2.5 * cm, 3.0 * cm],
        )
    )
    story.append(p("Table 4. Headline full-model and best LoRA scores.", s["caption"]))
    story.append(p("• Full fine-tuning beat LoRA (0.9481 vs 0.9395).", s["bullet"]))
    story.append(p("• Cosine continuation beat linear warmup by +0.0015 validation macro-F1.", s["bullet"]))
    story.append(p("• Label smoothing, stronger weight decay, and class weights did not enter the recipe.", s["bullet"]))

    story.append(p("6. Final test evaluation (once)", s["h1"]))
    story.append(
        p(
            "On 20 October 2026 the kept cosine checkpoint was declared the Stage 1 finalist. "
            "A fresh retrain from base was skipped because Kaggle GPU quota was 0.00h. "
            "The sealed test split was opened once.",
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
            ],
            [6.0 * cm, 10.5 * cm],
        )
    )
    story.append(p("Table 5. Finalist test summary.", s["caption"]))
    story.append(fig_image(figs["confusion"], 12.5))
    story.append(p("Figure 5. Finalist test confusion matrix (counts and row rates).", s["caption"]))
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
    story.append(p("Table 6. Per-label test metrics. Technology remains hardest.", s["caption"]))

    story.append(p("7. Limitations and remaining work", s["h1"]))
    story.append(p("• Four labels only in Stage 1; politics / health / world need another licensed corpus.", s["bullet"]))
    story.append(p("• E1 and E8–E10 were not completed before the deadline.", s["bullet"]))
    story.append(p("• Preferred fresh finalist retrain from base was not run (GPU quota).", s["bullet"]))
    story.append(p("• Track A (from-scratch Urdu transformer) is future work.", s["bullet"]))

    story.append(p("8. Conclusion", s["h1"]))
    story.append(
        p(
            "Stage 1 Track B delivered a reproducible fine-tuning ladder on xlm-roberta-base, "
            "selected the cosine-schedule full fine-tune on validation macro-F1 (0.9481), and "
            "scored the sealed test once (0.9384). LoRA saved parameters but did not match full "
            "fine-tuning. Schedule choice mattered more than label smoothing or class weights.",
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
    story.append(p("Table 7. Group roles for Stage 1 delivery.", s["caption"]))
    story.append(
        p(
            "Artifacts: public GitHub repository, experiments/registry.csv, score reports under "
            "data/processed/, diagrams under docs/figures/, and Hub checkpoint "
            "HammadHARahim/urdu-headline-stage1-cosine.",
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
    try:
        from PIL import Image as _PIL  # noqa: F401
    except ImportError:
        raise SystemExit("Pillow is required to embed diagrams.")
    s = styles()
    figs = build_figures()
    doc = SimpleDocTemplate(
        str(OUT_PATH),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=1.6 * cm,
        bottomMargin=2.2 * cm,
        title="Urdu News Headline Classification — Assignment Report",
        author="Hammad Rahim, Sultan Tufail, Adil Abdul Haq",
    )
    story = []
    cover(story, s)
    body(story, s, figs)
    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    print(f"Wrote figures in {FIG_DIR}")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
