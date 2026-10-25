"""Build Infinitic AI LinkedIn carousel PNGs with measured metrics only."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent / "pngs"
OUT.mkdir(parents=True, exist_ok=True)

# Brand palette — ink / steel / green accent (not purple/cream defaults)
BG = (246, 248, 251)
INK = (22, 32, 48)
MUTED = (90, 104, 122)
ACCENT = (20, 74, 120)
GREEN = (36, 110, 78)
AMBER = (140, 90, 40)
WHITE = (255, 255, 255)
CARD = (255, 255, 255)
RULE = (210, 218, 228)
W, H = 1080, 1080


def font(size, bold=False):
    paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for path in paths:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def new_canvas():
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    # top brand bar
    draw.rectangle([0, 0, W, 14], fill=ACCENT)
    return img, draw


def footer(draw, page, total=5):
    draw.rectangle([0, H - 70, W, H], fill=WHITE)
    draw.line([60, H - 70, W - 60, H - 70], fill=RULE, width=2)
    draw.text((60, H - 48), "Infinitic AI  ·  Urdu News Headline Classifier", font=font(22), fill=MUTED)
    draw.text((W - 120, H - 48), f"{page}/{total}", font=font(22, True), fill=ACCENT)


def wrap(draw, text, fnt, max_width):
    words = text.split()
    lines, cur = [], ""
    for word in words:
        trial = (cur + " " + word).strip()
        if draw.textlength(trial, font=fnt) <= max_width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def card1_hero():
    img, draw = new_canvas()
    draw.text((60, 60), "INFINITIC AI", font=font(28, True), fill=ACCENT)
    draw.text((60, 110), "Client briefing", font=font(24), fill=MUTED)

    title = "Urdu news headlines,\nclassified at production\nquality."
    y = 200
    for line in title.split("\n"):
        draw.text((60, y), line, font=font(56, True), fill=INK)
        y += 70

    # metric pills
    boxes = [
        (60, 460, "Test macro-F1", "0.9384"),
        (390, 460, "Test accuracy", "95.79%"),
        (720, 460, "Headlines scored", "11,252"),
    ]
    for x, y0, label, value in boxes:
        draw.rounded_rectangle([x, y0, x + 300, y0 + 160], radius=18, fill=WHITE, outline=RULE, width=2)
        draw.text((x + 24, y0 + 28), label, font=font(22), fill=MUTED)
        draw.text((x + 24, y0 + 72), value, font=font(44, True), fill=ACCENT)

    draw.text((60, 680), "4 categories  ·  XLM-RoBERTa full fine-tune  ·  sealed test once", font=font(24), fill=MUTED)
    draw.text((60, 740), "Built for media ops, newsrooms, and Urdu content platforms.", font=font(26), fill=INK)
    footer(draw, 1)
    img.save(OUT / "01_hero.png", quality=95)


def card2_achievements():
    img, draw = new_canvas()
    draw.text((60, 50), "INFINITIC AI", font=font(26, True), fill=ACCENT)
    draw.text((60, 100), "What we shipped", font=font(48, True), fill=INK)

    items = [
        ("111,860", "Urdu headlines audited and split\n(80/10/10, seed 42, sealed test)"),
        ("0.9481", "Best validation macro-F1\n(cosine schedule, epoch 46)"),
        ("0.9384", "Held-out test macro-F1\n(one evaluation only — no leakage)"),
        ("LoRA ladder", "Ranks 4 / 8 / 16 measured;\nfull fine-tune still wins for quality"),
    ]
    y = 190
    for value, desc in items:
        draw.rounded_rectangle([60, y, W - 60, y + 150], radius=16, fill=WHITE, outline=RULE, width=2)
        draw.rectangle([60, y, 78, y + 150], fill=GREEN)
        draw.text((110, y + 28), value, font=font(36, True), fill=INK)
        dy = y + 78
        for line in desc.split("\n"):
            draw.text((110, dy), line, font=font(24), fill=MUTED)
            dy += 32
        y += 170
    footer(draw, 2)
    img.save(OUT / "02_achievements.png", quality=95)


def card3_vs_baselines():
    img, draw = new_canvas()
    draw.text((60, 50), "INFINITIC AI", font=font(26, True), fill=ACCENT)
    draw.text((60, 100), "Better than our own baselines", font=font(42, True), fill=INK)
    draw.text((60, 160), "Same data · same split · fair comparison", font=font(24), fill=MUTED)

    rows = [
        ("Majority class", 0.1437, False),
        ("Frozen encoder", 0.8538, False),
        ("TF-IDF + logistic", 0.9230, False),
        ("Best LoRA (val)", 0.9395, False),
        ("Infinitic finalist (val)", 0.9481, True),
        ("Infinitic finalist (test)", 0.9384, True),
    ]
    y = 230
    max_bar = W - 280
    for name, score, highlight in rows:
        draw.text((60, y), name, font=font(24, highlight), fill=INK if highlight else MUTED)
        bar_w = int(max_bar * score)
        draw.rounded_rectangle([60, y + 36, 60 + max_bar, y + 64], radius=8, fill=RULE)
        color = GREEN if highlight else ACCENT
        draw.rounded_rectangle([60, y + 36, 60 + bar_w, y + 64], radius=8, fill=color)
        draw.text((70 + max_bar, y + 34), f"{score:.4f}", font=font(24, True), fill=INK)
        y += 100

    draw.text((60, 880), "Metric: macro-F1  ·  Task: 4-class Urdu news category", font=font(22), fill=MUTED)
    footer(draw, 3)
    img.save(OUT / "03_vs_baselines.png", quality=95)


def card4_external_context():
    img, draw = new_canvas()
    draw.text((60, 50), "INFINITIC AI", font=font(26, True), fill=ACCENT)
    draw.text((60, 100), "How we sit vs public Urdu work", font=font(40, True), fill=INK)

    note = (
        "Published numbers use different datasets and label sets. "
        "We do not claim a single global leaderboard win — we show context."
    )
    y = 170
    for line in wrap(draw, note, font(23), W - 120):
        draw.text((60, y), line, font=font(23), fill=MUTED)
        y += 32

    # comparison cards — wrap body text so nothing spills past the box edge
    panels = [
        (
            "Infinitic AI (this project)",
            [
                "Task: 4-class news category",
                "Corpus: Urdu News Dataset 1M",
                "Test macro-F1: 0.9384",
                "Test accuracy: 95.79%",
                "Process: sealed test + experiment ladder",
            ],
            GREEN,
        ),
        (
            "Spark NLP Urdu news classifier*",
            [
                "Task: 7-class news mix",
                "Public benchmark card",
                "Macro-F1: ~0.75",
                "Accuracy: ~0.82",
                "More classes → harder problem",
            ],
            AMBER,
        ),
    ]
    panel_w = 460
    panel_pad = 28
    inner_w = panel_w - (panel_pad * 2)
    body_font = font(21)
    title_font = font(23, True)
    x = 60
    for title, lines, accent in panels:
        draw.rounded_rectangle([x, 310, x + panel_w, 820], radius=18, fill=WHITE, outline=RULE, width=2)
        draw.rectangle([x, 310, x + panel_w, 330], fill=accent)
        title_y = 350
        for title_line in wrap(draw, title, title_font, inner_w):
            draw.text((x + panel_pad, title_y), title_line, font=title_font, fill=INK)
            title_y += 30
        yy = max(title_y + 18, 410)
        for line in lines:
            # Wrap the full drawn string so the bullet prefix cannot overflow the box.
            wrapped = wrap(draw, "•  " + line, body_font, inner_w)
            for i, part in enumerate(wrapped):
                if i > 0 and not part.startswith("•"):
                    part = "   " + part.lstrip()
                draw.text((x + panel_pad, yy), part, font=body_font, fill=MUTED)
                # hard assert for layout QA
                right_edge = x + panel_pad + draw.textlength(part, font=body_font)
                if right_edge > x + panel_w - panel_pad + 1:
                    raise SystemExit(f"overflow in card4: {part!r} ends at {right_edge}")
                yy += 32
            yy += 12
        x += 500

    draw.text((60, 860), "*John Snow Labs Spark NLP model card (classifierdl_bert_news, Urdu).", font=font(20), fill=MUTED)
    draw.text((60, 900), "Different label set — shown for market context, not identical eval.", font=font(20), fill=MUTED)
    footer(draw, 4)
    img.save(OUT / "04_external_context.png", quality=95)


def card5_client_value():
    img, draw = new_canvas()
    draw.text((60, 50), "INFINITIC AI", font=font(26, True), fill=ACCENT)
    draw.text((60, 100), "Why this matters for clients", font=font(42, True), fill=INK)

    points = [
        ("Routing & discovery", "Auto-tag Urdu stories into sports, business, entertainment, technology."),
        ("Measured quality", "95.79% test accuracy and 0.9384 macro-F1 on 11k held-out headlines."),
        ("Honest engineering", "We beat TF-IDF (0.9230) and LoRA; we publish what lost, too."),
        ("Ready next steps", "Add politics/health/world with licensed data; API packaging; monitoring."),
    ]
    y = 190
    for title, body in points:
        draw.rounded_rectangle([60, y, W - 60, y + 150], radius=16, fill=WHITE, outline=RULE, width=2)
        draw.text((90, y + 28), title, font=font(28, True), fill=ACCENT)
        for i, line in enumerate(wrap(draw, body, font(24), W - 180)):
            draw.text((90, y + 78 + i * 30), line, font=font(24), fill=INK)
        y += 170

    footer(draw, 5)
    img.save(OUT / "05_client_value.png", quality=95)


def write_caption():
    text = """Infinitic AI — Urdu News Headline Classifier (Stage 1)

We fine-tuned XLM-RoBERTa for 4-class Urdu news category routing on the Urdu News Dataset 1M research corpus.

Results (held-out test, scored once):
• Test macro-F1: 0.9384
• Test accuracy: 95.79%
• 11,252 sealed test headlines

Same-split baselines we beat:
• Majority → 0.1437
• TF-IDF + logistic → 0.9230
• Best LoRA (val) → 0.9395
• Our finalist (val) → 0.9481

Public context (different datasets/labels — not identical eval):
• Spark NLP’s public Urdu news classifier reports ~0.75 macro-F1 / ~0.82 accuracy on a 7-class mix. Our Stage 1 system is a focused 4-class production path with a sealed-test protocol.

What we offer clients: measurable Urdu content tagging, experiment discipline, and a clear roadmap to more labels once licensed data is in place.

#InfiniticAI #UrduNLP #AI #MachineLearning #NewsTech #LowResourceNLP
"""
    (OUT / "linkedin_caption.txt").write_text(text, encoding="utf-8")


def main():
    card1_hero()
    card2_achievements()
    card3_vs_baselines()
    card4_external_context()
    card5_client_value()
    write_caption()
    print(f"Wrote PNGs to {OUT}")
    for path in sorted(OUT.glob("*.png")):
        print(path.name, path.stat().st_size)


if __name__ == "__main__":
    main()
