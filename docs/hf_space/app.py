"""Infinitic AI Gradio demo for Urdu news headline classification."""

import gradio as gr
from transformers import pipeline

MODEL_NAME = "HammadHARahim/urdu-headline-stage1-cosine"
LABELS = ["sports", "business", "entertainment", "technology"]

classifier = pipeline(
    task="text-classification",
    model=MODEL_NAME,
    tokenizer=MODEL_NAME,
)

EXAMPLES = [
    "پاکستان کرکٹ ٹیم نے میچ جیت لیا",
    "کراچی سٹاک ایکسچینج میں اضافہ",
    "بالی ووڈ اداکارہ کی نئی فلم ریلیز",
    "نئی موبائل ٹیکنالوجی کی رونمائی",
]


def classify_headline(headline_text):
    """Return the predicted label and score for one Urdu headline."""
    cleaned_text = headline_text.strip()
    if cleaned_text == "":
        return {}, "Please paste an Urdu news headline."

    predictions = classifier(cleaned_text, top_k=None)
    score_by_label = {}
    for item in predictions:
        label_name = item["label"]
        score_value = float(item["score"])
        score_by_label[label_name] = score_value

    best_label = None
    best_score = -1.0
    for label_name in score_by_label:
        if score_by_label[label_name] > best_score:
            best_score = score_by_label[label_name]
            best_label = label_name

    summary = f"Predicted: {best_label} ({best_score:.2%})"
    return score_by_label, summary


demo = gr.Interface(
    fn=classify_headline,
    inputs=gr.Textbox(
        lines=3,
        label="Urdu news headline",
        placeholder="Paste an Urdu headline here",
    ),
    outputs=[
        gr.Label(num_top_classes=4, label="Class scores"),
        gr.Textbox(label="Summary"),
    ],
    title="Infinitic AI — Urdu News Headline Classifier",
    description=(
        "Stage 1 demo: sports · business · entertainment · technology. "
        "Model: HammadHARahim/urdu-headline-stage1-cosine. "
        "Test macro-F1 0.9384 · accuracy 95.79%."
    ),
    examples=EXAMPLES,
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch()
