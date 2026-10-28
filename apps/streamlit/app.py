"""Infinitic AI Streamlit demo for Urdu news headline classification."""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import streamlit as st
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

MODEL_ID = "HammadHARahim/urdu-headline-stage1-cosine"
ONNX_FILE = "onnx/model_quantized.onnx"
MAX_LENGTH = 64
LABELS = ["sports", "business", "entertainment", "technology"]
EXAMPLES = [
    "پاکستان کرکٹ ٹیم نے میچ جیت لیا",
    "کراچی سٹاک ایکسچینج میں اضافہ",
    "بالی ووڈ اداکارہ کی نئی فلم ریلیز",
    "نئی موبائل ٹیکنالوجی کی رونمائی",
]


def load_id_to_label():
    """Return the label map from the Hub model config."""
    config_path = hf_hub_download(repo_id=MODEL_ID, filename="config.json")
    config_text = Path(config_path).read_text(encoding="utf-8")
    config_data = json.loads(config_text)
    id_to_label = config_data.get("id2label", {})
    label_by_index = {}
    for key, label_name in id_to_label.items():
        label_by_index[int(key)] = label_name
    return label_by_index


@st.cache_resource(show_spinner="Loading Urdu classifier…")
def load_classifier():
    """Download the quantized ONNX model and tokenizer once per container."""
    model_path = hf_hub_download(repo_id=MODEL_ID, filename=ONNX_FILE)
    session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"],
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    label_by_index = load_id_to_label()
    return session, tokenizer, label_by_index


def softmax(logits):
    """Turn raw logits into probabilities that sum to 1."""
    shifted = logits - np.max(logits)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values)


def classify_headline(session, tokenizer, label_by_index, headline_text):
    """Return score-by-label and the best label for one headline."""
    encoded = tokenizer(
        headline_text,
        return_tensors="np",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
    )
    outputs = session.run(
        None,
        {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        },
    )
    logits = outputs[0][0]
    probabilities = softmax(logits)

    score_by_label = {}
    for index, score in enumerate(probabilities):
        label_name = label_by_index.get(index, LABELS[index])
        score_by_label[label_name] = float(score)

    best_label = max(score_by_label, key=score_by_label.get)
    best_score = score_by_label[best_label]
    return score_by_label, best_label, best_score


st.set_page_config(
    page_title="Infinitic AI — Urdu Headline Classifier",
    page_icon="📰",
    layout="centered",
)

st.title("Infinitic AI — Urdu News Headline Classifier")
st.caption(
    "Stage 1 demo · sports · business · entertainment · technology · "
    f"model [`{MODEL_ID}`](https://huggingface.co/{MODEL_ID}) · "
    "test macro-F1 **0.9384** · accuracy **95.79%**"
)

session, tokenizer, label_by_index = load_classifier()

example_choice = st.selectbox("Example headlines", ["(type your own)"] + EXAMPLES)
default_text = "" if example_choice == "(type your own)" else example_choice
headline_text = st.text_area(
    "Urdu news headline",
    value=default_text,
    height=120,
    placeholder="Paste an Urdu headline here",
)

if st.button("Classify", type="primary"):
    cleaned_text = headline_text.strip()
    if cleaned_text == "":
        st.warning("Please paste an Urdu news headline.")
    else:
        score_by_label, best_label, best_score = classify_headline(
            session,
            tokenizer,
            label_by_index,
            cleaned_text,
        )
        st.success(f"Predicted: **{best_label}** ({best_score:.1%})")
        chart_rows = []
        for label_name in LABELS:
            chart_rows.append(
                {
                    "label": label_name,
                    "score": score_by_label.get(label_name, 0.0),
                }
            )
        st.bar_chart({row["label"]: row["score"] for row in chart_rows})

st.markdown(
    "Code: "
    "[GitHub](https://github.com/HammadHARahim/Infinitic-AI-Urdu-News-Headline-Classifier) · "
    "Browser demo: "
    "[Hugging Face Space](https://huggingface.co/spaces/HammadHARahim/infinitic-urdu-headline-classifier)"
)
