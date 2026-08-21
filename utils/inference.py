import time

import streamlit as st
import torch
from fastcoref import FCoref

# Set to a local fine-tuned checkpoint (e.g. "finetune_runs/<run_id>/model") to use
# it instead of the base model. None/"" = base model. This is the one line that
# controls which model the whole app runs — active_model_label() below just
# describes it for the UI, so the two can never drift out of sync.
MODEL_NAME_OR_PATH = "/Users/Chakradhar/fastcoref_dashboard/finetune_runs/20260818-201057/model"


def active_model_label() -> str:
    """Human-readable description of the model load_model() will load. Reads
    MODEL_NAME_OR_PATH only — doesn't trigger an actual (slow, cached) model
    load, so it's cheap to call from anywhere in the UI, e.g. every rerun."""
    return "Fine-tuned model" if MODEL_NAME_OR_PATH else "Base model"


@st.cache_resource(show_spinner=False)
def load_model():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if MODEL_NAME_OR_PATH:
        return FCoref(device=device, model_name_or_path=MODEL_NAME_OR_PATH)
    return FCoref(device=device)


def run_inference(text: str):
    model = load_model()

    start = time.time()
    preds = model.predict(texts=[text])
    elapsed = time.time() - start

    pred = preds[0]
    span_clusters = pred.get_clusters(as_strings=False)
    string_clusters = pred.get_clusters(as_strings=True)

    mentions = []
    for cluster_id, (spans, strings) in enumerate(zip(span_clusters, string_clusters)):
        for (s, e), text_str in zip(spans, strings):
            mentions.append({"start": s, "end": e, "text": text_str, "cluster_id": cluster_id})
    mentions.sort(key=lambda m: m["start"])

    static_metrics = {
        "inference_seconds": round(elapsed, 3),
        "doc_words": len(text.split()),
        "model": "FCoref",
    }
    return mentions, len(span_clusters), static_metrics
