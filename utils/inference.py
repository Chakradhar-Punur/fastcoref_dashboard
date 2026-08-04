import time

import streamlit as st
import torch
from fastcoref import FCoref


@st.cache_resource(show_spinner=False)
def load_model():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
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
