"""Persists "Compare against another model" results across runs, so accuracy
can be tracked batch-over-batch (e.g. ids 150-199, then 200-249, ...) instead
of each comparison overwriting the last one in memory.

Deliberately a flat local CSV, not part of Streamlit session_state — a
session's state disappears on restart, but this history is exactly the kind
of thing you want to survive one (and to open in a spreadsheet directly)."""

import csv
import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HISTORY_PATH = REPO_ROOT / "comparison_history.csv"

FIELDS = [
    "logged_at", "label", "num_docs", "min_id", "max_id", "base_model_path",
    "base_precision", "base_recall", "base_f1",
    "pred_precision", "pred_recall", "pred_f1",
]


def log_comparison_result(
    label: str, doc_ids: list, base_model_path: str, base_agg: dict, pred_agg: dict
):
    """Append one "Score against gold" result to the history CSV. Only
    meaningful for gold-mode results — agreement-mode has no accuracy to log."""
    is_new = not HISTORY_PATH.exists()
    with HISTORY_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow({
            "logged_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "label": label,
            "num_docs": len(doc_ids),
            "min_id": min(doc_ids),
            "max_id": max(doc_ids),
            "base_model_path": base_model_path,
            "base_precision": round(base_agg["precision"], 4),
            "base_recall": round(base_agg["recall"], 4),
            "base_f1": round(base_agg["f1"], 4),
            "pred_precision": round(pred_agg["precision"], 4),
            "pred_recall": round(pred_agg["recall"], 4),
            "pred_f1": round(pred_agg["f1"], 4),
        })


def load_comparison_history() -> list:
    """[{field: value}, ...] in the order logged, oldest first. Empty list if
    nothing's been logged yet."""
    if not HISTORY_PATH.exists():
        return []
    with HISTORY_PATH.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))
