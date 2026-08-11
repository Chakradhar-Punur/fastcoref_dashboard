import streamlit as st

from utils.constants import NEW_SINGLETON


def cluster_status(doc: dict, cluster_id) -> str:
    return doc["cluster_statuses"].get(cluster_id, "Unverified")


def remove_mentions(doc: dict, mention_keys: list):
    """Remove a batch of mentions (identified by (start, end) pairs) in one pass."""
    keys = set(mention_keys)
    doc["mentions"] = [m for m in doc["mentions"] if (m["start"], m["end"]) not in keys]


def move_mentions(doc: dict, mention_keys: list, target):
    """Move a batch of mentions (identified by (start, end) pairs) to the same target
    cluster together, so a group selected as belonging to one entity stays grouped
    rather than each becoming its own singleton."""
    keys = set(mention_keys)
    if target == NEW_SINGLETON:
        new_id = doc["next_cluster_id"]
        doc["next_cluster_id"] += 1
    else:
        new_id = int(target)
    for m in doc["mentions"]:
        if (m["start"], m["end"]) in keys:
            m["cluster_id"] = new_id


def merge_clusters(doc: dict, source_id, target_id):
    for m in doc["mentions"]:
        if m["cluster_id"] == source_id:
            m["cluster_id"] = target_id


def find_text_occurrences(text: str, needle: str) -> list:
    """All (start, end) spans where `needle` appears in `text`, left to right.
    Used to let a reviewer add a mention the model missed entirely by typing/
    pasting its exact wording rather than needing a click target that doesn't
    exist yet (nothing in the document is highlighted for un-detected spans)."""
    if not needle:
        return []
    spans = []
    start = 0
    while True:
        idx = text.find(needle, start)
        if idx == -1:
            break
        spans.append((idx, idx + len(needle)))
        start = idx + 1
    return spans


def add_mention(doc: dict, start: int, end: int, text: str, target):
    """Add a brand-new mention (one the model never detected at all) into an
    existing cluster, or into a fresh one if target == NEW_SINGLETON."""
    if target == NEW_SINGLETON:
        cluster_id = doc["next_cluster_id"]
        doc["next_cluster_id"] += 1
    else:
        cluster_id = int(target)
    doc["mentions"].append({"start": start, "end": end, "text": text, "cluster_id": cluster_id})
    doc["mentions"].sort(key=lambda m: m["start"])


def clear_status_widget_keys(prefix: str = "status_"):
    # A segmented_control's own value (tied to its key) only falls back to our
    # `default=` when the key has never been set — so a plain dict reset isn't
    # enough to force a stale selection to update; the widget key itself must go too.
    for key in [k for k in st.session_state.keys() if k.startswith(prefix)]:
        del st.session_state[key]


def reset_corrections(doc: dict):
    doc["mentions"] = [dict(m) for m in doc["original_mentions"]]
    doc["next_cluster_id"] = doc["original_next_cluster_id"]
    doc["cluster_statuses"] = {}
    clear_status_widget_keys(f"status_{doc['id']}_")
    st.session_state.current_cluster_id = None


def clusters_for_finetuning(mentions: list) -> list:
    """Group mentions into a list of [start, end] spans per cluster — the flat
    shape most coreference fine-tuning pipelines expect as training data,
    rather than our internal flat mention-list-with-cluster-id form."""
    by_cluster = {}
    for m in mentions:
        by_cluster.setdefault(m["cluster_id"], []).append([m["start"], m["end"]])
    return [sorted(spans) for spans in by_cluster.values()]


def export_finetuning_records(use_corrected: bool) -> list:
    """One record per processed abstract — {id, label, text, clusters} — built from
    either the corrected (use_corrected=True) or original model-predicted
    (use_corrected=False) mentions. Meant to be written out as JSONL: one JSON
    object per line, one line per abstract."""
    key = "mentions" if use_corrected else "original_mentions"
    return [
        {
            "id": doc["id"],
            "label": doc["label"],
            "text": doc["text"],
            "clusters": clusters_for_finetuning(doc[key]),
        }
        for doc in st.session_state.documents
    ]


def export_session() -> dict:
    """Serialize every processed abstract into one JSON-able snapshot.

    Also carries CSV batch progress (how many rows processed so far, and an
    identity fingerprint of that CSV file) so that reloading this snapshot and
    then re-uploading the same CSV resumes the batch correctly instead of
    restarting from row 0 and re-processing (duplicating) already-done rows."""
    return {
        "documents": [
            {
                "id": doc["id"],
                "label": doc["label"],
                "text": doc["text"],
                "mentions": doc["mentions"],
                "original_mentions": doc["original_mentions"],
                "next_cluster_id": doc["next_cluster_id"],
                "original_next_cluster_id": doc["original_next_cluster_id"],
                "static_metrics": doc["static_metrics"],
                "cluster_statuses": {str(cid): s for cid, s in doc["cluster_statuses"].items()},
            }
            for doc in st.session_state.documents
        ],
        "csv_processed_count": st.session_state.get("csv_processed_count", 0),
        "csv_upload_key": st.session_state.get("csv_upload_key"),
    }


def load_session(snapshot: dict):
    documents = [
        {
            "id": doc["id"],
            "label": doc["label"],
            "text": doc["text"],
            "mentions": [dict(m) for m in doc["mentions"]],
            "original_mentions": [dict(m) for m in doc["original_mentions"]],
            "next_cluster_id": doc["next_cluster_id"],
            "original_next_cluster_id": doc["original_next_cluster_id"],
            "static_metrics": doc["static_metrics"],
            "cluster_statuses": {int(cid): s for cid, s in doc["cluster_statuses"].items()},
        }
        for doc in snapshot["documents"]
    ]
    st.session_state.documents = documents
    st.session_state.current_doc_id = documents[0]["id"] if documents else None
    st.session_state.current_cluster_id = None
    st.session_state.next_new_doc_id = max((d["id"] for d in documents), default=-1) + 1
    # Older session files won't have these keys — default to "no CSV in progress"
    # rather than erroring, so pre-existing saved sessions still load fine.
    st.session_state.csv_processed_count = snapshot.get("csv_processed_count", 0)
    st.session_state.csv_upload_key = snapshot.get("csv_upload_key")
    st.session_state.csv_data = None  # the parsed DataFrame itself isn't in the snapshot
    clear_status_widget_keys()
