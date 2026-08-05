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


def export_session() -> dict:
    """Serialize every processed abstract into one JSON-able snapshot."""
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
    clear_status_widget_keys()
