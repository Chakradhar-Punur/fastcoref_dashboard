"""
Coreference Resolution Dashboard — Streamlit version.

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""

import json

import pandas as pd
import requests
import streamlit as st

from utils.constants import NEW_SINGLETON, STATUS_OPTIONS
from utils.clusters import cluster_label, derive_clusters
from utils.inference import run_inference
from utils.mention_selector import mention_click_selector
from utils.rendering import mention_context
from utils.scoring import compute_gold_score, compute_pairwise_prf
from utils.session import (
    clear_status_widget_keys,
    cluster_status,
    export_session,
    load_session,
    merge_clusters,
    move_mentions,
    remove_mentions,
    reset_corrections,
)
from utils.text_processing import clean_text, extract_documents_from_url, extract_text

st.set_page_config(page_title="Coreference Resolution Dashboard", layout="wide")

NAV_SECTIONS = [
    ("Run inference", ":material/play_arrow:"),
    ("Correct", ":material/edit:"),
    ("View corrected & predicted clusters", ":material/compare_arrows:"),
    ("Metrics", ":material/analytics:"),
]

# Session state
st.session_state.setdefault("documents", [])  # list of per-abstract dicts
st.session_state.setdefault("current_doc_id", None)  # which abstract is on screen
st.session_state.setdefault("current_cluster_id", None)  # which cluster is expanded
st.session_state.setdefault("nav_section", NAV_SECTIONS[0][0])


def _doc_by_id(doc_id):
    return next((d for d in st.session_state.documents if d["id"] == doc_id), None)


# --- Sidebar: navigation + abstract picker ---
with st.sidebar:
    st.header("Coreference dashboard")

    for section, icon in NAV_SECTIONS:
        is_current = section == st.session_state.nav_section
        if st.button(
            section, key=f"nav_{section}", icon=icon, width="stretch",
            type="primary" if is_current else "secondary",
        ):
            st.session_state.nav_section = section
            st.rerun()

    st.divider()
    st.caption("Abstracts")

    if not st.session_state.documents:
        st.caption("None processed yet — go to Run inference.")
    else:
        for d in st.session_state.documents:
            is_current_doc = d["id"] == st.session_state.current_doc_id
            n_clusters = len({m["cluster_id"] for m in d["mentions"]})
            if st.button(
                d["label"][:70] + ("…" if len(d["label"]) > 70 else ""),
                key=f"docsel_{d['id']}",
                width="stretch",
                type="primary" if is_current_doc else "secondary",
                help=f"{n_clusters} clusters",
            ):
                st.session_state.current_doc_id = d["id"]
                st.session_state.current_cluster_id = None
                st.rerun()


# --- Main area ---
st.title("Coreference resolution dashboard")

section = st.session_state.nav_section
doc = _doc_by_id(st.session_state.current_doc_id)

if section == "Run inference":
    st.caption(
        "Upload PDFs/.txt files or paste a web page URL — a page listing several write-ups "
        "is split into one abstract per entry."
    )

    input_mode = st.segmented_control(
        "Input source", ["Upload file", "From URL"], key="input_mode", default="Upload file"
    )
    if input_mode == "From URL":
        url_input = st.text_input(
            "Web page URL",
            placeholder="https://example.com/abstracts",
            label_visibility="collapsed",
        )
        uploaded_files = None
    else:
        uploaded_files = st.file_uploader(
            "Upload document(s)", type=["pdf", "txt"], accept_multiple_files=True,
            label_visibility="collapsed",
        )
        url_input = ""

    run_clicked = st.button(
        "Run inference", type="primary", icon=":material/play_arrow:", width="stretch"
    )

    if run_clicked:
        if input_mode == "From URL":
            if not url_input.strip():
                st.warning("Please enter a URL first.")
                st.stop()
            with st.spinner("Fetching and splitting the page..."):
                try:
                    raw_docs = extract_documents_from_url(url_input.strip())
                except requests.RequestException as e:
                    st.error(f"Couldn't fetch that URL: {e}")
                    st.stop()
        else:
            if not uploaded_files:
                st.warning("Please upload at least one file first.")
                st.stop()
            with st.spinner("Extracting text..."):
                raw_docs = [{"label": f.name, "text": extract_text(f)} for f in uploaded_files]

        cleaned_docs = [{"label": d["label"], "text": clean_text(d["text"])} for d in raw_docs]
        cleaned_docs = [d for d in cleaned_docs if d["text"].strip()]

        if not cleaned_docs:
            st.error(
                "No text could be extracted. If this was a PDF, it may be scanned/image-based — "
                "try a different file, OCR extraction, or a URL instead."
            )
            st.stop()

        documents = []
        with st.spinner(f"Running inference on {len(cleaned_docs)} abstract(s)... this can take a while on CPU."):
            for i, d in enumerate(cleaned_docs):
                mentions, num_clusters, static_metrics = run_inference(d["text"])
                documents.append({
                    "id": i,
                    "label": d["label"],
                    "text": d["text"],
                    "mentions": mentions,
                    "original_mentions": [dict(m) for m in mentions],
                    "next_cluster_id": num_clusters,
                    "original_next_cluster_id": num_clusters,
                    "static_metrics": static_metrics,
                    "cluster_statuses": {},
                })

        st.session_state.documents = documents
        st.session_state.current_doc_id = documents[0]["id"]
        st.session_state.current_cluster_id = None
        clear_status_widget_keys()
        st.session_state.nav_section = "Correct"
        st.success(f"Done. Processed {len(documents)} abstract(s).")
        st.rerun()

    with st.expander("Resume a saved session"):
        st.caption(
            "Pick up where you left off. Upload a session file you previously saved with "
            "'Download session (JSON)' — this restores every abstract, your corrections, "
            "and verification statuses without re-running inference."
        )
        session_file = st.file_uploader(
            "Session file", type=["json"], label_visibility="collapsed", key="session_file_uploader"
        )
        if st.button("Load session", icon=":material/upload_file:", disabled=session_file is None):
            try:
                snapshot = json.load(session_file)
                load_session(snapshot)
            except (json.JSONDecodeError, KeyError) as e:
                st.error(f"Couldn't load this session file: {e}")
            else:
                st.rerun()

    if st.session_state.documents:
        st.divider()
        st.caption(
            f"{len(st.session_state.documents)} abstract(s) processed — pick one in the sidebar, "
            "then use Correct / View corrected & predicted clusters / Metrics above."
        )

elif doc is None:
    st.info("Go to 'Run inference' in the sidebar to add a file or URL first.")

else:
    doc_id = doc["id"]
    text = doc["text"]
    clusters = derive_clusters(doc["mentions"])
    original_clusters = derive_clusters(doc["original_mentions"])
    labels_by_id = {c["id"]: cluster_label(c) for c in clusters}
    clusters_by_id = {c["id"]: c for c in clusters}

    statuses = [cluster_status(doc, c["id"]) for c in clusters]
    verified_correct = statuses.count("Correct")
    verified_incorrect = statuses.count("Incorrect")
    verified_unsure = statuses.count("Unsure")
    verified_pending = statuses.count("Unverified")

    st.header(doc["label"])

    if section == "Correct":
        header_cols = st.columns([2, 2])
        with header_cols[0]:
            status_filter = st.segmented_control(
                "Filter", ["All"] + STATUS_OPTIONS, key=f"statusfilter_{doc_id}", default="All"
            )
        with header_cols[1]:
            show_document = st.toggle("Show document view", key=f"showdoc_{doc_id}")

        display_clusters = sorted(clusters, key=lambda c: c["id"])
        if status_filter != "All":
            display_clusters = [c for c in display_clusters if cluster_status(doc, c["id"]) == status_filter]

        with st.expander("Cluster size distribution"):
            if clusters:
                dist_df = pd.DataFrame([{"id": c["id"], "size": c["size"]} for c in clusters])
                st.bar_chart(dist_df.set_index("id")["size"])
            else:
                st.caption("No clusters to chart.")

        if not display_clusters:
            st.caption("No clusters match this filter.")
        else:
            pill_ids = [c["id"] for c in display_clusters]
            if st.session_state.current_cluster_id not in pill_ids:
                st.session_state.current_cluster_id = pill_ids[0]

            chosen_id = st.pills(
                "Click a cluster to inspect its entity and mentions",
                pill_ids,
                selection_mode="single",
                default=st.session_state.current_cluster_id,
                format_func=lambda x: f"Cluster {x}",
                key=f"clusterpills_{doc_id}",
            )
            if chosen_id is not None and chosen_id != st.session_state.current_cluster_id:
                st.session_state.current_cluster_id = chosen_id
                st.rerun()

            c = clusters_by_id[st.session_state.current_cluster_id]

            if show_document:
                left, right = st.columns(2)
            else:
                left, right = st.container(), None

            # --- Left: current cluster's review card ---
            with left:
                with st.container(border=True):
                    top_cols = st.columns([5, 3])
                    with top_cols[0]:
                        st.markdown(f"**Cluster {c['id']}** — {labels_by_id[c['id']]}")
                    with top_cols[1]:
                        current_status = cluster_status(doc, c["id"])
                        picked_status = st.segmented_control(
                            "Status",
                            STATUS_OPTIONS,
                            key=f"status_{doc_id}_{c['id']}",
                            default=current_status,
                            label_visibility="collapsed",
                        )
                        doc["cluster_statuses"][c["id"]] = picked_status or current_status

                    other_ids = [oc["id"] for oc in clusters if oc["id"] != c["id"]]

                    def render_mention_row(m, other_ids=other_ids):
                        before, mention_text, after = mention_context(text, m["start"], m["end"])
                        row_cols = st.columns([1, 9])
                        with row_cols[0]:
                            st.checkbox(
                                "Select for bulk move",
                                key=f"select_{doc_id}_{m['start']}_{m['end']}",
                                label_visibility="collapsed",
                                help="Select, then use 'Move selected'/'Remove selected' below",
                            )
                        with row_cols[1]:
                            st.markdown(f"{before}**{mention_text}**{after}")

                    inline_mentions = c["mentions"][:8]
                    overflow_mentions = c["mentions"][8:]
                    for m in inline_mentions:
                        render_mention_row(m)
                    if overflow_mentions:
                        with st.expander(f"{len(overflow_mentions)} more mentions"):
                            for m in overflow_mentions:
                                render_mention_row(m)

                    selected_mentions = [
                        m for m in c["mentions"]
                        if st.session_state.get(f"select_{doc_id}_{m['start']}_{m['end']}", False)
                    ]
                    if selected_mentions:
                        with st.container(border=True):
                            st.caption(f"{len(selected_mentions)} mention(s) selected")
                            bulk_options = ["Move selected to…"] + other_ids + [NEW_SINGLETON]
                            bulk_options_key = "-".join(str(i) for i in other_ids)
                            bulk_cols = st.columns([5, 2, 2])
                            with bulk_cols[0]:
                                bulk_target = st.selectbox(
                                    "Move selected to",
                                    bulk_options,
                                    key=f"bulkmove_{doc_id}_{c['id']}_{bulk_options_key}",
                                    label_visibility="collapsed",
                                    format_func=lambda x: (
                                        "Move selected to…" if x == "Move selected to…"
                                        else "Split selected into a new entity" if x == NEW_SINGLETON
                                        else f"Cluster {x} — {labels_by_id.get(x, '')}"
                                    ),
                                )
                            with bulk_cols[1]:
                                if st.button(
                                    "Move selected", icon=":material/drive_file_move:", width="stretch",
                                    key=f"bulkmovebtn_{doc_id}_{c['id']}",
                                    disabled=bulk_target == "Move selected to…",
                                ):
                                    move_mentions(
                                        doc, [(m["start"], m["end"]) for m in selected_mentions], bulk_target
                                    )
                                    st.rerun()
                            with bulk_cols[2]:
                                if st.button(
                                    "Remove selected", icon=":material/delete:", width="stretch",
                                    key=f"bulkremovebtn_{doc_id}_{c['id']}",
                                ):
                                    remove_mentions(doc, [(m["start"], m["end"]) for m in selected_mentions])
                                    st.rerun()

                    if other_ids:
                        options_key = "-".join(str(i) for i in other_ids)
                        merge_cols = st.columns([3, 2])
                        with merge_cols[0]:
                            target = st.selectbox(
                                "Merge into",
                                other_ids,
                                key=f"mergesel_{doc_id}_{c['id']}_{options_key}",
                                label_visibility="collapsed",
                                format_func=lambda x: f"Merge into cluster {x} — {labels_by_id.get(x, '')}",
                            )
                        with merge_cols[1]:
                            if st.button(
                                "Merge", key=f"mergebtn_{doc_id}_{c['id']}",
                                icon=":material/call_merge:", width="stretch",
                            ):
                                merge_clusters(doc, c["id"], target)
                                st.session_state.current_cluster_id = target
                                st.rerun()

            # --- Right: interactive document view (only when toggled on) ---
            if right is not None:
                with right:
                    st.subheader("Document view")
                    st.caption(
                        "Click a mention to select it; click again to deselect. "
                        "Or click and drag across a region to select every mention it passes over."
                    )

                    doc_selected = mention_click_selector(
                        text, c["mentions"], key=f"docselect_{doc_id}_{c['id']}"
                    )
                    # Drop any selected mentions that no longer belong to this cluster
                    # (e.g. moved/removed elsewhere since the selection was made).
                    valid_spans = {(m["start"], m["end"]) for m in c["mentions"]}
                    doc_selected = doc_selected & valid_spans

                    if doc_selected:
                        st.caption(f"{len(doc_selected)} mention(s) selected in the document")
                        doc_bulk_options = ["Move selected to…"] + other_ids + [NEW_SINGLETON]
                        doc_bulk_options_key = "-".join(str(i) for i in other_ids)
                        doc_bulk_cols = st.columns([5, 2, 2])
                        with doc_bulk_cols[0]:
                            doc_bulk_target = st.selectbox(
                                "Move selected to (document view)",
                                doc_bulk_options,
                                key=f"docbulkmove_{doc_id}_{c['id']}_{doc_bulk_options_key}",
                                label_visibility="collapsed",
                                format_func=lambda x: (
                                    "Move selected to…" if x == "Move selected to…"
                                    else "Split selected into a new entity" if x == NEW_SINGLETON
                                    else f"Cluster {x} — {labels_by_id.get(x, '')}"
                                ),
                            )
                        with doc_bulk_cols[1]:
                            if st.button(
                                "Move selected", icon=":material/drive_file_move:", width="stretch",
                                key=f"docbulkmovebtn_{doc_id}_{c['id']}",
                                disabled=doc_bulk_target == "Move selected to…",
                            ):
                                move_mentions(doc, list(doc_selected), doc_bulk_target)
                                st.rerun()
                        with doc_bulk_cols[2]:
                            if st.button(
                                "Remove selected", icon=":material/delete:", width="stretch",
                                key=f"docbulkremovebtn_{doc_id}_{c['id']}",
                            ):
                                remove_mentions(doc, list(doc_selected))
                                st.rerun()

        st.divider()
        action_cols = st.columns([1, 1, 4])
        with action_cols[0]:
            if st.button("Reset corrections (this abstract)", icon=":material/restart_alt:", width="stretch"):
                reset_corrections(doc)
                st.rerun()
        with action_cols[1]:
            st.download_button(
                "Download session (JSON)",
                data=json.dumps(export_session(), indent=2),
                file_name="coref_session.json",
                mime="application/json",
                icon=":material/download:",
                width="stretch",
                help=(
                    "Saves every processed abstract. Reload it later with 'Resume a saved "
                    "session' in Run inference to pick up where you left off."
                ),
            )

    elif section == "View corrected & predicted clusters":
        original_labels_by_id = {c["id"]: cluster_label(c) for c in original_clusters}

        col_orig, col_corrected = st.columns(2)
        with col_orig:
            st.caption(f"Original (model output) — {len(original_clusters)} clusters")
            for oc in sorted(original_clusters, key=lambda c: c["size"], reverse=True):
                st.markdown(f"**{original_labels_by_id[oc['id']]}** · {oc['size']} mentions")
        with col_corrected:
            st.caption(f"Corrected (your edits) — {len(clusters)} clusters")
            for c2 in sorted(clusters, key=lambda c: c["size"], reverse=True):
                st.markdown(f"**{labels_by_id[c2['id']]}** · {c2['size']} mentions")

    elif section == "Metrics":
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Clusters", len(clusters))
        m2.metric("Largest cluster", max((c["size"] for c in clusters), default=0))
        m3.metric("Singletons", sum(1 for c in clusters if c["size"] == 1))
        m4.metric("Words", doc["static_metrics"]["doc_words"])

        v1, v2, v3, v4 = st.columns(4)
        v1.metric("Verified correct", verified_correct)
        v2.metric("Verified incorrect", verified_incorrect)
        v3.metric("Marked unsure", verified_unsure)
        v4.metric("Still unverified", verified_pending)

        has_edits = {(m["start"], m["end"], m["cluster_id"]) for m in doc["mentions"]} != {
            (m["start"], m["end"], m["cluster_id"]) for m in doc["original_mentions"]
        }
        st.caption(
            "Model accuracy scored against your corrections (mention-pair precision/recall), "
            "treating your corrected clusters as gold:"
        )
        if not has_edits:
            st.info("No corrections made yet — go to Correct and edit clusters to see a score here.")
        else:
            prf = compute_pairwise_prf(doc["original_mentions"], doc["mentions"])
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Precision", f'{prf["precision"]:.2f}')
            p2.metric("Recall", f'{prf["recall"]:.2f}')
            p3.metric("F1", f'{prf["f1"]:.2f}')
            p4.metric("Mentions removed", prf["num_removed_mentions"])
            if verified_pending:
                st.caption(
                    f"{verified_pending} of {len(clusters)} clusters are still unverified — "
                    "this score may shift as you keep reviewing."
                )

        with st.expander("Validate against gold clusters"):
            st.caption(
                "Paste in the mentions you know belong to one real entity (comma-separated). "
                "The app merges every corrected cluster that overlaps with your list and scores against that."
            )
            gold_col1, gold_col2 = st.columns([1, 2])
            with gold_col1:
                entity_name = st.text_input(
                    "Entity name (label only)", placeholder="e.g. Dumbledore", key=f"goldname_{doc_id}"
                )
            with gold_col2:
                gold_input = st.text_area(
                    "Gold mentions (comma-separated)",
                    placeholder="A man, this man, He, Albus Dumbledore, his, him",
                    height=80,
                    key=f"goldmentions_{doc_id}",
                )

            if st.button("Compute score", icon=":material/analytics:", key=f"goldscorebtn_{doc_id}"):
                if not gold_input.strip():
                    st.warning("Paste at least one gold mention first.")
                else:
                    gold_mentions = gold_input.split(",")
                    score = compute_gold_score(gold_mentions, clusters)

                    s1, s2, s3, s4 = st.columns(4)
                    s1.metric("Precision", f'{score["precision"]:.2f}')
                    s2.metric("Recall", f'{score["recall"]:.2f}')
                    s3.metric("F1", f'{score["f1"]:.2f}')
                    s4.metric("Fragments merged", score["num_fragments"])

                    if score["num_fragments"] > 1:
                        st.warning(
                            f'"{entity_name or "This entity"}" was split across '
                            f'{score["num_fragments"]} separate clusters '
                            f'({score["matched_cluster_ids"]}) instead of being unified into one.'
                        )

                    if score["missed"]:
                        st.error(
                            f'Missed entirely (in your gold list, not found in any matched cluster): '
                            f'{score["missed"]}'
                        )
                    if score["extra"]:
                        st.info(
                            f'Extra (included, not in your gold list — check these for false merges): '
                            f'{score["extra"]}'
                        )
