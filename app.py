"""
Coreference Resolution Dashboard — Streamlit version.

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""

import json

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from utils.constants import NEW_SINGLETON, STATUS_OPTIONS
from utils.clusters import cluster_label, derive_clusters
from utils.gold_compare import (
    find_flagged_mentions_in_clusters,
    find_missing_gold_entities,
    load_llm_gold_jsonl,
    match_clusters_to_gold,
)
from utils.inference import run_inference
from utils.mention_selector import mention_click_selector
from utils.rendering import mention_context
from utils.scoring import aggregate_pairwise_prf, compute_gold_score, compute_pairwise_prf
from utils.session import (
    add_mention,
    clear_status_widget_keys,
    cluster_status,
    export_finetuning_records,
    export_session,
    find_text_occurrences,
    load_session,
    merge_clusters,
    move_mentions,
    remove_documents,
    remove_mentions,
    reset_corrections,
)
from utils.text_processing import (
    clean_text,
    csv_row_to_document,
    extract_documents_from_url,
    extract_text,
    guess_csv_columns,
    load_csv,
)

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
st.session_state.setdefault("next_new_doc_id", 0)  # monotonic id counter, never reused
st.session_state.setdefault("csv_upload_key", None)  # detects a newly-picked CSV file
st.session_state.setdefault("csv_processed_count", 0)  # rows already batched in from the CSV
st.session_state.setdefault("csv_data", None)  # parsed CSV, kept independent of the uploader widget


def _doc_by_id(doc_id):
    return next((d for d in st.session_state.documents if d["id"] == doc_id), None)


def _doc_is_done(d):
    """An abstract counts as done once every cluster it has has been verified
    (given a status other than the default 'Unverified')."""
    cluster_ids = {m["cluster_id"] for m in d["mentions"]}
    if not cluster_ids:
        return True
    return all(d["cluster_statuses"].get(cid, "Unverified") != "Unverified" for cid in cluster_ids)


def _next_doc_id():
    doc_id = st.session_state.next_new_doc_id
    st.session_state.next_new_doc_id += 1
    return doc_id


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

    if not st.session_state.documents:
        st.caption("No abstracts processed yet — go to Run inference.")
    else:
        total_docs = len(st.session_state.documents)
        done_docs = sum(1 for d in st.session_state.documents if _doc_is_done(d))
        st.progress(done_docs / total_docs, text=f"{done_docs} of {total_docs} abstracts done")

        doc_ids = [d["id"] for d in st.session_state.documents]
        if st.session_state.current_doc_id not in doc_ids:
            st.session_state.current_doc_id = doc_ids[0]
        cursor = doc_ids.index(st.session_state.current_doc_id)

        nav_cols = st.columns(2)
        with nav_cols[0]:
            if st.button(
                "Previous", icon=":material/chevron_left:", width="stretch", disabled=cursor == 0
            ):
                st.session_state.current_doc_id = doc_ids[cursor - 1]
                st.session_state.current_cluster_id = None
                st.rerun()
        with nav_cols[1]:
            if st.button(
                "Next", icon=":material/chevron_right:", width="stretch",
                disabled=cursor == total_docs - 1,
            ):
                st.session_state.current_doc_id = doc_ids[cursor + 1]
                st.session_state.current_cluster_id = None
                st.rerun()

        st.caption(f"Abstract {cursor + 1} of {total_docs}")

        goto_cols = st.columns([3, 1])
        with goto_cols[0]:
            # Keyed on `cursor` so the box remounts (and re-applies value=) every
            # time the current abstract changes via Previous/Next/Go — otherwise
            # Streamlit would keep showing whatever number was last typed here.
            goto_number = st.number_input(
                "Go to abstract #", min_value=1, max_value=total_docs,
                value=cursor + 1, step=1, key=f"goto_abstract_{cursor}",
                label_visibility="collapsed",
            )
        with goto_cols[1]:
            if st.button("Go", key="goto_abstract_btn", width="stretch"):
                target_cursor = int(goto_number) - 1
                if target_cursor != cursor:
                    st.session_state.current_doc_id = doc_ids[target_cursor]
                    st.session_state.current_cluster_id = None
                st.rerun()


# --- Main area ---
st.title("Coreference resolution dashboard")

section = st.session_state.nav_section
doc = _doc_by_id(st.session_state.current_doc_id)

if section == "Run inference":
    st.caption(
        "Upload PDFs/.txt files, paste a web page URL, or bulk-load a CSV of abstracts. "
        "A page or CSV with several write-ups is split into one abstract per entry."
    )

    # st.segmented_control only honors default= the first time its key is set; a widget
    # that goes unrendered for a run (e.g. while nav_section is elsewhere) gets its key
    # dropped by Streamlit and re-mounts as "unset" next time, silently reverting to
    # default=. Mirroring the last choice into our own state and feeding it back as
    # default= keeps the tab you were on instead of snapping back to "Upload file".
    st.session_state.setdefault("run_input_mode", "Upload file")
    input_mode = st.segmented_control(
        "Input source", ["Upload file", "From URL", "From CSV"],
        key="input_mode", default=st.session_state.run_input_mode,
    )
    input_mode = input_mode or st.session_state.run_input_mode
    st.session_state.run_input_mode = input_mode

    csv_df, csv_title_col, csv_text_col, csv_batch_size = None, None, None, 0

    if input_mode == "From URL":
        url_input = st.text_input(
            "Web page URL",
            placeholder="https://example.com/abstracts",
            label_visibility="collapsed",
        )
        uploaded_files, csv_file = None, None

    elif input_mode == "From CSV":
        csv_file = st.file_uploader(
            "CSV file", type=["csv"], label_visibility="collapsed", key="csv_uploader"
        )
        uploaded_files, url_input = None, ""

        if csv_file is not None:
            file_identity = f"{csv_file.name}:{csv_file.size}"
            is_new_file = st.session_state.csv_upload_key != file_identity
            # Re-parse when it's a genuinely new file, OR when csv_data is missing even
            # though the identity matches — that happens right after loading a saved
            # session (which restores csv_upload_key/csv_processed_count but can't carry
            # the parsed DataFrame itself). Only reset the progress counter for an
            # actually-new file — matching identity means "resume", not "start over".
            if is_new_file or st.session_state.csv_data is None:
                if is_new_file:
                    st.session_state.csv_upload_key = file_identity
                    st.session_state.csv_processed_count = 0
                loaded_df = load_csv(csv_file)
                guessed_title, guessed_text = guess_csv_columns(loaded_df)
                # Stored independently of the file_uploader widget: that widget's own
                # value disappears whenever this "From CSV" branch goes unrendered for a
                # run (same reset behavior as input_mode above), which would otherwise
                # force re-uploading the whole file before every single batch.
                st.session_state.csv_data = {
                    "filename": csv_file.name,
                    "df": loaded_df,
                    "title_col": guessed_title,
                    "text_col": guessed_text,
                }

        if st.session_state.csv_data is not None:
            csv_info = st.session_state.csv_data
            csv_df = csv_info["df"]
            columns = list(csv_df.columns)
            st.caption(f"Using {csv_info['filename']} ({len(csv_df)} rows).")

            pick_cols = st.columns(2)
            with pick_cols[0]:
                csv_title_col = st.selectbox(
                    "Title column", columns,
                    index=columns.index(csv_info["title_col"]) if csv_info["title_col"] in columns else 0,
                    key="csv_title_col",
                )
            with pick_cols[1]:
                csv_text_col = st.selectbox(
                    "Abstract/text column", columns,
                    index=columns.index(csv_info["text_col"]) if csv_info["text_col"] in columns else 0,
                    key="csv_text_col",
                )
            csv_info["title_col"], csv_info["text_col"] = csv_title_col, csv_text_col

            total_rows = len(csv_df)
            processed = min(st.session_state.csv_processed_count, total_rows)
            remaining = total_rows - processed
            st.caption(f"{processed} of {total_rows} rows processed so far.")

            if remaining > 0:
                csv_batch_size = st.number_input(
                    "Rows to process in this run",
                    min_value=1, max_value=remaining, value=min(50, remaining),
                    key="csv_batch_size",
                    help="Process a manageable batch at a time instead of all rows at once "
                         "— each abstract takes real inference time, and clicking Run inference "
                         "again picks up right where this batch left off.",
                )
            else:
                st.success("Every row in this CSV has been processed.")

    else:
        uploaded_files = st.file_uploader(
            "Upload document(s)", type=["pdf", "txt"], accept_multiple_files=True,
            label_visibility="collapsed",
        )
        url_input, csv_file = "", None

    run_disabled = input_mode == "From CSV" and (csv_df is None or csv_batch_size == 0)
    run_label = "Process next batch" if input_mode == "From CSV" else "Run inference"
    run_clicked = st.button(
        run_label, type="primary", icon=":material/play_arrow:", width="stretch", disabled=run_disabled
    )

    if run_clicked:
        replace_documents = True

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

        elif input_mode == "From CSV":
            replace_documents = False
            start = st.session_state.csv_processed_count
            batch_df = csv_df.iloc[start:start + csv_batch_size]
            raw_docs = [
                # batch_df.iterrows() yields the row's original position in the full CSV
                # (unaffected by slicing), so idx + 1 is the same 1-based row number
                # run_batch.py's abstract_num uses for this same file.
                csv_row_to_document(row, csv_title_col, csv_text_col, row_num=idx + 1)
                for idx, row in batch_df.iterrows()
            ]
            # Advance past this batch even if every row in it turns out empty below,
            # so a run of blank rows can't get the user stuck retrying the same slice.
            st.session_state.csv_processed_count = start + len(batch_df)

        else:
            if not uploaded_files:
                st.warning("Please upload at least one file first.")
                st.stop()
            with st.spinner("Extracting text..."):
                raw_docs = [{"label": f.name, "text": extract_text(f)} for f in uploaded_files]

        cleaned_docs = [
            {"label": d["label"], "text": clean_text(d["text"]), "csv_row_num": d.get("csv_row_num")}
            for d in raw_docs
        ]
        cleaned_docs = [d for d in cleaned_docs if d["text"].strip()]

        if not cleaned_docs:
            if input_mode == "From CSV":
                st.warning("This batch had no usable text (empty abstracts) — skipped ahead. Run inference again for the next batch.")
                st.rerun()
            st.error(
                "No text could be extracted. If this was a PDF, it may be scanned/image-based — "
                "try a different file, OCR extraction, or a URL instead."
            )
            st.stop()

        new_documents = []
        progress = st.progress(0.0, text=f"Processing 0/{len(cleaned_docs)}...")
        for i, d in enumerate(cleaned_docs):
            mentions, num_clusters, static_metrics = run_inference(d["text"])
            new_documents.append({
                "id": _next_doc_id(),
                "label": d["label"],
                "text": d["text"],
                "mentions": mentions,
                "original_mentions": [dict(m) for m in mentions],
                "next_cluster_id": num_clusters,
                "original_next_cluster_id": num_clusters,
                "static_metrics": static_metrics,
                "cluster_statuses": {},
                "csv_row_num": d.get("csv_row_num"),
            })
            progress.progress(
                (i + 1) / len(cleaned_docs),
                text=f"Processing {i + 1}/{len(cleaned_docs)}: {d['label'][:60]}",
            )
        progress.empty()

        if replace_documents:
            st.session_state.documents = new_documents
        else:
            st.session_state.documents.extend(new_documents)

        st.session_state.current_doc_id = new_documents[0]["id"]
        st.session_state.current_cluster_id = None
        clear_status_widget_keys()
        st.session_state.nav_section = "Correct"
        st.success(f"Done. Processed {len(new_documents)} abstract(s).")
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

        with st.expander(":material/content_cut: Trim the working set"):
            st.caption(
                "Shrink how many abstracts you're on the hook for — e.g. you've corrected the "
                "first 130 of 300 and only want 20 more to reach 150, instead of grinding through "
                "all 300. This permanently drops the trimmed abstracts (and any corrections on "
                "them) from this session; it doesn't touch the source CSV or re-queue those rows."
            )
            total_docs = len(st.session_state.documents)
            keep_count = st.number_input(
                "Keep only the first N abstracts (by processing order)",
                min_value=1, max_value=total_docs, value=total_docs, key="trim_keep_count",
            )
            to_remove = st.session_state.documents[keep_count:]
            if not to_remove:
                st.caption("Nothing to remove at this count.")
            else:
                verified_among_removed = sum(
                    1 for d in to_remove
                    for c in derive_clusters(d["mentions"])
                    if cluster_status(d, c["id"]) != "Unverified"
                )
                st.caption(
                    f"Would remove {len(to_remove)} abstract(s) — positions {keep_count + 1} "
                    f"to {total_docs}."
                )
                if verified_among_removed:
                    st.warning(
                        f"{verified_among_removed} cluster(s) among these abstracts already have "
                        "a verification status — that work would be lost too."
                    )
                if st.button(
                    f"Remove {len(to_remove)} abstract(s)", icon=":material/delete_sweep:",
                    key="trim_remove_btn",
                ):
                    remove_documents([d["id"] for d in to_remove])
                    st.rerun()

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
        with st.expander(
            ":material/compare_arrows: Compare against LLM gold clusters (optional)",
            expanded=False,
        ):
            st.caption(
                "Load the JSONL produced by scripts/llm_extraction/run_batch.py. Abstracts "
                "imported 'From CSV' are matched by their exact CSV row number; abstracts from "
                "a PDF/URL upload (no row number) fall back to matching by title. Clusters whose "
                "mentions exactly match a gold entity can be bulk-marked Correct. Clusters "
                "containing a mention gold says should be a singleton or is author self-reference "
                "(we/us/our/I/my) — i.e. something that should never be grouped with anything — "
                "can be bulk-marked Incorrect so you can find and remove them. Everything else "
                "stays Unverified so you can filter to just what actually needs a look."
            )
            gold_file = st.file_uploader("LLM gold JSONL", type=["jsonl"], key="llm_gold_uploader")
            if gold_file is not None:
                try:
                    st.session_state.llm_gold = load_llm_gold_jsonl(gold_file.getvalue())
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    st.error(f"Couldn't parse that file: {e}")
            if st.session_state.get("llm_gold"):
                st.caption(f"Loaded gold data for {st.session_state.llm_gold['count']} abstract(s).")

        gold_store = st.session_state.get("llm_gold") or {}
        if doc.get("csv_row_num") is not None:
            gold_data = gold_store.get("by_num", {}).get(doc["csv_row_num"])
        else:
            gold_data = gold_store.get("by_title", {}).get(doc["label"].strip().lower())
        gold_entities = gold_data["clusters"] if gold_data else None
        gold_matches = {}
        flagged_by_cluster = {}
        if gold_data:
            gold_matches = match_clusters_to_gold(clusters, gold_entities)
            missing_gold = find_missing_gold_entities(clusters, gold_entities)
            flagged_by_cluster = find_flagged_mentions_in_clusters(
                clusters, gold_data["singleton_mentions"], gold_data["self_reference_mentions"]
            )
            n_exact = sum(1 for m in gold_matches.values() if m and m["f1"] >= 0.999)
            n_flagged_clusters = len(flagged_by_cluster)
            st.caption(
                f":material/compare_arrows: Gold check: {n_exact}/{len(clusters)} cluster(s) exactly "
                f"match a gold entity · {len(missing_gold)} gold entit(y/ies) fastcoref missed entirely "
                f"· {n_flagged_clusters} cluster(s) contain a mention that should never be grouped"
            )
            gold_cols = st.columns(2)
            with gold_cols[0]:
                if st.button(
                    "Auto-mark exact matches Correct", key=f"automarkgold_{doc_id}",
                    disabled=n_exact == 0, width="stretch",
                ):
                    for c in clusters:
                        m = gold_matches.get(c["id"])
                        if m and m["f1"] >= 0.999 and cluster_status(doc, c["id"]) == "Unverified":
                            doc["cluster_statuses"][c["id"]] = "Correct"
                    st.rerun()
            with gold_cols[1]:
                if st.button(
                    "Auto-mark flagged clusters Incorrect", key=f"automarkflagged_{doc_id}",
                    disabled=n_flagged_clusters == 0, width="stretch",
                ):
                    for cid in flagged_by_cluster:
                        if cluster_status(doc, cid) == "Unverified":
                            doc["cluster_statuses"][cid] = "Incorrect"
                    st.rerun()
            if missing_gold:
                with st.expander(f"{len(missing_gold)} gold entit(y/ies) fastcoref found nothing for"):
                    for g in missing_gold:
                        st.markdown(f"**{g['label']}** — {', '.join(g['mentions'])}")

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

        # Without expanded=True, the expander snaps shut on the very next rerun —
        # which happens as soon as you type into the text box inside it. Keying
        # openness off whether it already has text keeps it open once you start
        # using it, instead of collapsing after your first keystroke commits.
        add_text_key = f"addmention_text_{doc_id}"
        with st.expander(
            ":material/add_circle: Add a missed mention",
            expanded=bool(st.session_state.get(add_text_key, "").strip()),
        ):
            st.caption(
                "For an entity the model missed entirely — nothing is highlighted for it "
                "in the document, so there's no existing mention to click. Paste its exact "
                "wording from the abstract below, then assign it to a cluster."
            )
            add_text = st.text_input(
                "Mention text", key=add_text_key,
                placeholder='Copy the exact wording, e.g. "an orthogonality loss"',
            )
            add_text = add_text.strip()

            if add_text:
                occurrences = find_text_occurrences(text, add_text)
                if not occurrences:
                    st.warning("That exact text wasn't found in this abstract — check spelling/casing.")
                else:
                    if len(occurrences) == 1:
                        chosen_start, chosen_end = occurrences[0]
                    else:
                        occ_labels = [
                            f"…{text[max(0, s - 30):s]}[{text[s:e]}]{text[e:e + 30]}…"
                            for s, e in occurrences
                        ]
                        occ_choice = st.selectbox(
                            f"Found {len(occurrences)} occurrences — pick the right one",
                            range(len(occurrences)), format_func=lambda i: occ_labels[i],
                            key=f"addmention_occ_{doc_id}",
                        )
                        chosen_start, chosen_end = occurrences[occ_choice]

                    already_tracked = any(
                        m["start"] == chosen_start and m["end"] == chosen_end for m in doc["mentions"]
                    )
                    if already_tracked:
                        st.caption("This exact span is already tracked as a mention.")
                    else:
                        add_target = st.selectbox(
                            "Add to",
                            [NEW_SINGLETON] + [c["id"] for c in clusters],
                            format_func=lambda x: (
                                "Create a new cluster" if x == NEW_SINGLETON
                                else f"Cluster {x} — {labels_by_id.get(x, '')}"
                            ),
                            key=f"addmention_target_{doc_id}",
                        )
                        if st.button(
                            "Add mention", icon=":material/add:", width="stretch",
                            key=f"addmentionbtn_{doc_id}",
                        ):
                            add_mention(doc, chosen_start, chosen_end, add_text, add_target)
                            st.rerun()

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
                        if gold_entities:
                            gm = gold_matches.get(c["id"])
                            if gm is None:
                                st.caption(":material/help: no gold entity shares any mention with this cluster")
                            elif gm["f1"] >= 0.999:
                                st.caption(f":material/check_circle: exact match — gold entity **{gm['gold_label']}**")
                            else:
                                st.caption(
                                    f":material/warning: partial match (F1 {gm['f1']:.2f}) — closest gold "
                                    f"entity **{gm['gold_label']}** (P {gm['precision']:.2f} / R {gm['recall']:.2f})"
                                )
                                if gm["missed"]:
                                    missed_list = ", ".join(f'"{m}"' for m in gm["missed"])
                                    st.caption(
                                        f":material/search: still missing from this cluster: {missed_list} "
                                        "— paste one into 'Add a missed mention' below to add it here."
                                    )
                                if gm["extra"]:
                                    extra_list = ", ".join(f'"{m}"' for m in gm["extra"])
                                    st.caption(
                                        f":material/report: in this cluster but not in gold's entity "
                                        f"(check for a false merge): {extra_list}"
                                    )
                    with top_cols[1]:
                        current_status = cluster_status(doc, c["id"])
                        picked_status = st.segmented_control(
                            "Status",
                            STATUS_OPTIONS,
                            key=f"status_{doc_id}_{c['id']}",
                            default=current_status,
                            label_visibility="collapsed",
                        )
                        new_status = picked_status or current_status
                        if new_status != current_status:
                            doc["cluster_statuses"][c["id"]] = new_status
                            # The sidebar's "abstracts done" progress bar is rendered
                            # earlier in the script than this card, so without a rerun
                            # it would show last run's count instead of this change.
                            st.rerun()
                        doc["cluster_statuses"][c["id"]] = new_status

                    flagged_here = flagged_by_cluster.get(c["id"], [])
                    if flagged_here:
                        with st.container(border=True):
                            st.caption(
                                ":material/report: gold says these mention(s) should never be grouped "
                                "with anything — likely a false merge:"
                            )
                            for fm in flagged_here:
                                st.markdown(f"- “{fm['text']}” — {fm['reason']}")
                            if st.button(
                                f"Remove flagged mention(s) from this cluster",
                                icon=":material/delete_sweep:", width="stretch",
                                key=f"removeflagged_{doc_id}_{c['id']}",
                            ):
                                remove_mentions(doc, [(fm["start"], fm["end"]) for fm in flagged_here])
                                st.rerun()

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
                        # All candidates are shown as chips up front — no dropdown to open —
                        # so you can see every other cluster's entity before picking one,
                        # instead of reading them one at a time behind a select popup.
                        st.caption("Merge this cluster into:")
                        merge_target = st.pills(
                            "Merge into",
                            other_ids,
                            selection_mode="single",
                            format_func=lambda x: f"Cluster {x} — {labels_by_id.get(x, '')}",
                            key=f"mergepills_{doc_id}_{c['id']}",
                            label_visibility="collapsed",
                        )
                        if st.button(
                            "Merge", key=f"mergebtn_{doc_id}_{c['id']}",
                            icon=":material/call_merge:", width="stretch",
                            disabled=merge_target is None,
                        ):
                            merge_clusters(doc, c["id"], merge_target)
                            st.session_state.current_cluster_id = merge_target
                            st.rerun()

            # --- Right: interactive document view (only when toggled on) ---
            if right is not None:
                with right:
                    st.subheader("Document view")
                    st.caption(
                        "Click a mention to select it; click again to deselect. "
                        "Or click and drag across a region to select every mention it passes over. "
                        "Drag across plain text to select and copy it as usual."
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
        scope = st.segmented_control(
            "Scope", ["This abstract", "All abstracts"], key="compare_scope", default="This abstract"
        )

        if scope == "All abstracts":
            all_docs = st.session_state.documents
            all_original_counts = [len(derive_clusters(d["original_mentions"])) for d in all_docs]
            all_corrected_counts = [len(derive_clusters(d["mentions"])) for d in all_docs]

            m1, m2, m3 = st.columns(3)
            m1.metric("Abstracts processed", len(all_docs))
            m2.metric("Total original clusters", sum(all_original_counts))
            m3.metric(
                "Total corrected clusters", sum(all_corrected_counts),
                delta=sum(all_corrected_counts) - sum(all_original_counts),
            )

            status_totals = {s: 0 for s in STATUS_OPTIONS}
            for d in all_docs:
                for c in derive_clusters(d["mentions"]):
                    status_totals[cluster_status(d, c["id"])] += 1

            st.caption("Cluster verification status — every cluster across every processed abstract")
            status_df = pd.DataFrame({"status": list(status_totals), "count": list(status_totals.values())})
            st.bar_chart(status_df.set_index("status")["count"])

            st.caption("Per-abstract breakdown — click a column header to sort")
            rows = []
            for d in all_docs:
                oc = derive_clusters(d["original_mentions"])
                cc = derive_clusters(d["mentions"])
                rows.append({
                    "Abstract": d["label"],
                    "Original clusters": len(oc),
                    "Corrected clusters": len(cc),
                })
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

            st.divider()
            st.caption(
                "Export for fine-tuning — one JSON object per line (JSONL), one line per abstract: "
                "`{id, label, text, clusters}`, where clusters is a list of [start, end] mention "
                "spans per entity. Use the corrected file as your training/gold data; the original "
                "(model-predicted) file is for before/after evaluation, not for training on directly "
                "— it still has the errors you just fixed."
            )
            export_cols = st.columns(2)
            with export_cols[0]:
                corrected_jsonl = "\n".join(
                    json.dumps(r) for r in export_finetuning_records(use_corrected=True)
                )
                st.download_button(
                    "Download corrected clusters (JSONL)",
                    data=corrected_jsonl,
                    file_name="corrected_clusters.jsonl",
                    mime="application/jsonl",
                    icon=":material/download:",
                    width="stretch",
                )
            with export_cols[1]:
                original_jsonl = "\n".join(
                    json.dumps(r) for r in export_finetuning_records(use_corrected=False)
                )
                st.download_button(
                    "Download original (predicted) clusters (JSONL)",
                    data=original_jsonl,
                    file_name="original_clusters.jsonl",
                    mime="application/jsonl",
                    icon=":material/download:",
                    width="stretch",
                )

        else:
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
        metrics_scope = st.segmented_control(
            "Scope", ["This abstract", "All abstracts"], key="metrics_scope", default="This abstract"
        )

        if metrics_scope == "All abstracts":
            all_docs = st.session_state.documents
            all_clusters = [derive_clusters(d["mentions"]) for d in all_docs]

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Abstracts processed", len(all_docs))
            m2.metric("Total clusters", sum(len(cs) for cs in all_clusters))
            m3.metric("Total singletons", sum(1 for cs in all_clusters for c in cs if c["size"] == 1))
            m4.metric("Total words", sum(d["static_metrics"]["doc_words"] for d in all_docs))

            status_totals = {s: 0 for s in STATUS_OPTIONS}
            for d, cs in zip(all_docs, all_clusters):
                for c in cs:
                    status_totals[cluster_status(d, c["id"])] += 1

            v1, v2, v3, v4 = st.columns(4)
            v1.metric("Verified correct", status_totals["Correct"])
            v2.metric("Verified incorrect", status_totals["Incorrect"])
            v3.metric("Marked unsure", status_totals["Unsure"])
            v4.metric("Still unverified", status_totals["Unverified"])

            with st.expander("Clusters needing attention", icon=":material/flag:"):
                review_filter = st.segmented_control(
                    "Show", ["Singletons", "Incorrect", "Unsure"],
                    key="review_filter", default="Singletons",
                )
                review_filter = review_filter or "Singletons"

                flagged = []
                for d, cs in zip(all_docs, all_clusters):
                    for c in cs:
                        status = cluster_status(d, c["id"])
                        is_match = (
                            (review_filter == "Singletons" and c["size"] == 1)
                            or (review_filter == "Incorrect" and status == "Incorrect")
                            or (review_filter == "Unsure" and status == "Unsure")
                        )
                        if is_match:
                            flagged.append((d, c))

                if not flagged:
                    st.caption(f"No {review_filter.lower()} clusters found.")
                else:
                    st.caption(f"{len(flagged)} cluster(s) found.")
                    for d, c in flagged:
                        row_cols = st.columns([3, 5, 2])
                        with row_cols[0]:
                            st.caption(d["label"][:40])
                        with row_cols[1]:
                            st.markdown(f"Cluster {c['id']} — {cluster_label(c)}")
                        with row_cols[2]:
                            if st.button(
                                "Go to cluster", icon=":material/open_in_new:", width="stretch",
                                key=f"gotocluster_{d['id']}_{c['id']}",
                            ):
                                st.session_state.current_doc_id = d["id"]
                                st.session_state.current_cluster_id = c["id"]
                                st.session_state.nav_section = "Correct"
                                st.rerun()

            st.caption(
                "Model accuracy scored against your corrections, pooled across every abstract "
                "that has at least one edit (mention-pair precision/recall):"
            )
            edited_docs_prf = []
            per_abstract_metrics = []
            for d in all_docs:
                has_edits = {(m["start"], m["end"], m["cluster_id"]) for m in d["mentions"]} != {
                    (m["start"], m["end"], m["cluster_id"]) for m in d["original_mentions"]
                }
                if has_edits:
                    prf = compute_pairwise_prf(d["original_mentions"], d["mentions"])
                    edited_docs_prf.append(prf)
                    per_abstract_metrics.append({
                        "Abstract": d["label"][:25],
                        "Precision": prf["precision"],
                        "Recall": prf["recall"],
                        "F1": prf["f1"],
                    })

            if not edited_docs_prf:
                st.info("No corrections made yet on any abstract — edit clusters in Correct to see a score here.")
            else:
                agg_prf = aggregate_pairwise_prf(edited_docs_prf)
                p1, p2, p3, p4 = st.columns(4)
                p1.metric("Precision", f'{agg_prf["precision"]:.2f}')
                p2.metric("Recall", f'{agg_prf["recall"]:.2f}')
                p3.metric("F1", f'{agg_prf["f1"]:.2f}')
                p4.metric("Mentions removed", agg_prf["num_removed_mentions"])
                st.caption(
                    f"Pooled from {len(edited_docs_prf)} of {len(all_docs)} abstracts that have edits so far."
                )
                if status_totals["Unverified"]:
                    st.caption(
                        f"{status_totals['Unverified']} clusters are still unverified across all abstracts — "
                        "this score may shift as you keep reviewing."
                    )

                metrics_df = pd.DataFrame(per_abstract_metrics)
                n_edited = len(metrics_df)
                window = 10  # abstracts visible at once before you have to scroll

                st.caption(
                    "Metric trends across abstracts with edits"
                    + (
                        f" — showing {min(window, n_edited)} of {n_edited} at a time; "
                        "drag the range slider below the chart (or its handles) to scroll left/right."
                        if n_edited > window
                        else ""
                    )
                )

                # Plot against a numeric x-axis (with abstract names as tick labels)
                # rather than the labels themselves, so the range slider works and
                # doesn't need to squeeze every label into view at once.
                x_vals = list(range(n_edited))
                fig = go.Figure()
                for col in ["Precision", "Recall", "F1"]:
                    fig.add_trace(go.Scatter(
                        x=x_vals,
                        y=metrics_df[col],
                        mode="lines+markers",
                        name=col,
                        line=dict(width=3),
                        marker=dict(size=8)
                    ))

                fig.update_xaxes(
                    tickmode="array",
                    tickvals=x_vals,
                    ticktext=metrics_df["Abstract"],
                    rangeslider=dict(visible=True),
                    range=[-0.5, min(window, n_edited) - 0.5],
                )
                fig.update_layout(
                    title="",
                    xaxis_title="Abstract",
                    yaxis_title="Score",
                    height=560,
                    hovermode="x unified",
                    template="plotly_dark" if st.get_option("theme.base") == "dark" else "plotly",
                )
                st.plotly_chart(fig, width="stretch")

        else:
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
