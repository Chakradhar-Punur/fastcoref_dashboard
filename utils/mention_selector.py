import os

import streamlit as st
import streamlit.components.v1 as components

# `st.components.v2` does not exist in any released Streamlit version (1.50.0
# here, and there is no v2 on PyPI) — it was only ever a comment in
# Streamlit's own source about a *future* cleanup. The real, released API for
# custom components is `streamlit.components.v1.declare_component`, which
# serves a static frontend directory and talks to it over the standard
# postMessage protocol. The frontend lives in ./mention_selector_frontend and
# implements that protocol by hand (no npm build step).
_FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "mention_selector_frontend")

_MENTION_SELECTOR = components.declare_component(
    "coref_mention_selector",
    path=_FRONTEND_DIR,
)


def mention_click_selector(text: str, mentions: list, *, key: str, initial_selected: set = frozenset()) -> set:
    """
    Renders the document with the given mentions highlighted and clickable. Clicking
    a mention toggles it in/out of the selection; returns the current selection as a
    set of (start, end) pairs. Persists its own selection across reruns under `key`,
    seeded from `initial_selected` only the first time this key is mounted.
    """
    stored = st.session_state.get(key)
    if stored is not None:
        selected_pairs = [[s, e] for (s, e) in stored]
    else:
        selected_pairs = [[s, e] for (s, e) in initial_selected]

    result = _MENTION_SELECTOR(
        text=text,
        mentions=[[m["start"], m["end"]] for m in mentions],
        selected=selected_pairs,
        key=key,
        default=selected_pairs,
    )
    return {(pair[0], pair[1]) for pair in result}
