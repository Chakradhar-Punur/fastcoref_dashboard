import streamlit as st

_HTML = '<div id="coref-mention-root"></div>'

_CSS = """
.coref-doc {
  max-height: 560px;
  overflow-y: auto;
  line-height: 1.6;
  font-size: 14px;
  white-space: pre-wrap;
  padding: 8px;
}
.coref-mention {
  background: #f7b955;
  color: #1a1300;
  border-radius: 3px;
  padding: 0 2px;
  cursor: pointer;
}
.coref-mention.selected {
  background: #2f6fed;
  color: #ffffff;
  box-shadow: 0 0 0 2px #2f6fed;
}
.coref-mention.dragging {
  outline: 2px dashed #2f6fed;
  outline-offset: 1px;
}
"""

_JS = """
export default function (component) {
  const { data, parentElement, setStateValue } = component
  const root = parentElement.querySelector("#coref-mention-root")
  if (!root) return

  const text = data?.text ?? ""
  const mentions = data?.mentions ?? []
  const selectedSet = new Set((data?.selected ?? []).map(([s, e]) => `${s}-${e}`))

  const escapeHtml = (s) =>
    s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")

  const spans = [...mentions].sort((a, b) => a[0] - b[0])

  let html = ""
  let cursor = 0
  for (const [start, end] of spans) {
    if (start < cursor) continue
    html += escapeHtml(text.slice(cursor, start))
    const spanKey = `${start}-${end}`
    const isSelected = selectedSet.has(spanKey)
    html += `<mark class="coref-mention${isSelected ? " selected" : ""}" data-start="${start}" data-end="${end}" tabindex="0" role="button" aria-pressed="${isSelected}">${escapeHtml(text.slice(start, end))}</mark>`
    cursor = end
  }
  html += escapeHtml(text.slice(cursor))

  // Reuse the same scrollable element across reruns instead of recreating it —
  // recreating it resets its scroll position to the top on every click.
  let doc = root.querySelector(".coref-doc")
  if (!doc) {
    doc = document.createElement("div")
    doc.className = "coref-doc"
    root.appendChild(doc)
  }
  const scrollTop = doc.scrollTop
  doc.innerHTML = html
  doc.scrollTop = scrollTop

  const emit = (nextSet) => {
    setStateValue("selected", Array.from(nextSet).map((k) => k.split("-").map(Number)))
  }

  const clearDragStyles = () => {
    doc.querySelectorAll(".coref-mention.dragging").forEach((el) => el.classList.remove("dragging"))
  }

  let isDragging = false
  let dragTouched = new Set()

  doc.querySelectorAll(".coref-mention").forEach((el) => {
    const spanKey = `${el.dataset.start}-${el.dataset.end}`
    el.onmouseenter = () => {
      if (isDragging) {
        dragTouched.add(spanKey)
        el.classList.add("dragging")
      }
    }
    el.onkeydown = (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault()
        const next = new Set(selectedSet)
        if (next.has(spanKey)) next.delete(spanKey)
        else next.add(spanKey)
        emit(next)
      }
    }
  })

  // Click-and-drag: press anywhere in the document, sweep across it, release —
  // every mention the pointer passed over gets added to the selection. A plain
  // click (the pointer never enters a second mention) still just toggles the one
  // mention pressed, same as before.
  doc.onmousedown = (e) => {
    isDragging = true
    dragTouched = new Set()
    const startMark = e.target.closest(".coref-mention")
    if (startMark) {
      const spanKey = `${startMark.dataset.start}-${startMark.dataset.end}`
      dragTouched.add(spanKey)
      startMark.classList.add("dragging")
    }
    e.preventDefault()
  }

  const onMouseUp = () => {
    if (!isDragging) return
    isDragging = false
    clearDragStyles()

    if (dragTouched.size === 0) {
      dragTouched = new Set()
      return
    }

    const next = new Set(selectedSet)
    if (dragTouched.size === 1) {
      const only = dragTouched.values().next().value
      if (next.has(only)) next.delete(only)
      else next.add(only)
    } else {
      dragTouched.forEach((k) => next.add(k))
    }
    emit(next)
    dragTouched = new Set()
  }

  // Attach to window (not just the doc container) so releasing outside it still
  // finalizes the drag. Remove any listener from a previous render first — the
  // element persists across reruns, so re-adding without removing would leak a
  // new stale listener (with a stale closure) on every single render.
  if (root._mouseUpHandler) {
    window.removeEventListener("mouseup", root._mouseUpHandler)
  }
  root._mouseUpHandler = onMouseUp
  window.addEventListener("mouseup", onMouseUp)
}
"""

_MENTION_SELECTOR = st.components.v2.component(
    "coref_mention_selector",
    html=_HTML,
    js=_JS,
    css=_CSS,
)


def mention_click_selector(text: str, mentions: list, *, key: str, initial_selected: set = frozenset()) -> set:
    """
    Renders the document with the given mentions highlighted and clickable. Clicking
    a mention toggles it in/out of the selection; returns the current selection as a
    set of (start, end) pairs. Persists its own selection across reruns under `key`,
    seeded from `initial_selected` only the first time this key is mounted.
    """
    component_state = st.session_state.get(key, {})
    selected_pairs = component_state.get("selected")
    if selected_pairs is None:
        selected_pairs = [[s, e] for (s, e) in initial_selected]

    result = _MENTION_SELECTOR(
        key=key,
        data={
            "text": text,
            "mentions": [[m["start"], m["end"]] for m in mentions],
            "selected": selected_pairs,
        },
        on_selected_change=lambda: None,
    )
    if result.selected is None:
        return {(pair[0], pair[1]) for pair in selected_pairs}
    return {(pair[0], pair[1]) for pair in result.selected}
