# Coreference Resolution Dashboard

A Streamlit app for running coreference resolution on documents and manually correcting the results — built for verifying and fixing model output at scale (single files, web pages, or CSVs of hundreds/thousands of abstracts).

## What it does

- **Run inference** on PDFs, TXT files, a web page URL (auto-split into one document per entry), or a bulk CSV of abstracts, using [FCoref](https://github.com/shon-otmazgin/fastcoref).
- **Correct** the model's clusters by hand: verify each cluster as Correct/Incorrect/Unsure, move or remove mentions individually or in bulk, merge two clusters together, or select mentions directly in the rendered document (click, or click-and-drag across a region).
- **Compare** original (model) vs. corrected clusters, per abstract or aggregated across everything processed so far.
- **Score accuracy** with mention-pair precision/recall against your corrections, optionally validated against a hand-typed gold mention list — again either per abstract or pooled across the whole batch.
- **Resume later**: download a session as JSON and reload it to pick up corrections without re-running inference.

## Setup

Requires Python 3.9+.

```bash
git clone https://github.com/Chakradhar-Punur/fastcoref_dashboard.git
cd fastcoref_dashboard
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. On macOS with Safari, use `http://127.0.0.1:8501` instead — Safari has a known issue with Streamlit's WebSocket connection over `localhost`'s IPv6 loopback.

## Project structure

```
app.py                      # Streamlit UI — navigation, all four sections
utils/
  constants.py               # Shared constants (status options, sentinels)
  clusters.py                 # Group mentions into clusters, derive labels
  inference.py                 # FCoref model loading + inference
  session.py                    # Cluster edits (move/merge/remove), session save/load
  text_processing.py             # PDF/URL/CSV parsing into documents
  rendering.py                    # Full-sentence context around each mention
  scoring.py                       # Precision/recall/F1 scoring, incl. multi-doc pooling
  mention_selector.py               # Custom component: clickable/drag-selectable document view
requirements.txt
```

## Notes

- Inference runs on CPU unless an Apple Silicon GPU (MPS) is available; budget roughly 0.25s per abstract for planning large CSV batches.
- CSV mode processes in configurable batches (default 50 rows) so large files don't block the UI in one long run — click "Process next batch" repeatedly to work through the whole file.
