import io
import re
from collections import Counter

import pandas as pd
import pdfplumber
import requests
from bs4 import BeautifulSoup

_USER_AGENT = (
    "Mozilla/5.0 (compatible; CorefDashboard/1.0; "
    "+https://github.com/) requests"
)
_HEADING_TAGS = ["h1", "h2", "h3", "h4", "h5", "h6"]


def extract_text(uploaded_file) -> str:
    if uploaded_file.name.lower().endswith(".pdf"):
        text = ""
        with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    return uploaded_file.read().decode("utf-8", errors="ignore")


def _fetch_main_content(url: str, timeout: int):
    response = requests.get(url, timeout=timeout, headers={"User-Agent": _USER_AGENT})
    response.raise_for_status()

    soup = BeautifulSoup(response.content, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form", "noscript"]):
        tag.decompose()

    main = soup.find("article") or soup.find("main") or soup.body or soup
    page_title = soup.title.get_text(strip=True) if soup.title else "Document"
    return main, page_title


def extract_text_from_url(url: str, timeout: int = 15) -> str:
    """Fetch a web page and pull out its main readable text, stripping nav/header/
    footer chrome and script/style content."""
    main, _ = _fetch_main_content(url, timeout)
    return main.get_text(separator="\n")


def extract_documents_from_url(url: str, timeout: int = 15) -> list:
    """Fetch a web page and split it into one document per entry, when the page reads
    like a list of separate write-ups (e.g. a page of sample abstracts). Splits at
    whichever heading level repeats most on the page — the level most likely to be
    "one heading per entry" rather than an incidental subheading — and falls back to
    a single document (the whole page) when no such repeating heading is found."""
    main, page_title = _fetch_main_content(url, timeout)

    headings = main.find_all(_HEADING_TAGS)
    if not headings:
        return [{"label": page_title, "text": main.get_text(separator="\n")}]

    level_counts = Counter(h.name for h in headings)
    split_level = level_counts.most_common(1)[0][0]
    split_headings = [h for h in headings if h.name == split_level]

    if len(split_headings) < 2:
        return [{"label": page_title, "text": main.get_text(separator="\n")}]

    split_heading_ids = {id(h) for h in split_headings}
    documents = []
    for i, heading in enumerate(split_headings):
        title = heading.get_text(strip=True)
        parts = []
        node = heading.next_sibling
        while node is not None and id(node) not in split_heading_ids:
            if hasattr(node, "get_text"):
                part = node.get_text(separator="\n")
            else:
                part = str(node)
            if part.strip():
                parts.append(part)
            node = node.next_sibling
        body = "\n".join(parts)
        documents.append({
            "label": title or f"Abstract {i + 1}",
            "text": f"{title}\n{body}" if title else body,
        })

    return documents


def load_csv(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)


def guess_csv_columns(df: pd.DataFrame):
    """Best-effort guess of (title_col, text_col) from common column names."""
    cols_lower = {c.lower(): c for c in df.columns}
    title_col = next((cols_lower[c] for c in ("title", "headline", "name") if c in cols_lower), None)
    text_col = next(
        (cols_lower[c] for c in ("abstract", "text", "body", "content", "summary") if c in cols_lower),
        None,
    )
    return title_col, text_col


def csv_row_to_document(row, title_col, text_col, row_num=None) -> dict:
    title = str(row[title_col]).strip() if title_col and pd.notna(row[title_col]) else ""
    body = str(row[text_col]).strip() if text_col and pd.notna(row[text_col]) else ""
    text = f"{title}\n{body}" if title else body
    label = title or (body[:70] + "…" if len(body) > 70 else body) or "Untitled abstract"
    # 1-based row number within the source CSV, when known — lets gold-comparison match
    # this abstract exactly rather than by title, which can collide on duplicate titles.
    return {"label": label, "text": text, "csv_row_num": row_num}


def clean_text(text: str) -> str:
    text = re.sub(r"-\n", "", text)
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
