"""
Compare fastcoref's clusters for one abstract against LLM-generated gold
data (from scripts/llm_extraction/run_batch.py's JSONL output), so the
Correct page can auto-flag matches and false merges instead of the reviewer
eyeballing a separate CSV by hand.

Gold data per abstract has three parts (see scripts/llm_extraction/extract_clusters.py):
  - clusters: real multi-mention entities — used to auto-detect exact matches.
  - singleton_mentions / self_reference_mentions: things that should NEVER
    end up grouped with anything else — used to catch fastcoref false merges.

Auto-marking is deliberately conservative: only a clean, exact mention-set
match (F1 == 1.0) is treated as strong enough to auto-mark a cluster
"Correct" — anything short of that is left "Unverified" so a human still
looks at it. A cluster containing a flagged singleton/self-reference mention
is auto-marked "Incorrect" instead, since gold says that mention shouldn't
be linked to anything at all.
"""

import json


def load_llm_gold_jsonl(raw_bytes: bytes) -> dict:
    """Parse run_batch.py's output JSONL into two lookup indexes, so the caller can match
    on whichever key the abstract actually has:

      - "by_num": {abstract_num: {...}} — the source CSV's 1-based row number. Exact and
        collision-free; use this whenever the abstract was imported via 'From CSV' (which
        tracks its row number — see csv_row_num on the document).
      - "by_title": {title_lower: {...}} — fallback for abstracts with no known row number
        (PDF/URL uploads). Can silently collide if the source CSV has duplicate titles,
        which "by_num" doesn't have — prefer "by_num" whenever it's available.

    Each entry is {"clusters": [...], "singleton_mentions": [...], "self_reference_mentions": [...]}.
    Tolerant of older JSONL files written before singleton_mentions/self_reference_mentions
    existed — those abstracts just get empty lists for the new fields."""
    by_num = {}
    by_title = {}
    count = 0
    for line in raw_bytes.decode("utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        count += 1
        entry = {
            "clusters": row.get("clusters", []),
            "singleton_mentions": row.get("singleton_mentions", []),
            "self_reference_mentions": row.get("self_reference_mentions", []),
        }
        if "abstract_num" in row:
            by_num[row["abstract_num"]] = entry
        title = (row.get("title") or "").strip().lower()
        if title:
            by_title[title] = entry
    return {"by_num": by_num, "by_title": by_title, "count": count}


def match_clusters_to_gold(fc_clusters: list, gold_entities: list) -> dict:
    """For each fastcoref cluster, find the gold entity it overlaps with most —
    by mention-text overlap, case-insensitive — and score that match.

    Returns {cluster_id: None | {"gold_label", "precision", "recall", "f1"}}.
    None means no gold entity shares even one mention with this cluster.
    """
    gold_sets = [
        ({m.strip().lower() for m in g["mentions"] if m.strip()}, g["label"])
        for g in gold_entities
        if g.get("mentions")
    ]

    result = {}
    for c in fc_clusters:
        c_mentions_lower = {m["text"].strip().lower() for m in c["mentions"]}
        best = None
        for gold_set, label in gold_sets:
            overlap = c_mentions_lower & gold_set
            if not overlap:
                continue
            precision = len(overlap) / len(c_mentions_lower) if c_mentions_lower else 0.0
            recall = len(overlap) / len(gold_set) if gold_set else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
            if best is None or f1 > best["f1"]:
                best = {"gold_label": label, "precision": precision, "recall": recall, "f1": f1}
        result[c["id"]] = best
    return result


def find_missing_gold_entities(fc_clusters: list, gold_entities: list) -> list:
    """Gold entities that share zero mentions with any fastcoref cluster —
    i.e. fastcoref found nothing at all for this entity. These won't show up
    as a mismatched cluster (there's no cluster to flag) — surface them
    separately so the reviewer knows to use 'Add a missed mention'."""
    all_fc_mentions_lower = {
        m["text"].strip().lower() for c in fc_clusters for m in c["mentions"]
    }
    missing = []
    for g in gold_entities:
        gold_set = {m.strip().lower() for m in g.get("mentions", []) if m.strip()}
        if gold_set and not (gold_set & all_fc_mentions_lower):
            missing.append(g)
    return missing


def find_flagged_mentions_in_clusters(
    fc_clusters: list, singleton_mentions: list, self_reference_mentions: list
) -> dict:
    """Mentions inside fastcoref's clusters that gold says should NEVER be
    grouped with anything — a singleton entity (mentioned exactly once, so
    nothing else in the text corefers with it) or author self-reference
    (we/us/our/I/my), which fastcoref may have merged into a real entity's
    cluster instead of keeping separate.

    Returns {cluster_id: [{"start", "end", "text", "reason"}, ...]} for
    clusters containing at least one such mention. `reason` is "singleton"
    or "self-reference".
    """
    singleton_set = {m.strip().lower() for m in singleton_mentions if m.strip()}
    self_ref_set = {m.strip().lower() for m in self_reference_mentions if m.strip()}

    result = {}
    for c in fc_clusters:
        flagged = []
        for m in c["mentions"]:
            t = m["text"].strip().lower()
            if t in singleton_set:
                flagged.append({"start": m["start"], "end": m["end"], "text": m["text"], "reason": "singleton"})
            elif t in self_ref_set:
                flagged.append(
                    {"start": m["start"], "end": m["end"], "text": m["text"], "reason": "self-reference"}
                )
        if flagged:
            result[c["id"]] = flagged
    return result
