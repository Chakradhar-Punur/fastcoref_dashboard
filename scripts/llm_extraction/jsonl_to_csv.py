"""
Convert run_batch.py's output JSONL into the same CSV shape used for the
manual gold file (abstract_num, title, cluster_label, mentions, mention_count)
so the two can be diffed/compared directly, or loaded side-by-side with
fastcoref's output in the dashboard.
"""

from __future__ import annotations

import argparse
import csv
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    args = ap.parse_args()

    rows = []
    with open(args.in_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda r: r["abstract_num"])

    with open(args.out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["abstract_num", "title", "cluster_label", "mentions", "mention_count"])
        n_clusters = 0
        for r in rows:
            if not r["clusters"]:
                w.writerow([r["abstract_num"], r["title"], "(no multi-mention clusters found)", "", 0])
                continue
            for c in r["clusters"]:
                w.writerow([r["abstract_num"], r["title"], c["label"], "; ".join(c["mentions"]), len(c["mentions"])])
                n_clusters += 1

    print(f"Wrote {len(rows)} abstracts, {n_clusters} clusters -> {args.out_path}")


if __name__ == "__main__":
    main()
