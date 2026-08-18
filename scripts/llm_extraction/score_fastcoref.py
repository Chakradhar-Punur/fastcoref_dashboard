"""
Headless comparison: run fastcoref (locally, no API key needed) on a range of
abstracts and score it against the LLM-generated gold data — using this
project's own utils.scoring / utils.gold_compare, so the numbers mean
exactly the same thing as the dashboard's gold-comparison features.

Produces two reports:
  1. Entity coverage (--out): for each gold entity, whether fastcoref found
     it at all, and if so how precisely/completely.
  2. False merges (--flagged-out): fastcoref clusters that contain a mention
     gold says should never be grouped with anything (a singleton entity, or
     author self-reference like we/us/our/I/my) — likely fastcoref errors.

Usage:
    python score_fastcoref.py \
        --csv "/path/to/ai_abstracts.csv" \
        --llm-gold clusters_llm_300.jsonl \
        --start-row 1 --end-row 300 \
        --out fastcoref_score_report.csv \
        --flagged-out fastcoref_flagged_mentions.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.clusters import derive_clusters
from utils.gold_compare import find_flagged_mentions_in_clusters
from utils.inference import run_inference
from utils.scoring import compute_gold_score
from utils.text_processing import clean_text


def load_llm_gold(path: Path) -> dict:
    """{abstract_num: {"clusters", "singleton_mentions", "self_reference_mentions"}}.
    Tolerant of older JSONL rows written before the latter two fields existed."""
    gold = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            gold[row["abstract_num"]] = {
                "clusters": row.get("clusters", []),
                "singleton_mentions": row.get("singleton_mentions", []),
                "self_reference_mentions": row.get("self_reference_mentions", []),
            }
    return gold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--llm-gold", required=True, help="JSONL from run_batch.py")
    ap.add_argument("--start-row", type=int, default=1)
    ap.add_argument("--end-row", type=int, default=300)
    ap.add_argument("--out", required=True, help="Entity coverage report CSV")
    ap.add_argument(
        "--flagged-out", default=None,
        help="False-merge report CSV (default: <out>, with _flagged appended before the extension)",
    )
    args = ap.parse_args()

    flagged_out = args.flagged_out
    if flagged_out is None:
        out_path = Path(args.out)
        flagged_out = str(out_path.with_name(out_path.stem + "_flagged" + out_path.suffix))

    gold_by_abstract = load_llm_gold(Path(args.llm_gold))

    report_rows = []
    flagged_rows = []
    n_entities = 0
    n_entities_missed_entirely = 0  # fastcoref found zero overlap at all
    sum_precision = 0.0
    sum_recall = 0.0
    total_gold_mentions = 0
    total_true_positives = 0
    total_predicted_mentions = 0

    with open(args.csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_num, row in enumerate(reader, start=1):
            if row_num < args.start_row:
                continue
            if row_num > args.end_row:
                break

            gold = gold_by_abstract.get(row_num)
            if gold is None:
                print(f"  [{row_num}] SKIPPED — no LLM gold data for this abstract yet", file=sys.stderr)
                continue
            gold_clusters = gold["clusters"]

            title = str(row["title"]).strip()
            body = str(row["abstract"]).strip()
            text = f"{title}\n{body}" if title else body  # matches csv_row_to_document exactly
            text = clean_text(text)  # app.py always cleans before run_inference — match that here too

            mentions, num_clusters, _ = run_inference(text)
            fc_clusters = derive_clusters(mentions)

            flagged_by_cluster = find_flagged_mentions_in_clusters(
                fc_clusters, gold["singleton_mentions"], gold["self_reference_mentions"]
            )
            for cid, flagged in flagged_by_cluster.items():
                cluster_mentions = next(c["mentions"] for c in fc_clusters if c["id"] == cid)
                cluster_label_text = max(cluster_mentions, key=lambda m: len(m["text"]))["text"]
                for fm in flagged:
                    flagged_rows.append(
                        {
                            "abstract_num": row_num,
                            "title": title,
                            "fastcoref_cluster_label": cluster_label_text,
                            "cluster_size": len(cluster_mentions),
                            "flagged_mention": fm["text"],
                            "reason": fm["reason"],
                        }
                    )

            if not gold_clusters:
                print(
                    f"  [{row_num}] (no gold entities to score — fastcoref found {num_clusters} cluster(s), "
                    f"{len(flagged_by_cluster)} flagged)"
                )
                continue

            for g in gold_clusters:
                score = compute_gold_score(g["mentions"], fc_clusters)
                found = score["num_fragments"] > 0
                n_entities += 1
                sum_precision += score["precision"]
                sum_recall += score["recall"]
                if not found:
                    n_entities_missed_entirely += 1

                gold_set_size = len({m.strip().lower() for m in g["mentions"] if m.strip()})
                predicted_set_size = gold_set_size - len(score["missed"]) + len(score["extra"])
                total_gold_mentions += gold_set_size
                total_true_positives += gold_set_size - len(score["missed"])
                total_predicted_mentions += predicted_set_size

                report_rows.append(
                    {
                        "abstract_num": row_num,
                        "title": title,
                        "gold_entity": g["label"],
                        "gold_mentions": "; ".join(g["mentions"]),
                        "found_by_fastcoref": found,
                        "precision": round(score["precision"], 3),
                        "recall": round(score["recall"], 3),
                        "f1": round(score["f1"], 3),
                        "missed_by_fastcoref": "; ".join(score["missed"]),
                        "extra_in_fastcoref_cluster": "; ".join(score["extra"]),
                    }
                )

            print(
                f"  [{row_num}] {len(gold_clusters)} gold entit(y/ies) scored, "
                f"{len(flagged_by_cluster)} flagged cluster(s) — {title[:60]}"
            )

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(report_rows[0].keys()) if report_rows else [])
        if report_rows:
            w.writeheader()
            w.writerows(report_rows)
    print(f"\nWrote {len(report_rows)} scored entities -> {args.out}")

    with open(flagged_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(flagged_rows[0].keys()) if flagged_rows else [])
        if flagged_rows:
            w.writeheader()
            w.writerows(flagged_rows)
    print(f"Wrote {len(flagged_rows)} flagged (likely false-merge) mention(s) -> {flagged_out}")

    if n_entities:
        print(f"\n=== Summary across {n_entities} gold entities ===")
        print(f"Entities fastcoref missed completely (0 overlap): {n_entities_missed_entirely} "
              f"({n_entities_missed_entirely / n_entities:.1%})")
        print(f"Mean per-entity precision: {sum_precision / n_entities:.3f}")
        print(f"Mean per-entity recall:    {sum_recall / n_entities:.3f}")
        if total_predicted_mentions and total_gold_mentions:
            micro_p = total_true_positives / total_predicted_mentions
            micro_r = total_true_positives / total_gold_mentions
            micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) else 0.0
            print(f"Micro-averaged (pooled)  precision/recall/F1: {micro_p:.3f} / {micro_r:.3f} / {micro_f1:.3f}")
    if flagged_rows:
        print(f"False merges flagged: {len(flagged_rows)} mention(s) across "
              f"{len({(r['abstract_num'], r['fastcoref_cluster_label']) for r in flagged_rows})} cluster(s)")


if __name__ == "__main__":
    main()
