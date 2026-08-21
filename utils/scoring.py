def clusters_to_mentions(clusters: list) -> list:
    """[[[start, end], ...], ...] (export_finetuning_records' per-cluster span
    shape, in utils/session.py) -> the flat {start, end, cluster_id} list
    compute_pairwise_prf expects. Shared by the Metrics tab's model-comparison
    tool and scripts/finetune/compare_models.py so the two never drift."""
    return [
        {"start": start, "end": end, "cluster_id": cluster_id}
        for cluster_id, spans in enumerate(clusters)
        for start, end in spans
    ]


def compute_gold_score(gold_mentions: list, clusters: list):
    """
    Compare a hand-labeled gold mention list for one entity against the
    corrected clusters.
    """
    gold_set = {m.strip().lower() for m in gold_mentions if m.strip()}

    matched_cluster_ids = []
    predicted_set = set()

    for c in clusters:
        cluster_mentions_lower = {m["text"].strip().lower() for m in c["mentions"]}
        if cluster_mentions_lower & gold_set:  # any overlap at all
            matched_cluster_ids.append(c["id"])
            predicted_set |= cluster_mentions_lower

    true_positives = predicted_set & gold_set
    precision = len(true_positives) / len(predicted_set) if predicted_set else 0.0
    recall = len(true_positives) / len(gold_set) if gold_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    missed = gold_set - predicted_set        # mentions you listed that were never caught at all
    extra = predicted_set - gold_set         # mentions included that weren't in your gold list

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matched_cluster_ids": matched_cluster_ids,
        "num_fragments": len(matched_cluster_ids),
        "missed": sorted(missed),
        "extra": sorted(extra),
    }


def aggregate_pairwise_prf(prf_results: list):
    """Pool several compute_pairwise_prf() results (one per document) into a single
    micro-averaged score, rather than just averaging their precision/recall ratios —
    a document with few clusters shouldn't count as much as one with many."""
    total_overlap = sum(r["num_overlap_pairs"] for r in prf_results)
    total_original = sum(r["num_original_pairs"] for r in prf_results)
    total_corrected = sum(r["num_corrected_pairs"] for r in prf_results)
    total_removed = sum(r["num_removed_mentions"] for r in prf_results)

    precision = total_overlap / total_original if total_original else 1.0
    recall = total_overlap / total_corrected if total_corrected else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "num_removed_mentions": total_removed,
    }


def compute_pairwise_prf(original_mentions: list, corrected_mentions: list):
    """
    Score the original model output against the reviewer-corrected clusters,
    treating the corrections as gold. Uses mention-pair precision/recall:
    a "link" is a pair of mentions placed in the same cluster.

    A mention added during correction (e.g. via "Add a missed mention") has no
    counterpart in the original model output at all, so it can't contribute to
    "original" pairs — scoring precision only makes sense over spans the model
    actually produced. It still counts on the "corrected" side, which correctly
    penalizes recall for whatever the model missed entirely.
    """
    original_cluster_of = {(m["start"], m["end"]): m["cluster_id"] for m in original_mentions}
    corrected_cluster_of = {(m["start"], m["end"]): m["cluster_id"] for m in corrected_mentions}

    surviving_original_spans = sorted(s for s in corrected_cluster_of if s in original_cluster_of)
    corrected_spans = sorted(corrected_cluster_of.keys())

    def pairs_for(cluster_of, spans):
        by_cluster = {}
        for span in spans:
            by_cluster.setdefault(cluster_of[span], []).append(span)
        pairs = set()
        for spans_in_cluster in by_cluster.values():
            for i in range(len(spans_in_cluster)):
                for j in range(i + 1, len(spans_in_cluster)):
                    pairs.add(frozenset((spans_in_cluster[i], spans_in_cluster[j])))
        return pairs

    original_pairs = pairs_for(original_cluster_of, surviving_original_spans)
    corrected_pairs = pairs_for(corrected_cluster_of, corrected_spans)
    overlap = original_pairs & corrected_pairs

    precision = len(overlap) / len(original_pairs) if original_pairs else 1.0
    recall = len(overlap) / len(corrected_pairs) if corrected_pairs else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    original_spans_set = set(original_cluster_of.keys())
    corrected_spans_set = set(corrected_cluster_of.keys())

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "num_removed_mentions": len(original_spans_set - corrected_spans_set),
        "num_added_mentions": len(corrected_spans_set - original_spans_set),
        # Raw counts, so scores from multiple documents can be pooled into one
        # proper (micro-averaged) score instead of just averaging ratios.
        "num_overlap_pairs": len(overlap),
        "num_original_pairs": len(original_pairs),
        "num_corrected_pairs": len(corrected_pairs),
    }
