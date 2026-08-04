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


def compute_pairwise_prf(original_mentions: list, corrected_mentions: list):
    """
    Score the original model output against the reviewer-corrected clusters,
    treating the corrections as gold. Uses mention-pair precision/recall:
    a "link" is a pair of mentions placed in the same cluster.

    Only mentions that survived correction (weren't removed) are scored,
    since a removed mention has no gold cluster to compare against.
    """
    original_cluster_of = {(m["start"], m["end"]): m["cluster_id"] for m in original_mentions}
    corrected_cluster_of = {(m["start"], m["end"]): m["cluster_id"] for m in corrected_mentions}
    surviving_spans = sorted(corrected_cluster_of.keys())

    def pairs_for(cluster_of):
        by_cluster = {}
        for span in surviving_spans:
            by_cluster.setdefault(cluster_of[span], []).append(span)
        pairs = set()
        for spans in by_cluster.values():
            for i in range(len(spans)):
                for j in range(i + 1, len(spans)):
                    pairs.add(frozenset((spans[i], spans[j])))
        return pairs

    original_pairs = pairs_for(original_cluster_of)
    corrected_pairs = pairs_for(corrected_cluster_of)
    overlap = original_pairs & corrected_pairs

    precision = len(overlap) / len(original_pairs) if original_pairs else 1.0
    recall = len(overlap) / len(corrected_pairs) if corrected_pairs else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "num_removed_mentions": len(original_mentions) - len(corrected_mentions),
    }
