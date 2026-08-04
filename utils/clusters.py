def derive_clusters(mentions: list) -> list:
    """Group the flat mention list (the single source of truth) back into clusters for display."""
    by_id = {}
    for m in mentions:
        by_id.setdefault(m["cluster_id"], []).append(m)

    clusters = []
    for cid, ms in by_id.items():
        clusters.append({
            "id": cid,
            "mentions": sorted(ms, key=lambda m: m["start"]),
            "size": len(ms),
        })
    return clusters


def cluster_label(cluster: dict) -> str:
    """A human-readable stand-in for a cluster id, e.g. its longest mention ('Sarah' over 'her')."""
    if not cluster["mentions"]:
        return ""
    return max(cluster["mentions"], key=lambda m: len(m["text"]))["text"]
