"""
Compares the base model against the fine-tuned model on one batch of
abstracts, in either of two modes:

  - WITH --gold: scores both models' clusters against your corrections —
    tells you which one is more ACCURATE. Needs the abstracts to actually
    be corrected first (see the "identical to predictions" warning below).
  - WITHOUT --gold: skips scoring against corrections entirely and just
    measures how much the two models' raw predictions AGREE with each
    other. No corrections needed — but agreement isn't accuracy: the two
    models can agree and both be wrong, or disagree with either one right.
    Useful as a quick "did fine-tuning even change anything" sanity check,
    or to find the abstracts where they disagree — those are the
    interesting ones to prioritize correcting, since abstracts where they
    already agree tell you less either way.

Point this at the JSONL file(s) the dashboard's "View corrected & predicted
clusters" tab already lets you download for the whole session:

    - "Download corrected clusters (JSONL)"             -> --gold (optional)
    - "Download original (predicted) clusters (JSONL)"  -> --finetuned-predicted

Both cover every abstract processed so far, base-model batch and
fine-tuned-model batch together — use --min-id to select just the batch
that ran on the fine-tuned model. Document ids are assigned once,
monotonically, for the life of a session (see _next_doc_id in app.py), so
if you processed 150 abstracts on the base model before switching and then
50 more on the fine-tuned model, those 50 are ids 150-199.

Usage:
    # Accuracy against your corrections:
    python compare_models.py \\
        --gold corrected_clusters.jsonl \\
        --finetuned-predicted original_clusters.jsonl \\
        --min-id 150

    # Just agreement, no corrections needed:
    python compare_models.py \\
        --finetuned-predicted original_clusters.jsonl \\
        --min-id 150

Prints one aggregate precision/recall/F1 row per model (mention-pair
scoring, same math the dashboard's own Metrics tab uses), pooled across
the whole batch. Add --verbose for a per-abstract breakdown too.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utils.finetune import load_finetune_jsonl  # noqa: E402
from utils.scoring import aggregate_pairwise_prf, clusters_to_mentions, compute_pairwise_prf  # noqa: E402


def load_jsonl(path) -> dict:
    """{doc_id: record} from one of the dashboard's exported JSONL files."""
    return load_finetune_jsonl(Path(path).read_bytes())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--gold", default=None,
        help="Corrected clusters JSONL (the whole session is fine). Omit to skip accuracy scoring "
        "and just measure agreement between the two models' raw predictions instead.",
    )
    ap.add_argument(
        "--finetuned-predicted", required=True,
        help="Original (predicted) clusters JSONL from the same session — the fine-tuned model's raw output",
    )
    ap.add_argument("--min-id", type=int, default=0, help="Only compare docs with id >= this (default 0 = all)")
    ap.add_argument("--max-id", type=int, default=None, help="Only compare docs with id <= this (default: no cap)")
    ap.add_argument(
        "--base-model", default="biu-nlp/f-coref",
        help="Base model to run for comparison (default: biu-nlp/f-coref, fastcoref's stock checkpoint)",
    )
    ap.add_argument("--device", default=None, help="cuda / mps / cpu (default: auto)")
    ap.add_argument("--verbose", action="store_true", help="Also print a per-abstract breakdown")
    args = ap.parse_args()

    agreement_mode = args.gold is None
    finetuned_by_id = load_jsonl(args.finetuned_predicted)
    # In agreement mode there's no separate gold file — the predicted-clusters file
    # already carries "text" too, so it doubles as the source of abstract text.
    gold_by_id = load_jsonl(args.gold) if not agreement_mode else finetuned_by_id

    doc_ids = sorted(
        i for i in (set(gold_by_id) & set(finetuned_by_id))
        if i >= args.min_id and (args.max_id is None or i <= args.max_id)
    )
    if not doc_ids:
        raise SystemExit(
            f"No docs in both files with id in [{args.min_id}, {args.max_id or 'inf'}] — "
            f"gold has {len(gold_by_id)} docs, finetuned-predicted has {len(finetuned_by_id)}. "
            "Check --min-id/--max-id, or that both files came from the same session."
        )
    if not agreement_mode:
        missing_gold = set(finetuned_by_id) - set(gold_by_id)
        missing_ft = set(gold_by_id) - set(finetuned_by_id)
        if missing_gold or missing_ft:
            print(
                f"Note: {len(missing_gold)} doc(s) in --finetuned-predicted not in --gold, "
                f"{len(missing_ft)} the other way — skipping those, comparing the {len(doc_ids)} in common.",
                file=sys.stderr,
            )

    print(f"Loading base model ({args.base_model})...", file=sys.stderr)
    from fastcoref import FCoref

    base_model = FCoref(device=args.device, model_name_or_path=args.base_model)
    texts = [gold_by_id[i]["text"] for i in doc_ids]
    print(f"Running base model on {len(texts)} abstract(s)...", file=sys.stderr)
    base_preds = base_model.predict(texts=texts)

    base_prfs, finetuned_prfs = [], []
    per_doc_rows = []

    for doc_id, base_pred in zip(doc_ids, base_preds):
        finetuned_mentions = clusters_to_mentions(finetuned_by_id[doc_id]["clusters"])
        base_mentions = clusters_to_mentions(base_pred.get_clusters(as_strings=False))

        if agreement_mode:
            # No gold: score the two models directly against each other. Precision/recall
            # are directional (base against fine-tuned-as-reference) but F1 reads fine as a
            # symmetric-ish agreement score — same math as accuracy mode, different reference.
            base_prf = compute_pairwise_prf(base_mentions, finetuned_mentions)
            finetuned_prf = {"precision": 1.0, "recall": 1.0, "f1": 1.0}  # trivially agrees with itself
        else:
            gold_mentions = clusters_to_mentions(gold_by_id[doc_id]["clusters"])
            base_prf = compute_pairwise_prf(base_mentions, gold_mentions)
            finetuned_prf = compute_pairwise_prf(finetuned_mentions, gold_mentions)
        base_prfs.append(base_prf)
        finetuned_prfs.append(finetuned_prf)

        per_doc_rows.append((doc_id, gold_by_id[doc_id].get("label", str(doc_id)), base_prf, finetuned_prf))

    base_agg = aggregate_pairwise_prf(base_prfs)

    if args.verbose and not agreement_mode:
        print(f"\n{'Abstract':<30}{'Base F1':>10}{'Fine-tuned F1':>16}")
        for doc_id, label, base_prf, finetuned_prf in per_doc_rows:
            print(f"{label[:28]:<30}{base_prf['f1']:>10.3f}{finetuned_prf['f1']:>16.3f}")
    elif args.verbose and agreement_mode:
        print(f"\n{'Abstract':<30}{'Agreement F1':>14}")
        for doc_id, label, base_prf, _ in per_doc_rows:
            print(f"{label[:28]:<30}{base_prf['f1']:>14.3f}")

    print(f"\nPooled over {len(doc_ids)} abstract(s) (ids {doc_ids[0]}-{doc_ids[-1]}):\n")

    if agreement_mode:
        print(f"{'Precision':>10}{'Recall':>10}{'F1':>10}   (base vs. fine-tuned, no gold)")
        print(f"{base_agg['precision']:>10.3f}{base_agg['recall']:>10.3f}{base_agg['f1']:>10.3f}")
        print(
            f"\nAgreement F1 = {base_agg['f1']:.3f} — this is how much the two models' raw "
            "predictions overlap, NOT accuracy (no gold was used). Low agreement means "
            "fine-tuning changed real behavior; it doesn't say which model is more correct. "
            "Re-run with --gold once these abstracts are corrected to get an actual accuracy number."
        )
    else:
        finetuned_agg = aggregate_pairwise_prf(finetuned_prfs)
        print(f"{'Model':<14}{'Precision':>10}{'Recall':>10}{'F1':>10}")
        print(f"{'Base':<14}{base_agg['precision']:>10.3f}{base_agg['recall']:>10.3f}{base_agg['f1']:>10.3f}")
        print(f"{'Fine-tuned':<14}{finetuned_agg['precision']:>10.3f}{finetuned_agg['recall']:>10.3f}{finetuned_agg['f1']:>10.3f}")

        delta = finetuned_agg["f1"] - base_agg["f1"]
        if delta > 0.005:
            verdict = "the fine-tuned model beats the base model"
        elif delta < -0.005:
            verdict = "the fine-tuned model is worse than the base model"
        else:
            verdict = "the fine-tuned model is about the same as the base model"
        print(f"\nFine-tuned F1 is {delta:+.3f} vs base — {verdict} on this batch.")


if __name__ == "__main__":
    main()
