"""
Split the JSONL exported from the dashboard ("Download corrected clusters
(JSONL)" button, built from export_finetuning_records() in
utils/session.py) into train/dev/test files for fastcoref's CorefTrainer.

A dev split isn't optional here: CorefTrainer only ever saves a checkpoint
when it beats the best-so-far dev F1 (see train.py's docstring) — skip
--dev and training will run to completion and save nothing.

Usage:
    python split_dataset.py --in corrected_clusters.jsonl --out-dir data --dev 50
    python split_dataset.py --in corrected_clusters.jsonl --out-dir data --dev 50 --test 50
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_path", required=True, help="JSONL exported from the dashboard")
    ap.add_argument("--out-dir", default="data", help="Directory to write train/dev/test .jsonl into (default: data)")
    ap.add_argument("--dev", type=int, default=50, help="Docs held out for dev, scored during training (default 50)")
    ap.add_argument("--test", type=int, default=0, help="Docs held out for a final test score (default 0 — skip)")
    ap.add_argument("--seed", type=int, default=42, help="Shuffle seed, for a reproducible split (default 42)")
    args = ap.parse_args()

    records = []
    with open(args.in_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        raise SystemExit(f"{args.in_path} has no records — nothing to split")
    if args.dev + args.test >= len(records):
        raise SystemExit(
            f"--dev {args.dev} + --test {args.test} >= {len(records)} total docs — nothing left for train"
        )

    rng = random.Random(args.seed)
    rng.shuffle(records)

    dev = records[: args.dev]
    test = records[args.dev : args.dev + args.test]
    train = records[args.dev + args.test :]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def write(name, rows):
        path = out_dir / name
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"{name}: {len(rows)} docs -> {path}")

    write("train.jsonl", train)
    write("dev.jsonl", dev)
    if args.test:
        write("test.jsonl", test)

    print(
        f"\n{len(records)} total -> {len(train)} train / {len(dev)} dev"
        + (f" / {len(test)} test" if args.test else "")
    )


if __name__ == "__main__":
    main()
