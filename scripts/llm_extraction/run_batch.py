"""
Agentic loop: run extract_clusters() once per abstract in a CSV, writing
results incrementally so the run can be interrupted and resumed.

Usage:
    # 1. Put your key in scripts/llm_extraction/.env (see .env.example) —
    #    never paste it into chat or a shell command.
    # 2. Dry run first (no API calls, no cost) to sanity-check the prompt
    #    and CSV parsing:
    python run_batch.py --csv path/to/ai_abstracts.csv --out out.jsonl --limit 5 --dry-run

    # 3. Small real test:
    python run_batch.py --csv path/to/ai_abstracts.csv --out out.jsonl --limit 5

    # 4. Full run (resumes automatically if interrupted — just re-run the
    #    same command; already-processed rows are skipped):
    python run_batch.py --csv path/to/ai_abstracts.csv --out out.jsonl

    # Then convert the JSONL to the same CSV shape as the manual gold file:
    python jsonl_to_csv.py --in out.jsonl --out clusters.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from extract_clusters import DEFAULT_MODEL, build_user_content, extract_clusters  # noqa: E402

# (input $/MTok, output $/MTok) — approximate, cache reads priced separately at ~0.1x
# input in the estimate below. Update if pricing changes; see console.anthropic.com/settings/billing
# for exact current rates and your actual invoice for exact spend.
MODEL_PRICE_PER_MTOK = {
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}


def load_already_processed(out_path: Path) -> set[int]:
    if not out_path.exists():
        return set()
    done = set()
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                done.add(row["abstract_num"])
            except (json.JSONDecodeError, KeyError):
                continue  # tolerate a truncated last line from a killed run
    return done


def call_with_retry(client, title: str, abstract: str, model: str, max_retries: int = 5):
    """Extra backoff on top of the SDK's own retries (max_retries=2 by default) —
    worth it on a run spanning thousands of calls where a transient 429/5xx
    streak is more likely to occur somewhere."""
    import anthropic

    delay = 2.0
    for attempt in range(max_retries):
        try:
            return extract_clusters(client, title, abstract, model=model)
        except (anthropic.RateLimitError, anthropic.APIConnectionError) as e:
            if attempt == max_retries - 1:
                raise
            print(f"    retrying after {delay:.0f}s ({type(e).__name__})", file=sys.stderr)
            time.sleep(delay)
            delay = min(delay * 2, 60)
        except anthropic.APIStatusError as e:
            if e.status_code >= 500 and attempt < max_retries - 1:
                print(f"    retrying after {delay:.0f}s (HTTP {e.status_code})", file=sys.stderr)
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue
            raise


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="Path to the abstracts CSV (title,abstract columns)")
    ap.add_argument("--out", required=True, help="Output JSONL path (appended to; resumable)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"Model ID (default: {DEFAULT_MODEL})")
    ap.add_argument("--limit", type=int, default=None, help="Only process the first N not-yet-done abstracts")
    ap.add_argument("--start-row", type=int, default=1, help="1-based row number to start from (default 1)")
    ap.add_argument("--end-row", type=int, default=None, help="1-based row number to stop at (inclusive)")
    ap.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between calls (default 0)")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't call the API — just print what would be sent, for the first rows in range/limit",
    )
    args = ap.parse_args()

    load_dotenv(Path(__file__).parent / ".env")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    already_done = load_already_processed(out_path)
    if already_done:
        print(f"Resuming: {len(already_done)} abstracts already in {out_path}")

    client = None
    if not args.dry_run:
        import anthropic

        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            print(
                "No ANTHROPIC_API_KEY/ANTHROPIC_AUTH_TOKEN found in the environment or .env file.\n"
                "Get a key at https://console.anthropic.com, then put it in "
                f"{Path(__file__).parent / '.env'} as ANTHROPIC_API_KEY=... (see .env.example).\n"
                "Or run with --dry-run to test without a key.",
                file=sys.stderr,
            )
            sys.exit(1)
        client = anthropic.Anthropic()

    n_attempted = 0  # rows we actually tried this run (success + error) — what --limit bounds
    n_processed = 0  # rows that succeeded
    n_errors = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_cache_read = 0

    with out_path.open("a", encoding="utf-8") as out_f:
        with open(args.csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row_num, row in enumerate(reader, start=1):
                if row_num < args.start_row:
                    continue
                if args.end_row is not None and row_num > args.end_row:
                    break
                if row_num in already_done:
                    continue
                if args.limit is not None and n_attempted >= args.limit:
                    break

                title = row["title"]
                abstract = row["abstract"]

                if args.dry_run:
                    print(f"=== abstract {row_num}: {title} ===")
                    print(build_user_content(title, abstract)[:300], "...")
                    print()
                    n_attempted += 1
                    n_processed += 1
                    continue

                n_attempted += 1
                try:
                    result = call_with_retry(client, title, abstract, args.model)
                except anthropic.AuthenticationError as e:
                    # Not transient and not row-specific — every remaining row will fail
                    # identically, so stop immediately instead of burning through the file.
                    print(f"  [{row_num}] AUTH ERROR — key is invalid/rejected: {e}", file=sys.stderr)
                    print(
                        "Stopping: this key will fail on every row. Check .env, fix the key, "
                        "then re-run the same command (already-succeeded rows are skipped).",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                except Exception as e:  # noqa: BLE001 — log and keep going; don't lose the whole run to one bad row
                    print(f"  [{row_num}] ERROR: {type(e).__name__}: {e}", file=sys.stderr)
                    n_errors += 1
                    continue

                clusters = result.parsed_output.clusters
                record = {
                    "abstract_num": row_num,
                    "title": title,
                    "clusters": [c.model_dump() for c in clusters],
                }
                out_f.write(json.dumps(record) + "\n")
                out_f.flush()
                n_processed += 1
                usage = result.usage
                total_input_tokens += usage.input_tokens
                total_output_tokens += usage.output_tokens
                total_cache_read += usage.cache_read_input_tokens or 0
                print(f"  [{row_num}] {len(clusters)} cluster(s) — {title[:60]}")

                if args.sleep:
                    time.sleep(args.sleep)

    if args.dry_run:
        print(f"\nDry run: would process {n_processed} abstract(s). No API calls made, nothing written.")
    else:
        print(
            f"\nDone. Processed {n_processed} abstract(s) this run "
            f"({n_errors} error(s)); {len(already_done) + n_processed} total in {out_path}."
        )
        if n_processed:
            print(
                f"Usage this run: {total_input_tokens:,} input tokens "
                f"({total_cache_read:,} from cache), {total_output_tokens:,} output tokens."
            )
            price = MODEL_PRICE_PER_MTOK.get(args.model)
            if price:
                in_price, out_price = price
                uncached_input = total_input_tokens - total_cache_read
                cost = (
                    uncached_input / 1_000_000 * in_price
                    + total_cache_read / 1_000_000 * (in_price * 0.1)
                    + total_output_tokens / 1_000_000 * out_price
                )
                print(f"Estimated cost this run: ${cost:.4f} (approximate — see Anthropic Console for exact billing)")


if __name__ == "__main__":
    main()
