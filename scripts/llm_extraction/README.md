# LLM-based cluster extraction

Automates what was previously done by hand in chat: for each abstract, call
Claude once with a fixed prompt to extract multi-mention coreference
clusters (excluding author self-reference), so the result can be diffed
against fastcoref's output at scale.

## Files

- `extract_clusters.py` — the prompt template + one-call extraction function
  (`extract_clusters`). This is the piece to iterate on and validate first.
- `run_batch.py` — the loop: reads the abstracts CSV, calls
  `extract_clusters` once per row, writes results incrementally to a JSONL
  file. Resumable — re-running the same command skips rows already in the
  output file.
- `jsonl_to_csv.py` — converts the JSONL into the same
  `abstract_num,title,cluster_label,mentions,mention_count` CSV shape as the
  manually-built gold file, for direct comparison.

## Setup

```bash
cd scripts/llm_extraction
pip install -r ../../requirements.txt   # adds anthropic + python-dotenv
cp .env.example .env
# edit .env and paste in your own key from https://console.anthropic.com —
# do not paste the key into chat or a shell command
```

## Validate the prompt first (no API key needed, no cost)

```bash
python run_batch.py --csv "/path/to/ai_abstracts.csv" --out /tmp/test.jsonl --limit 5 --dry-run
```

Confirms the CSV is being read correctly and shows exactly what would be
sent to the model.

## Small real test

```bash
python run_batch.py --csv "/path/to/ai_abstracts.csv" --out /tmp/test.jsonl --limit 5
python jsonl_to_csv.py --in /tmp/test.jsonl --out /tmp/test.csv
```

Open `/tmp/test.csv` and spot-check a handful of abstracts against the
manually-built gold CSV (abstracts 1-200) — that's the real validation of
whether the prompt is "correct" before scaling up.

## Full run

```bash
python run_batch.py --csv "/path/to/ai_abstracts.csv" --out clusters_llm.jsonl
python jsonl_to_csv.py --in clusters_llm.jsonl --out clusters_llm.csv
```

If it's interrupted (Ctrl-C, crash, laptop sleeps), just re-run the exact
same command — already-processed abstracts are skipped via `--out`.

## Model / cost

Defaults to `claude-opus-5`. For ~6,700 abstracts that adds up — consider
`claude-sonnet-5` or `claude-haiku-4-5` (`--model claude-sonnet-5`) for a
much cheaper batch run; on a task this well-specified (extract mentions,
verbatim, from a short abstract) a cheaper model may do just as well. Worth
A/B-ing a small `--limit 20` run on 2-3 models before committing to the full
file. The system prompt is cached (`cache_control`) so it's only paid at
full price on the very first call, not on every one of the ~6,700.
