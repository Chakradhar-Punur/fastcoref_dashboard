# Fine-tuning FCoref on your corrections

Turns the corrections you make in the dashboard into an actual fine-tuned
model, using fastcoref's own `CorefTrainer`. This is a separate, manual
step from the dashboard — the app only gets you as far as exporting the
JSONL.

## Files

- `split_dataset.py` — splits the exported JSONL into `train.jsonl` /
  `dev.jsonl` (/ `test.jsonl`).
- `train.py` — runs `CorefTrainer` on the split and saves the fine-tuned
  model.
- `compare_models.py` — scores the base model against the fine-tuned model
  on the same batch of abstracts, against your corrections as gold. See
  step 5 below.

## Setup

```bash
pip install wandb                     # CorefTrainer imports it unconditionally
python -m spacy download en_core_web_sm   # tokenizer used to align clusters to spans
```

## 1. Export your corrections

In the dashboard's Score/Compare tab, click **"Download corrected clusters
(JSONL)"**. This calls `export_finetuning_records(use_corrected=True)`
([utils/session.py](../../utils/session.py)) — one JSON object per
processed abstract: `{"id", "label", "text", "clusters"}`, where `clusters`
is a list of clusters, each a list of `[start, end]` character spans. This
is exactly the shape fastcoref's own loader expects (`text` + `clusters`),
so no reformatting is needed.

Save the download as e.g. `corrected_clusters.jsonl` in this directory.

## 2. Split into train/dev

```bash
python split_dataset.py --in corrected_clusters.jsonl --out-dir data --dev 50
```

A dev split isn't optional — see the warning in `train.py`'s docstring:
`CorefTrainer` only ever saves a checkpoint on a new best dev F1, so
without `--dev` training completes and saves nothing.

With ~110-300 corrected docs, holding out 50 for dev is a reasonable start;
add `--test 50` too once you have enough docs to spare a third slice for a
final, only-look-at-it-once number.

## 3. Train

```bash
python train.py \
    --train data/train.jsonl \
    --dev data/dev.jsonl \
    --output-dir output/finetuned-abstracts
```

Defaults to continuing from `biu-nlp/f-coref` — the same base model
[utils/inference.py](../../utils/inference.py) uses — with a low learning
rate (1e-5 / 3e-4 head) and 3 epochs, since a few hundred docs overfits
fast. Watch the printed dev F1 as it trains; if it's still climbing at the
end, more epochs (`--epochs`) or more corrected data will help more than
tuning the learning rate.

Runs fully offline (no wandb account needed) unless you pass `--wandb`.

## 4. Try the fine-tuned model in the dashboard

The best checkpoint is saved to `output/finetuned-abstracts/model` (a
normal `save_pretrained` directory — config, weights, tokenizer). Point
`load_model()` in [utils/inference.py](../../utils/inference.py) at it:

```python
return FCoref(device=device, model_name_or_path="output/finetuned-abstracts/model")
```

Then re-run inference on a batch you *haven't* corrected yet and compare
its clusters against your gold corrections the same way you already do for
the base model, to see whether fine-tuning actually helped.

## 5. Compare base vs fine-tuned on the same abstracts

If a batch of abstracts only ever ran through the fine-tuned model (e.g.
you switched models partway through a longer session), there's no base
model prediction for them to compare against yet — `compare_models.py`
runs the base model on those same abstracts' text right now and scores
both against your corrections:

```bash
python compare_models.py \
    --gold corrected_clusters.jsonl \
    --finetuned-predicted original_clusters.jsonl \
    --min-id 150   # first id that ran on the fine-tuned model
```

Both input files are the dashboard's usual "All abstracts" exports
(**Download corrected clusters** / **Download original (predicted)
clusters**, from "View corrected & predicted clusters") — they cover the
whole session, base-model batch and fine-tuned-model batch together;
`--min-id`/`--max-id` pick out just the fine-tuned batch by document id
(ids are assigned once, monotonically, for the life of a session — see
`_next_doc_id` in `app.py`). Add `--verbose` for a per-abstract F1
breakdown alongside the pooled precision/recall/F1 for each model.

## Gotchas

- **No `--dev` → no saved model.** `CorefTrainer.train()` runs to
  completion either way; it just never writes a checkpoint.
- **Small datasets need a small `--eval-steps`.** The library default is
  500 steps between evals; a few hundred abstracts can finish training in
  fewer steps than that, so eval (and therefore saving) never fires.
  `train.py` defaults this to 20 — raise it once your dataset is large
  enough that 20 steps is a tiny fraction of an epoch.
- **`wandb.init()` is called unconditionally** by `CorefTrainer`, even
  offline — `train.py` sets `WANDB_MODE=disabled` for you unless you pass
  `--wandb`.
