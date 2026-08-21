"""
Fine-tune fastcoref's FCoref model on your corrected clusters.

This is a one-off training step, not part of the dashboard app — it uses
fastcoref's own CorefTrainer. Not everything it needs is in requirements.txt
(that file only covers what the dashboard itself needs):
    pip install wandb
    python -m spacy download en_core_web_sm   # if not already present

IMPORTANT — CorefTrainer only saves a checkpoint when a dev-set eval beats
the best F1 seen so far, and only evaluates every --eval-steps optimizer
steps. With a few hundred docs, a whole training run can finish in well
under 500 steps — the library's own default — in which case eval never
fires once and NOTHING gets saved. This script defaults --eval-steps much
lower (20) for that reason; if your dataset is larger, raise it. A --dev
file is required for the same reason: no dev set means no checkpoint, ever.

Usage:
    # 1. Export "corrected_clusters.jsonl" from the dashboard's Score tab,
    #    then split it (train/dev, ideally +test):
    python split_dataset.py --in corrected_clusters.jsonl --out-dir data --dev 50

    # 2. Train — defaults to continuing from the same base model the
    #    dashboard uses for inference (biu-nlp/f-coref):
    python train.py --train data/train.jsonl --dev data/dev.jsonl \\
        --output-dir output/finetuned-abstracts

    # 3. The fine-tuned model lands in output/finetuned-abstracts/model
    #    (standard save_pretrained layout). Point the dashboard at it by
    #    editing utils/inference.py's load_model():
    #        return FCoref(device=device, model_name_or_path="output/finetuned-abstracts/model")

Runs offline by default (WANDB_MODE=disabled) so it works with no wandb
account — pass --wandb to log to Weights & Biases instead.
"""

from __future__ import annotations

import argparse
import os


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True, help="Training JSONL ({text, clusters} per line)")
    ap.add_argument("--dev", required=True, help="Dev JSONL — required, see docstring: no dev, no saved checkpoint")
    ap.add_argument("--test", default=None, help="Optional test JSONL, scored once after training")
    ap.add_argument("--output-dir", required=True, help="Where the fine-tuned model is saved (as output-dir/model)")
    ap.add_argument(
        "--model-name-or-path",
        default="biu-nlp/f-coref",
        help="Base model to fine-tune from (default: biu-nlp/f-coref, matching utils/inference.py). "
        "Use biu-nlp/lingmess-coref if the dashboard is switched to LingMess.",
    )
    ap.add_argument("--epochs", type=float, default=3, help="Default 3 — small datasets overfit fast; watch dev F1")
    ap.add_argument("--learning-rate", type=float, default=1e-5)
    ap.add_argument("--head-learning-rate", type=float, default=3e-4)
    ap.add_argument("--max-tokens-in-batch", type=int, default=5000)
    ap.add_argument(
        "--eval-steps",
        type=int,
        default=20,
        help="Steps between dev evals; the model only saves on a new best dev F1 (default 20 — see docstring)",
    )
    ap.add_argument("--logging-steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None, help="cuda / mps / cpu (default: auto — cuda if available, else cpu)")
    ap.add_argument("--wandb", action="store_true", help="Log to Weights & Biases instead of running offline")
    args = ap.parse_args()

    if not args.wandb:
        # WANDB_MODE=disabled alone isn't enough: CorefTrainer calls
        # wandb.init(project=output_dir, ...) unconditionally, and wandb
        # validates the project name's *format* before it even looks at the
        # mode — our output_dir is a nested path (finetune_runs/<run_id>),
        # and "/" isn't legal in a wandb project name, so it raises
        # regardless of WANDB_MODE. Stub out the three calls CorefTrainer
        # actually uses instead of fighting wandb's validation.
        os.environ.setdefault("WANDB_MODE", "disabled")
        import wandb

        wandb.init = lambda *a, **k: None
        wandb.log = lambda *a, **k: None
        wandb.run = type("_NoOpRun", (), {"summary": {}})()

    from fastcoref import CorefTrainer, TrainingArgs

    training_args = TrainingArgs(
        model_name_or_path=args.model_name_or_path,
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        learning_rate=args.learning_rate,
        head_learning_rate=args.head_learning_rate,
        epochs=args.epochs,
        max_tokens_in_batch=args.max_tokens_in_batch,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        seed=args.seed,
        device=args.device,
    )

    trainer = CorefTrainer(
        args=training_args,
        train_file=args.train,
        dev_file=args.dev,
        test_file=args.test,
    )
    trainer.train()

    if args.test:
        trainer.evaluate(test=True)

    model_path = os.path.join(args.output_dir, "model")
    if os.path.isdir(model_path):
        print(f"\nDone. Best checkpoint saved to {model_path}")
        print(f'Try it in the dashboard: FCoref(device=device, model_name_or_path="{model_path}")')
    else:
        print(
            f"\nDone, but {model_path} was never created — dev F1 never improved on step 0, "
            "or eval never fired. Try a lower --eval-steps or more --epochs."
        )


if __name__ == "__main__":
    main()
