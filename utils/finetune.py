"""Launches scripts/finetune/train.py as a background subprocess and lets the
Fine-tune dashboard tab watch it, instead of re-implementing training inline.

Training pulls in torch/transformers/fastcoref's trainer and can run for a
long time — running it in-process would freeze the UI for the whole run and
risks fighting the already-loaded inference model (utils/inference.py) for
memory. A subprocess keeps it isolated: it survives navigating to other
tabs, and a crash in training can't take the dashboard down with it.
"""

import json
import random
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FINETUNE_SCRIPT = REPO_ROOT / "scripts" / "finetune" / "train.py"
RUNS_DIR = REPO_ROOT / "finetune_runs"


def prepare_split(records: list, run_dir: Path, dev_n: int, test_n: int = 0, seed: int = 42) -> tuple[dict, dict]:
    """Shuffle `records` (as returned by export_finetuning_records) and write
    train.jsonl / dev.jsonl (/ test.jsonl) into run_dir. Same logic as
    scripts/finetune/split_dataset.py, kept in sync by hand since one runs as
    a standalone CLI step and the other from inside Streamlit.

    Returns ({split_name: Path}, {split_name: count})."""
    if dev_n + test_n >= len(records):
        raise ValueError(
            f"dev ({dev_n}) + test ({test_n}) must leave at least 1 document for training "
            f"(have {len(records)} total)"
        )

    shuffled = records[:]
    random.Random(seed).shuffle(shuffled)
    dev = shuffled[:dev_n]
    test = shuffled[dev_n : dev_n + test_n]
    train = shuffled[dev_n + test_n :]

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    def write(name, rows):
        path = run_dir / name
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        return path

    paths = {"train": write("train.jsonl", train), "dev": write("dev.jsonl", dev)}
    counts = {"train": len(train), "dev": len(dev)}
    if test_n:
        paths["test"] = write("test.jsonl", test)
        counts["test"] = len(test)
    return paths, counts


def start_finetune_job(
    train_path,
    dev_path,
    output_dir,
    *,
    model_name_or_path: str = "biu-nlp/f-coref",
    epochs: float = 3,
    learning_rate: float = 1e-5,
    eval_steps: int = 20,
    log_path=None,
) -> tuple[subprocess.Popen, Path]:
    """Launch scripts/finetune/train.py in the background, appending its
    output to log_path (default: output_dir/train.log). Returns the Popen
    handle (poll() is None while running, .returncode after) and the log
    path to tail."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(log_path) if log_path else output_dir / "train.log"

    cmd = [
        sys.executable,
        str(FINETUNE_SCRIPT),
        "--train", str(train_path),
        "--dev", str(dev_path),
        "--output-dir", str(output_dir),
        "--model-name-or-path", model_name_or_path,
        "--epochs", str(epochs),
        "--learning-rate", str(learning_rate),
        "--eval-steps", str(eval_steps),
    ]

    log_file = log_path.open("w", encoding="utf-8")
    try:
        # Once Popen has forked+exec'd, the child holds its own copy of the
        # fd — safe (and correct, to avoid leaking it) to close ours here.
        process = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, cwd=str(REPO_ROOT))
    finally:
        log_file.close()

    return process, log_path


def tail(path, max_chars: int = 6000) -> str:
    path = Path(path)
    if not path.exists():
        return ""
    data = path.read_text(encoding="utf-8", errors="replace")
    return data[-max_chars:]


def model_dir(output_dir) -> Path:
    return Path(output_dir) / "model"


def model_ready(output_dir) -> bool:
    return (model_dir(output_dir) / "config.json").exists()
