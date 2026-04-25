"""Run all runnable lab scripts with small settings.

This is a standard-library smoke test so it works before pytest is installed.
It intentionally skips the long-running HTTP server script.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


COMMANDS = [
    ["labs/week01-linear-regression/scripts/train.py", "--epochs", "8"],
    ["labs/week02-logistic-regression/scripts/train.py", "--epochs", "8"],
    ["labs/week03-pytorch-autograd/scripts/tiny_autograd.py", "--epochs", "8"],
    ["labs/week04-mlp-mnist/scripts/train.py", "--epochs", "3"],
    ["labs/week05-regularization/scripts/compare_regularization.py", "--epochs", "3"],
    ["labs/week06-cnn-cifar10/scripts/train.py", "--epochs", "3"],
    ["labs/week07-transfer-learning/scripts/transfer_demo.py", "--epochs", "3"],
    ["labs/week08-text-classification/scripts/train.py", "--epochs", "5"],
    ["labs/week09-attention-transformer/scripts/attention_demo.py"],
    ["labs/week10-recommender-movielens/scripts/train.py", "--epochs", "5"],
    ["labs/week11-model-serving/scripts/batch_predict.py"],
    ["labs/week12-final-project/scripts/train.py", "--epochs", "5"],
]


JSON_OUTPUTS = [
    "labs/week01-linear-regression/outputs/metrics.json",
    "labs/week02-logistic-regression/outputs/metrics.json",
    "labs/week03-pytorch-autograd/outputs/metrics.json",
    "labs/week04-mlp-mnist/outputs/metrics.json",
    "labs/week05-regularization/outputs/regularization_metrics.json",
    "labs/week06-cnn-cifar10/outputs/metrics.json",
    "labs/week07-transfer-learning/outputs/transfer_metrics.json",
    "labs/week08-text-classification/outputs/metrics.json",
    "labs/week09-attention-transformer/outputs/attention.json",
    "labs/week10-recommender-movielens/outputs/metrics.json",
    "labs/week11-model-serving/outputs/batch_summary.json",
    "labs/week12-final-project/outputs/metrics.json",
]


def main() -> None:
    for command in COMMANDS:
        result = subprocess.run(
            [sys.executable, *command],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr, file=sys.stderr)
            raise SystemExit(f"failed: {' '.join(command)}")
        print(result.stdout.strip())

    for relative_path in JSON_OUTPUTS:
        path = REPO_ROOT / relative_path
        if not path.exists():
            raise SystemExit(f"missing output: {relative_path}")
        json.loads(path.read_text(encoding="utf-8"))

    print("smoke tests passed")


if __name__ == "__main__":
    main()
