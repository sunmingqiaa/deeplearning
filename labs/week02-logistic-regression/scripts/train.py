"""Week 02: binary logistic regression and classification metrics."""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dl_playground.tiny import binary_classification_report, dot, ensure_dir, make_binary_blobs, mean, sigmoid, write_json


def train(epochs: int, lr: float, seed: int) -> dict[str, object]:
    xs, ys = make_binary_blobs(seed=seed)
    rng = random.Random(seed)
    w = [0.0, 0.0]
    b = 0.0
    history = []

    for _ in range(epochs):
        order = list(range(len(xs)))
        rng.shuffle(order)
        losses = []
        for idx in order:
            x, y = xs[idx], ys[idx]
            prob = sigmoid(dot(w, x) + b)
            losses.append(-(y * math.log(max(prob, 1e-12)) + (1 - y) * math.log(max(1 - prob, 1e-12))))
            grad = prob - y
            w[0] -= lr * grad * x[0]
            w[1] -= lr * grad * x[1]
            b -= lr * grad
        history.append(mean(losses))

    probs = [sigmoid(dot(w, x) + b) for x in xs]
    report_05 = binary_classification_report(probs, ys, threshold=0.5)
    report_07 = binary_classification_report(probs, ys, threshold=0.7)
    return {
        "epochs": epochs,
        "learning_rate": lr,
        "initial_loss": history[0],
        "final_loss": history[-1],
        "weights": w,
        "bias": b,
        "threshold_0_5": report_05,
        "threshold_0_7": report_07,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    metrics = train(args.epochs, args.lr, args.seed)
    out_dir = ensure_dir(Path(__file__).resolve().parents[1] / "outputs")
    write_json(out_dir / "metrics.json", metrics)
    report = metrics["threshold_0_5"]
    print(
        "week02 "
        f"loss={metrics['final_loss']:.4f} "
        f"accuracy={report['accuracy']:.3f} "
        f"precision={report['precision']:.3f} "
        f"recall={report['recall']:.3f}"
    )


if __name__ == "__main__":
    main()
