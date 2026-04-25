"""Week 12: end-to-end mini project for ticket priority classification."""

from __future__ import annotations

import argparse
import math
import random
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dl_playground.tiny import accuracy, ensure_dir, sigmoid, write_json


DATA = [
    ("payment job failed and users cannot checkout", 1),
    ("data pipeline is delayed but dashboard works", 0),
    ("login service returns 500 for all users", 1),
    ("daily report has a typo in one title", 0),
    ("recommendation batch wrote empty results", 1),
    ("add a new chart to weekly report", 0),
    ("model serving latency is above threshold", 1),
    ("rename one column in exported csv", 0),
    ("orders table ingestion stopped for two hours", 1),
    ("archive old debug logs this week", 0),
]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def vectorize(tokens: list[str], vocab: dict[str, int]) -> list[float]:
    vec = [0.0] * len(vocab)
    for token in tokens:
        if token in vocab:
            vec[vocab[token]] += 1.0
    total = max(1.0, sum(vec))
    return [v / total for v in vec]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--lr", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    tokenized = [(tokenize(text), label) for text, label in DATA]
    vocab = {token: idx for idx, token in enumerate(sorted({tok for tokens, _ in tokenized for tok in tokens}))}
    xs = [vectorize(tokens, vocab) for tokens, _ in tokenized]
    ys = [label for _, label in tokenized]
    weights = [0.0] * len(vocab)
    bias = 0.0
    losses = []
    order = list(range(len(xs)))
    for _ in range(args.epochs):
        rng.shuffle(order)
        epoch_losses = []
        for idx in order:
            x, y = xs[idx], ys[idx]
            prob = sigmoid(sum(w * v for w, v in zip(weights, x, strict=True)) + bias)
            epoch_losses.append(-(y * math.log(max(prob, 1e-12)) + (1 - y) * math.log(max(1 - prob, 1e-12))))
            grad = prob - y
            for j in range(len(weights)):
                weights[j] -= args.lr * grad * x[j]
            bias -= args.lr * grad
        losses.append(sum(epoch_losses) / len(epoch_losses))
    probs = [sigmoid(sum(w * v for w, v in zip(weights, x, strict=True)) + bias) for x in xs]
    preds = [1 if p >= 0.5 else 0 for p in probs]
    out_dir = ensure_dir(Path(__file__).resolve().parents[1] / "outputs")
    model = {"vocab": vocab, "weights": weights, "bias": bias}
    metrics = {
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "accuracy": accuracy(preds, ys),
        "predictions": [
            {"text": text, "label": label, "prob_high_priority": probs[idx], "prediction": preds[idx]}
            for idx, (text, label) in enumerate(DATA)
        ],
    }
    write_json(out_dir / "model.json", model)
    write_json(out_dir / "metrics.json", metrics)
    print(
        "week12 "
        f"vocab={len(vocab)} "
        f"loss={metrics['final_loss']:.4f} "
        f"accuracy={metrics['accuracy']:.3f}"
    )


if __name__ == "__main__":
    main()
