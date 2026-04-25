"""Week 08: tiny embedding-based text classifier."""

from __future__ import annotations

import argparse
import math
import random
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dl_playground.tiny import accuracy, ensure_dir, mean, sigmoid, write_json


DATA = [
    ("this model is accurate and useful", 1),
    ("training is stable and results are good", 1),
    ("the prediction is fast and reliable", 1),
    ("features are clean and the metric improves", 1),
    ("loss keeps falling and accuracy improves", 1),
    ("this model is broken and unstable", 0),
    ("training failed and results are bad", 0),
    ("the prediction is slow and unreliable", 0),
    ("features are noisy and the metric drops", 0),
    ("loss explodes and accuracy is poor", 0),
]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def train(epochs: int, lr: float, seed: int) -> dict[str, object]:
    rng = random.Random(seed)
    tokenized = [(tokenize(text), label) for text, label in DATA]
    vocab = sorted({token for tokens, _ in tokenized for token in tokens})
    token_to_id = {token: idx for idx, token in enumerate(vocab)}
    dim = 8
    embeddings = [[rng.gauss(0, 0.1) for _ in range(dim)] for _ in vocab]
    weights = [rng.gauss(0, 0.1) for _ in range(dim)]
    bias = 0.0
    history = []

    for _ in range(epochs):
        rng.shuffle(tokenized)
        losses = []
        for tokens, label in tokenized:
            ids = [token_to_id[token] for token in tokens]
            pooled = [mean(embeddings[idx][j] for idx in ids) for j in range(dim)]
            prob = sigmoid(sum(w * v for w, v in zip(weights, pooled, strict=True)) + bias)
            losses.append(-(label * math.log(max(prob, 1e-12)) + (1 - label) * math.log(max(1 - prob, 1e-12))))
            grad = prob - label
            old_weights = weights[:]
            for j in range(dim):
                weights[j] -= lr * grad * pooled[j]
            bias -= lr * grad
            for idx in ids:
                for j in range(dim):
                    embeddings[idx][j] -= lr * grad * old_weights[j] / len(ids)
        history.append(mean(losses))

    probs = []
    labels = []
    for tokens, label in tokenized:
        ids = [token_to_id[token] for token in tokens]
        pooled = [mean(embeddings[idx][j] for idx in ids) for j in range(dim)]
        probs.append(sigmoid(sum(w * v for w, v in zip(weights, pooled, strict=True)) + bias))
        labels.append(label)
    preds = [1 if p >= 0.5 else 0 for p in probs]
    return {
        "epochs": epochs,
        "learning_rate": lr,
        "vocab_size": len(vocab),
        "embedding_dim": dim,
        "initial_loss": history[0],
        "final_loss": history[-1],
        "accuracy": accuracy(preds, labels),
        "example_predictions": [
            {"text": text, "label": label, "prob_positive": probs[idx]}
            for idx, (text, label) in enumerate(DATA[:4])
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    metrics = train(args.epochs, args.lr, args.seed)
    out_dir = ensure_dir(Path(__file__).resolve().parents[1] / "outputs")
    write_json(out_dir / "metrics.json", metrics)
    print(
        "week08 "
        f"vocab={metrics['vocab_size']} "
        f"loss={metrics['final_loss']:.4f} "
        f"accuracy={metrics['accuracy']:.3f}"
    )


if __name__ == "__main__":
    main()
