"""Small standard-library helpers for the learning labs.

The goal is educational clarity and zero external dependencies. These helpers
are intentionally small; they are not replacements for NumPy or PyTorch.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Iterable, Sequence


Number = int | float


def set_seed(seed: int = 42) -> None:
    random.seed(seed)


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_json(path: str | Path, payload: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def mean(values: Iterable[Number]) -> float:
    values = list(values)
    return sum(values) / max(1, len(values))


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def softmax(logits: Sequence[float]) -> list[float]:
    max_logit = max(logits)
    exps = [math.exp(v - max_logit) for v in logits]
    total = sum(exps)
    return [v / total for v in exps]


def relu(x: float) -> float:
    return x if x > 0 else 0.0


def relu_grad(x: float) -> float:
    return 1.0 if x > 0 else 0.0


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def argmax(values: Sequence[float]) -> int:
    return max(range(len(values)), key=lambda idx: values[idx])


def accuracy(preds: Sequence[int], labels: Sequence[int]) -> float:
    if not labels:
        return 0.0
    return sum(int(p == y) for p, y in zip(preds, labels, strict=True)) / len(labels)


def train_val_split(
    xs: Sequence[list[float]],
    ys: Sequence[int | float],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[list[float]], list[int | float], list[list[float]], list[int | float]]:
    rng = random.Random(seed)
    indices = list(range(len(xs)))
    rng.shuffle(indices)
    val_size = max(1, int(len(xs) * val_ratio))
    val_indices = set(indices[:val_size])
    train_x, train_y, val_x, val_y = [], [], [], []
    for idx, (x, y) in enumerate(zip(xs, ys, strict=True)):
        if idx in val_indices:
            val_x.append(list(x))
            val_y.append(y)
        else:
            train_x.append(list(x))
            train_y.append(y)
    return train_x, train_y, val_x, val_y


def binary_classification_report(probs: Sequence[float], labels: Sequence[int], threshold: float = 0.5) -> dict[str, float | int]:
    preds = [1 if p >= threshold else 0 for p in probs]
    tp = sum(1 for p, y in zip(preds, labels, strict=True) if p == 1 and y == 1)
    tn = sum(1 for p, y in zip(preds, labels, strict=True) if p == 0 and y == 0)
    fp = sum(1 for p, y in zip(preds, labels, strict=True) if p == 1 and y == 0)
    fn = sum(1 for p, y in zip(preds, labels, strict=True) if p == 0 and y == 1)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {
        "threshold": threshold,
        "accuracy": (tp + tn) / max(1, len(labels)),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(1e-12, precision + recall),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def make_linear_regression_data(n: int = 120, seed: int = 42) -> tuple[list[float], list[float]]:
    rng = random.Random(seed)
    xs, ys = [], []
    for _ in range(n):
        x = rng.uniform(-5, 5)
        noise = rng.gauss(0, 0.6)
        xs.append(x)
        ys.append(2.7 * x - 0.8 + noise)
    return xs, ys


def make_binary_blobs(n: int = 160, seed: int = 42) -> tuple[list[list[float]], list[int]]:
    rng = random.Random(seed)
    xs, ys = [], []
    for i in range(n):
        label = i % 2
        cx, cy = (1.5, 1.5) if label else (-1.5, -1.5)
        xs.append([rng.gauss(cx, 0.8), rng.gauss(cy, 0.8)])
        ys.append(label)
    return xs, ys


def make_tiny_digit_data(samples_per_class: int = 80, seed: int = 42) -> tuple[list[list[float]], list[int]]:
    """Generate 8x8 noisy patterns for classes 0, 1 and 2."""
    rng = random.Random(seed)
    base = []
    zero = [[0.0] * 8 for _ in range(8)]
    one = [[0.0] * 8 for _ in range(8)]
    two = [[0.0] * 8 for _ in range(8)]
    for i in range(8):
        zero[0][i] = zero[7][i] = zero[i][0] = zero[i][7] = 1.0
        one[i][4] = 1.0
        if i in (0, 3, 7):
            for j in range(8):
                two[i][j] = 1.0
        two[i][7 if i < 3 else 0] = 1.0
    base = [zero, one, two]
    xs, ys = [], []
    for label, pattern in enumerate(base):
        for _ in range(samples_per_class):
            row = []
            for r in range(8):
                for c in range(8):
                    value = pattern[r][c] + rng.gauss(0, 0.22)
                    row.append(min(1.0, max(0.0, value)))
            xs.append(row)
            ys.append(label)
    combined = list(zip(xs, ys, strict=True))
    rng.shuffle(combined)
    return [x for x, _ in combined], [y for _, y in combined]


def make_bar_images(samples_per_class: int = 60, seed: int = 42) -> tuple[list[list[list[float]]], list[int]]:
    """Generate 8x8 images: vertical, horizontal and diagonal bars."""
    rng = random.Random(seed)
    images, labels = [], []
    for label in range(3):
        for _ in range(samples_per_class):
            image = [[rng.uniform(0, 0.15) for _ in range(8)] for _ in range(8)]
            offset = rng.choice([2, 3, 4, 5])
            for i in range(8):
                if label == 0:
                    image[i][offset] = rng.uniform(0.85, 1.0)
                elif label == 1:
                    image[offset][i] = rng.uniform(0.85, 1.0)
                else:
                    image[i][i] = rng.uniform(0.85, 1.0)
            images.append(image)
            labels.append(label)
    combined = list(zip(images, labels, strict=True))
    rng.shuffle(combined)
    return [x for x, _ in combined], [y for _, y in combined]


class MLPClassifier:
    """One-hidden-layer classifier trained with stochastic gradient descent."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        seed: int = 42,
        dropout: float = 0.0,
        l2: float = 0.0,
    ) -> None:
        rng = random.Random(seed)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.l2 = l2
        self.w1 = [[rng.gauss(0, 0.12) for _ in range(input_dim)] for _ in range(hidden_dim)]
        self.b1 = [0.0 for _ in range(hidden_dim)]
        self.w2 = [[rng.gauss(0, 0.12) for _ in range(hidden_dim)] for _ in range(output_dim)]
        self.b2 = [0.0 for _ in range(output_dim)]

    def forward(self, x: Sequence[float], train: bool = False) -> tuple[list[float], dict[str, list[float]]]:
        z1 = [dot(row, x) + b for row, b in zip(self.w1, self.b1, strict=True)]
        h = [relu(v) for v in z1]
        mask = [1.0] * self.hidden_dim
        if train and self.dropout > 0:
            keep_prob = 1.0 - self.dropout
            mask = [1.0 / keep_prob if random.random() < keep_prob else 0.0 for _ in h]
        h_used = [v * m for v, m in zip(h, mask, strict=True)]
        logits = [dot(row, h_used) + b for row, b in zip(self.w2, self.b2, strict=True)]
        probs = softmax(logits)
        return probs, {"z1": z1, "h": h, "h_used": h_used, "mask": mask}

    def predict(self, xs: Sequence[Sequence[float]]) -> list[int]:
        return [argmax(self.forward(x, train=False)[0]) for x in xs]

    def evaluate(self, xs: Sequence[Sequence[float]], ys: Sequence[int]) -> dict[str, float]:
        losses = []
        preds = []
        for x, y in zip(xs, ys, strict=True):
            probs, _ = self.forward(x, train=False)
            losses.append(-math.log(max(1e-12, probs[y])))
            preds.append(argmax(probs))
        return {"loss": mean(losses), "accuracy": accuracy(preds, ys)}

    def train_epoch(self, xs: Sequence[Sequence[float]], ys: Sequence[int], lr: float = 0.05) -> float:
        order = list(range(len(xs)))
        random.shuffle(order)
        losses = []
        for idx in order:
            x = xs[idx]
            y = ys[idx]
            probs, cache = self.forward(x, train=True)
            losses.append(-math.log(max(1e-12, probs[y])))

            dlogits = list(probs)
            dlogits[y] -= 1.0
            h_used = cache["h_used"]
            z1 = cache["z1"]
            mask = cache["mask"]

            old_w2 = [row[:] for row in self.w2]
            for out_idx in range(self.output_dim):
                for hidden_idx in range(self.hidden_dim):
                    grad = dlogits[out_idx] * h_used[hidden_idx] + self.l2 * self.w2[out_idx][hidden_idx]
                    self.w2[out_idx][hidden_idx] -= lr * grad
                self.b2[out_idx] -= lr * dlogits[out_idx]

            dh = []
            for hidden_idx in range(self.hidden_dim):
                grad = sum(old_w2[out_idx][hidden_idx] * dlogits[out_idx] for out_idx in range(self.output_dim))
                grad *= mask[hidden_idx] * relu_grad(z1[hidden_idx])
                dh.append(grad)

            for hidden_idx in range(self.hidden_dim):
                for input_idx in range(self.input_dim):
                    grad = dh[hidden_idx] * x[input_idx] + self.l2 * self.w1[hidden_idx][input_idx]
                    self.w1[hidden_idx][input_idx] -= lr * grad
                self.b1[hidden_idx] -= lr * dh[hidden_idx]
        return mean(losses)


def conv2d_valid(image: Sequence[Sequence[float]], kernel: Sequence[Sequence[float]]) -> list[list[float]]:
    height = len(image)
    width = len(image[0])
    kh = len(kernel)
    kw = len(kernel[0])
    out = []
    for r in range(height - kh + 1):
        row = []
        for c in range(width - kw + 1):
            value = 0.0
            for kr in range(kh):
                for kc in range(kw):
                    value += image[r + kr][c + kc] * kernel[kr][kc]
            row.append(value)
        out.append(row)
    return out


def max_pool2x2(feature: Sequence[Sequence[float]]) -> list[list[float]]:
    out = []
    for r in range(0, len(feature) - 1, 2):
        row = []
        for c in range(0, len(feature[0]) - 1, 2):
            row.append(max(feature[r][c], feature[r + 1][c], feature[r][c + 1], feature[r + 1][c + 1]))
        out.append(row)
    return out


def flatten(matrix: Sequence[Sequence[float]]) -> list[float]:
    return [value for row in matrix for value in row]


def conv_features(image: Sequence[Sequence[float]]) -> list[float]:
    kernels = [
        [[1, -1], [1, -1]],
        [[1, 1], [-1, -1]],
        [[1, 0], [0, -1]],
        [[0, 1], [-1, 0]],
    ]
    features: list[float] = []
    for kernel in kernels:
        conv = conv2d_valid(image, kernel)
        pooled = max_pool2x2(conv)
        features.extend(flatten(pooled))
    return features
