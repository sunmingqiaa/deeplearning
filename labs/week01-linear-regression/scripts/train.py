"""Week 01: linear regression with hand-written gradient descent."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dl_playground.tiny import ensure_dir, make_linear_regression_data, mean, set_seed, write_json


def mse(xs: list[float], ys: list[float], w: float, b: float) -> float:
    return mean((w * x + b - y) ** 2 for x, y in zip(xs, ys, strict=True))


def train(epochs: int, lr: float, seed: int) -> dict[str, float | int]:
    set_seed(seed)
    xs, ys = make_linear_regression_data(seed=seed)
    w, b = 0.0, 0.0
    losses = []

    for _ in range(epochs):
        dw = mean(2 * (w * x + b - y) * x for x, y in zip(xs, ys, strict=True))
        db = mean(2 * (w * x + b - y) for x, y in zip(xs, ys, strict=True))
        w -= lr * dw
        b -= lr * db
        losses.append(mse(xs, ys, w, b))

    return {
        "epochs": epochs,
        "learning_rate": lr,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "learned_w": w,
        "learned_b": b,
        "target_w": 2.7,
        "target_b": -0.8,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    metrics = train(args.epochs, args.lr, args.seed)
    out_dir = ensure_dir(Path(__file__).resolve().parents[1] / "outputs")
    write_json(out_dir / "metrics.json", metrics)
    print(
        "week01 "
        f"initial_loss={metrics['initial_loss']:.4f} "
        f"final_loss={metrics['final_loss']:.4f} "
        f"w={metrics['learned_w']:.3f} "
        f"b={metrics['learned_b']:.3f}"
    )


if __name__ == "__main__":
    main()
