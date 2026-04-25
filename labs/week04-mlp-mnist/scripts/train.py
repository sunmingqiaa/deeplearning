"""Week 04: MLP classifier on synthetic 8x8 digit-like images."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dl_playground.tiny import MLPClassifier, ensure_dir, make_tiny_digit_data, set_seed, train_val_split, write_json


def train(epochs: int, lr: float, seed: int) -> dict[str, object]:
    set_seed(seed)
    xs, ys = make_tiny_digit_data(samples_per_class=90, seed=seed)
    train_x, train_y, val_x, val_y = train_val_split(xs, ys, val_ratio=0.25, seed=seed)
    model = MLPClassifier(input_dim=64, hidden_dim=24, output_dim=3, seed=seed)
    history = []
    for epoch in range(1, epochs + 1):
        train_loss = model.train_epoch(train_x, [int(v) for v in train_y], lr=lr)
        train_metrics = model.evaluate(train_x, [int(v) for v in train_y])
        val_metrics = model.evaluate(val_x, [int(v) for v in val_y])
        history.append({"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_metrics["accuracy"], "val_accuracy": val_metrics["accuracy"]})
    return {
        "epochs": epochs,
        "learning_rate": lr,
        "train_size": len(train_x),
        "val_size": len(val_x),
        "final_train": model.evaluate(train_x, [int(v) for v in train_y]),
        "final_val": model.evaluate(val_x, [int(v) for v in val_y]),
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--lr", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    metrics = train(args.epochs, args.lr, args.seed)
    out_dir = ensure_dir(Path(__file__).resolve().parents[1] / "outputs")
    write_json(out_dir / "metrics.json", metrics)
    print(
        "week04 "
        f"train_acc={metrics['final_train']['accuracy']:.3f} "
        f"val_acc={metrics['final_val']['accuracy']:.3f} "
        f"val_loss={metrics['final_val']['loss']:.4f}"
    )


if __name__ == "__main__":
    main()
