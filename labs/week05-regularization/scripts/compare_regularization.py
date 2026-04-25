"""Week 05: compare no regularization with dropout and L2 weight decay."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dl_playground.tiny import MLPClassifier, ensure_dir, make_tiny_digit_data, set_seed, train_val_split, write_json


def run_case(name: str, dropout: float, l2: float, epochs: int, lr: float, seed: int) -> dict[str, object]:
    set_seed(seed)
    xs, ys = make_tiny_digit_data(samples_per_class=24, seed=seed)
    train_x, train_y, val_x, val_y = train_val_split(xs, ys, val_ratio=0.5, seed=seed)
    model = MLPClassifier(input_dim=64, hidden_dim=48, output_dim=3, seed=seed, dropout=dropout, l2=l2)
    for _ in range(epochs):
        model.train_epoch(train_x, [int(v) for v in train_y], lr=lr)
    return {
        "name": name,
        "dropout": dropout,
        "l2": l2,
        "train": model.evaluate(train_x, [int(v) for v in train_y]),
        "val": model.evaluate(val_x, [int(v) for v in val_y]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results = [
        run_case("baseline", 0.0, 0.0, args.epochs, args.lr, args.seed),
        run_case("dropout_l2", 0.25, 0.001, args.epochs, args.lr, args.seed),
    ]
    out_dir = ensure_dir(Path(__file__).resolve().parents[1] / "outputs")
    write_json(out_dir / "regularization_metrics.json", {"results": results})
    for item in results:
        print(
            "week05 "
            f"{item['name']} "
            f"train_acc={item['train']['accuracy']:.3f} "
            f"val_acc={item['val']['accuracy']:.3f} "
            f"val_loss={item['val']['loss']:.4f}"
        )


if __name__ == "__main__":
    main()
