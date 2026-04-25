"""Week 06: CNN-style feature extraction with convolution and pooling."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dl_playground.tiny import MLPClassifier, conv2d_valid, conv_features, ensure_dir, make_bar_images, set_seed, train_val_split, write_json


def train(epochs: int, lr: float, seed: int) -> dict[str, object]:
    set_seed(seed)
    images, labels = make_bar_images(samples_per_class=80, seed=seed)
    xs = [conv_features(image) for image in images]
    train_x, train_y, val_x, val_y = train_val_split(xs, labels, val_ratio=0.25, seed=seed)
    model = MLPClassifier(input_dim=len(xs[0]), hidden_dim=16, output_dim=3, seed=seed)
    for _ in range(epochs):
        model.train_epoch(train_x, [int(v) for v in train_y], lr=lr)
    example_conv = conv2d_valid(images[0], [[1, -1], [1, -1]])
    return {
        "epochs": epochs,
        "learning_rate": lr,
        "image_shape": [8, 8],
        "conv_output_shape": [len(example_conv), len(example_conv[0])],
        "feature_dim": len(xs[0]),
        "final_train": model.evaluate(train_x, [int(v) for v in train_y]),
        "final_val": model.evaluate(val_x, [int(v) for v in val_y]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    metrics = train(args.epochs, args.lr, args.seed)
    out_dir = ensure_dir(Path(__file__).resolve().parents[1] / "outputs")
    write_json(out_dir / "metrics.json", metrics)
    print(
        "week06 "
        f"conv_shape={metrics['conv_output_shape']} "
        f"feature_dim={metrics['feature_dim']} "
        f"val_acc={metrics['final_val']['accuracy']:.3f}"
    )


if __name__ == "__main__":
    main()
