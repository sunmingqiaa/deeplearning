"""Week 07: transfer-learning style frozen feature extractor demo."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dl_playground.tiny import MLPClassifier, conv_features, ensure_dir, flatten, make_bar_images, set_seed, train_val_split, write_json


def train_head(features: list[list[float]], labels: list[int], epochs: int, lr: float, seed: int) -> dict[str, object]:
    train_x, train_y, val_x, val_y = train_val_split(features, labels, val_ratio=0.45, seed=seed)
    model = MLPClassifier(input_dim=len(features[0]), hidden_dim=12, output_dim=3, seed=seed)
    for _ in range(epochs):
        model.train_epoch(train_x, [int(v) for v in train_y], lr=lr)
    return {
        "train": model.evaluate(train_x, [int(v) for v in train_y]),
        "val": model.evaluate(val_x, [int(v) for v in val_y]),
        "feature_dim": len(features[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    images, labels = make_bar_images(samples_per_class=18, seed=args.seed)
    frozen_conv_features = [conv_features(image) for image in images]
    raw_pixel_features = [flatten(image) for image in images]
    results = {
        "frozen_conv_feature_head": train_head(frozen_conv_features, labels, args.epochs, args.lr, args.seed),
        "raw_pixel_head": train_head(raw_pixel_features, labels, args.epochs, args.lr, args.seed),
    }
    out_dir = ensure_dir(Path(__file__).resolve().parents[1] / "outputs")
    write_json(out_dir / "transfer_metrics.json", results)
    print(
        "week07 "
        f"frozen_val_acc={results['frozen_conv_feature_head']['val']['accuracy']:.3f} "
        f"raw_val_acc={results['raw_pixel_head']['val']['accuracy']:.3f}"
    )


if __name__ == "__main__":
    main()
