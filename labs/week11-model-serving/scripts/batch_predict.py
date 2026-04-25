"""Week 11: batch prediction with a saved linear model format."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dl_playground.tiny import ensure_dir, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--w", type=float, default=2.7)
    parser.add_argument("--b", type=float, default=-0.8)
    args = parser.parse_args()

    out_dir = ensure_dir(Path(__file__).resolve().parents[1] / "outputs")
    inputs = [-3, -1, 0, 1, 3]
    rows = [{"x": x, "prediction": args.w * x + args.b} for x in inputs]
    csv_path = out_dir / "batch_predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["x", "prediction"])
        writer.writeheader()
        writer.writerows(rows)
    write_json(out_dir / "batch_summary.json", {"model": {"w": args.w, "b": args.b}, "rows": rows})
    print(f"week11 wrote {csv_path}")


if __name__ == "__main__":
    main()
