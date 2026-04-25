"""Week 09: scaled dot-product self-attention demo."""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dl_playground.tiny import ensure_dir, softmax, write_json


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def matvec(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [dot(row, vector) for row in matrix]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    tokens = ["data", "engineer", "learns", "attention"]
    dim = 4
    embeddings = [[rng.gauss(0, 1) for _ in range(dim)] for _ in tokens]
    wq = [[rng.gauss(0, 0.4) for _ in range(dim)] for _ in range(dim)]
    wk = [[rng.gauss(0, 0.4) for _ in range(dim)] for _ in range(dim)]
    wv = [[rng.gauss(0, 0.4) for _ in range(dim)] for _ in range(dim)]
    q = [matvec(wq, emb) for emb in embeddings]
    k = [matvec(wk, emb) for emb in embeddings]
    v = [matvec(wv, emb) for emb in embeddings]

    weights = []
    outputs = []
    for query in q:
        scores = [dot(query, key) / math.sqrt(dim) for key in k]
        attn = softmax(scores)
        weights.append(attn)
        outputs.append([sum(attn[i] * v[i][j] for i in range(len(tokens))) for j in range(dim)])

    payload = {
        "tokens": tokens,
        "embedding_shape": [len(tokens), dim],
        "q_shape": [len(q), dim],
        "k_shape": [len(k), dim],
        "v_shape": [len(v), dim],
        "attention_shape": [len(weights), len(weights[0])],
        "attention_weights": weights,
        "output_shape": [len(outputs), dim],
    }
    out_dir = ensure_dir(Path(__file__).resolve().parents[1] / "outputs")
    write_json(out_dir / "attention.json", payload)
    print(
        "week09 "
        f"tokens={len(tokens)} "
        f"dim={dim} "
        f"attention_shape={payload['attention_shape']}"
    )


if __name__ == "__main__":
    main()
