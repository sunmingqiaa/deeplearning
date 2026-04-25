"""Week 10: matrix-factorization recommender on synthetic interactions."""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dl_playground.tiny import dot, ensure_dir, mean, write_json


def make_interactions(users: int, items: int, dim: int, seed: int) -> list[tuple[int, int, float]]:
    rng = random.Random(seed)
    true_users = [[rng.gauss(0, 1) for _ in range(dim)] for _ in range(users)]
    true_items = [[rng.gauss(0, 1) for _ in range(dim)] for _ in range(items)]
    interactions = []
    for user in range(users):
        sampled_items = rng.sample(range(items), k=min(items, 8))
        for item in sampled_items:
            score = dot(true_users[user], true_items[item]) / dim + rng.gauss(0, 0.1)
            rating = max(1.0, min(5.0, 3.0 + score))
            interactions.append((user, item, rating))
    rng.shuffle(interactions)
    return interactions


def train(epochs: int, lr: float, seed: int) -> dict[str, object]:
    rng = random.Random(seed)
    users, items, dim = 12, 18, 5
    data = make_interactions(users, items, dim, seed)
    user_emb = [[rng.gauss(0, 0.1) for _ in range(dim)] for _ in range(users)]
    item_emb = [[rng.gauss(0, 0.1) for _ in range(dim)] for _ in range(items)]
    user_bias = [0.0] * users
    item_bias = [0.0] * items
    global_bias = mean(r for _, _, r in data)
    history = []
    for _ in range(epochs):
        rng.shuffle(data)
        losses = []
        for user, item, rating in data:
            pred = global_bias + user_bias[user] + item_bias[item] + dot(user_emb[user], item_emb[item])
            err = pred - rating
            losses.append(err * err)
            old_user = user_emb[user][:]
            for j in range(dim):
                user_emb[user][j] -= lr * err * item_emb[item][j]
                item_emb[item][j] -= lr * err * old_user[j]
            user_bias[user] -= lr * err
            item_bias[item] -= lr * err
        history.append(math.sqrt(mean(losses)))
    recommendations = []
    target_user = 0
    for item in range(items):
        pred = global_bias + user_bias[target_user] + item_bias[item] + dot(user_emb[target_user], item_emb[item])
        recommendations.append({"item": item, "score": pred})
    recommendations.sort(key=lambda row: row["score"], reverse=True)
    return {
        "epochs": epochs,
        "learning_rate": lr,
        "users": users,
        "items": items,
        "embedding_dim": dim,
        "initial_rmse": history[0],
        "final_rmse": history[-1],
        "top5_user0": recommendations[:5],
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
        "week10 "
        f"initial_rmse={metrics['initial_rmse']:.4f} "
        f"final_rmse={metrics['final_rmse']:.4f} "
        f"top_item={metrics['top5_user0'][0]['item']}"
    )


if __name__ == "__main__":
    main()
