"""Week 03: a tiny scalar autograd demo.

This mirrors the PyTorch mental model: build a graph, call backward, then
update parameters and clear gradients. It is intentionally scalar-only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dl_playground.tiny import ensure_dir, make_linear_regression_data, mean, write_json


class Value:
    def __init__(self, data: float, children: tuple["Value", ...] = (), op: str = "") -> None:
        self.data = data
        self.grad = 0.0
        self._prev = set(children)
        self._backward = lambda: None
        self.op = op

    def __add__(self, other: "Value" | float) -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def backward() -> None:
            self.grad += out.grad
            other.grad += out.grad

        out._backward = backward
        return out

    def __mul__(self, other: "Value" | float) -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def backward() -> None:
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = backward
        return out

    def __sub__(self, other: "Value" | float) -> "Value":
        return self + (other * -1 if isinstance(other, Value) else -other)

    def __pow__(self, power: int | float) -> "Value":
        out = Value(self.data**power, (self,), f"**{power}")

        def backward() -> None:
            self.grad += power * (self.data ** (power - 1)) * out.grad

        out._backward = backward
        return out

    def backward(self) -> None:
        topo = []
        visited = set()

        def build(node: "Value") -> None:
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build(child)
                topo.append(node)

        build(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


def zero_grad(*params: Value) -> None:
    for param in params:
        param.grad = 0.0


def train(epochs: int, lr: float) -> dict[str, float | int]:
    xs, ys = make_linear_regression_data(n=30, seed=7)
    w = Value(0.0)
    b = Value(0.0)
    losses = []
    for _ in range(epochs):
        zero_grad(w, b)
        sample_losses = []
        for x, y in zip(xs, ys, strict=True):
            pred = w * x + b
            sample_losses.append((pred - y) ** 2)
        loss = sample_losses[0]
        for item in sample_losses[1:]:
            loss = loss + item
        loss = loss * (1 / len(sample_losses))
        loss.backward()
        w.data -= lr * w.grad
        b.data -= lr * b.grad
        losses.append(loss.data)
    return {
        "epochs": epochs,
        "learning_rate": lr,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "w": w.data,
        "b": b.data,
        "w_grad_last_epoch": w.grad,
        "b_grad_last_epoch": b.grad,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=0.01)
    args = parser.parse_args()

    metrics = train(args.epochs, args.lr)
    out_dir = ensure_dir(Path(__file__).resolve().parents[1] / "outputs")
    write_json(out_dir / "metrics.json", metrics)
    print(
        "week03 "
        f"initial_loss={metrics['initial_loss']:.4f} "
        f"final_loss={metrics['final_loss']:.4f} "
        f"w={metrics['w']:.3f} "
        f"b={metrics['b']:.3f}"
    )


if __name__ == "__main__":
    main()
