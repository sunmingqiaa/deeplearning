"""Check the local learning environment.

The current labs are written to run with the Python standard library. This
script reports optional packages that become useful when moving to PyTorch.
"""

from __future__ import annotations

import importlib
import platform


OPTIONAL_MODULES = [
    "numpy",
    "torch",
    "torchvision",
    "sklearn",
    "matplotlib",
    "fastapi",
    "pytest",
]


def main() -> None:
    print(f"python: {platform.python_version()}")
    print(f"platform: {platform.platform()}")
    for name in OPTIONAL_MODULES:
        try:
            module = importlib.import_module(name)
        except Exception as exc:  # pragma: no cover - diagnostic script
            print(f"{name}: missing ({exc.__class__.__name__}: {exc})")
        else:
            version = getattr(module, "__version__", "installed")
            print(f"{name}: {version}")


if __name__ == "__main__":
    main()
