"""Small profiling harness for Scalene and similar tools."""

from __future__ import annotations

import numpy as np

from pymars import Earth


def main() -> None:
    np.random.seed(42)
    x_train = np.random.rand(500, 10)
    y_train = np.sin(x_train[:, 0]) + x_train[:, 1] + x_train[:, 2] ** 2

    model = Earth(max_degree=2)
    model.fit(x_train, y_train)
    model.predict(x_train[:10])


if __name__ == "__main__":
    main()
