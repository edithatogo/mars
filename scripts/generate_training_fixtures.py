"""Script to generate Python baseline fixtures for training orchestration."""

import json
import numpy as np
from pymars import Earth


def generate_basic_fixture():
    """Generate a basic linear fit baseline fixture."""
    # Simple linear data: y = 2*x + 1
    np.random.seed(42)
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    y = np.array([1.0, 3.0, 5.0, 7.0, 9.0])

    model = Earth(max_terms=5, max_degree=1)
    model.fit(X, y)

    spec = model.get_model_spec()
    spec["metrics"] = {
        "rss": float(model.rss_),
        "mse": float(model.mse_),
        "gcv": float(model.gcv_) if hasattr(model, "gcv_") else None,
    }

    with open("tests/fixtures/training_full_fit_baseline_v1.json", "w") as f:
        json.dump(spec, f, indent=2)
    print("Created training_full_fit_baseline_v1.json")


def generate_sample_weight_fixture():
    """Generate a sample-weight baseline fixture."""
    np.random.seed(42)
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    y = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
    sample_weight = np.array([1.0, 2.0, 1.0, 2.0, 1.0])

    model = Earth(max_terms=5, max_degree=1)
    model.fit(X, y, sample_weight=sample_weight)

    spec = model.get_model_spec()
    spec["X"] = X.tolist()
    spec["y"] = y.tolist()
    spec["sample_weight"] = sample_weight.tolist()
    spec["metrics"] = {
        "rss": float(model.rss_),
        "mse": float(model.mse_),
    }

    with open("tests/fixtures/training_sample_weight_baseline_v1.json", "w") as f:
        json.dump(spec, f, indent=2)
    print("Created training_sample_weight_baseline_v1.json")


def generate_interaction_fixture():
    """Generate an interaction term baseline fixture."""
    np.random.seed(42)
    # 2D data with interaction: y = x1 + x2 + x1*x2
    X = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
        ]
    )
    y = np.array([0.0, 1.0, 1.0, 2.0, 3.0])

    model = Earth(max_terms=10, max_degree=2)
    model.fit(X, y)

    spec = model.get_model_spec()
    spec["metrics"] = {
        "rss": float(model.rss_) if hasattr(model, "rss_") else None,
        "mse": float(model.mse_) if hasattr(model, "mse_") else None,
    }

    with open("tests/fixtures/training_interaction_baseline_v1.json", "w") as f:
        json.dump(spec, f, indent=2)
    print("Created training_interaction_baseline_v1.json")


if __name__ == "__main__":
    generate_basic_fixture()
    generate_sample_weight_fixture()
    generate_interaction_fixture()
    print("All baseline fixtures generated successfully!")
