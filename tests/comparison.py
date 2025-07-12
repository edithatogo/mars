import numpy as np
import pymars as pm
import json
import os

# --- Configuration ---
# Hyperparameters to test
HYPERPARAMS = {
    "max_degree": [1, 2],
    "penalty": [0.01, 0.1, 1.0],
}

# Datasets to use for comparison
# (Using a simple synthetic dataset for this example)
def generate_friedman1(n_samples=100, n_features=10, noise=0.1, random_state=0):
    """Generate the Friedman #1 dataset."""
    rng = np.random.RandomState(random_state)
    X = rng.rand(n_samples, n_features)
    y = (10 * np.sin(np.pi * X[:, 0] * X[:, 1]) +
         20 * (X[:, 2] - 0.5) ** 2 +
         10 * X[:, 3] +
         5 * X[:, 4] +
         noise * rng.randn(n_samples))
    return X, y

# --- Helper Functions ---
def save_results(library_name, dataset_name, params, results):
    """Save the results of a model to a JSON file."""
    if not os.path.exists("comparison_results"):
        os.makedirs("comparison_results")
    filename = f"comparison_results/{library_name}_{dataset_name}_{params}.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=4)

def load_results(library_name, dataset_name, params):
    """Load the results of a model from a JSON file."""
    filename = f"comparison_results/{library_name}_{dataset_name}_{params}.json"
    with open(filename, "r") as f:
        return json.load(f)

# --- Model Execution ---
def run_pymars(X, y, params):
    """Fit a pymars model and return the results."""
    model = pm.Earth(feature_importance_type='gcv', **params)
    model.fit(X, y)
    y_hat = model.predict(X)

    # Convert basis functions to string representation for JSON serialization
    basis_functions = [str(b) for b in model.basis_]

    return {
        "predictions": y_hat.tolist(),
        "basis_functions": basis_functions,
        "coefficients": model.coef_.tolist(),
        "feature_importances": model.feature_importances_.tolist(),
    }

def run_pyearth_placeholder(X, y, params):
    """
    Placeholder function for running py-earth.
    In a real scenario, this would be run in a separate environment.
    """
    print("--- Running py-earth (placeholder) ---")
    print("In a real scenario, you would run the py-earth model here and save the results.")
    # As a placeholder, we'll just create an empty results dictionary.
    # In a real test, you would populate this with actual results from py-earth.
    results = {
        "predictions": [],
        "basis_functions": [],
        "coefficients": [],
        "feature_importances": [],
    }
    # In a real test, you would save the results to a file.
    # save_results("pyearth", "friedman1", str(params), results)
    return results


# --- Comparison ---
def compare_results(pymars_results, pyearth_results):
    """Compare the results of the two models."""
    print("\n--- Comparison ---")

    # Compare predictions
    if pymars_results["predictions"] and pyearth_results["predictions"]:
        mse = np.mean((np.array(pymars_results["predictions"]) - np.array(pyearth_results["predictions"])) ** 2)
        print(f"MSE of predictions: {mse}")
    else:
        print("Could not compare predictions (one or both are missing).")

    # Compare basis functions
    print("\nBasis Functions:")
    print("pymars:", pymars_results["basis_functions"])
    print("py-earth:", pyearth_results["basis_functions"])

    # Compare coefficients
    print("\nCoefficients:")
    print("pymars:", pymars_results["coefficients"])
    print("py-earth:", pyearth_results["coefficients"])

    # Compare feature importances
    print("\nFeature Importances:")
    print("pymars:", pymars_results["feature_importances"])
    print("py-earth:", pyearth_results["feature_importances"])


# --- Main Execution ---
if __name__ == "__main__":
    X, y = generate_friedman1()

    for max_degree in HYPERPARAMS["max_degree"]:
        for penalty in HYPERPARAMS["penalty"]:
            params = {"max_degree": max_degree, "penalty": penalty}
            print(f"\n--- Testing with params: {params} ---")

            # Run pymars
            pymars_results = run_pymars(X, y, params)
            save_results("pymars", "friedman1", str(params), pymars_results)

            # Run py-earth (placeholder)
            # In a real scenario, this would be a separate script execution
            run_pyearth_placeholder(X, y, params)

            # Load results and compare
            # In a real scenario, you would load the py-earth results from a file
            # pyearth_results = load_results("pyearth", "friedman1", str(params))
            pyearth_results = {
                "predictions": [], "basis_functions": [], "coefficients": [], "feature_importances": []
            } # Placeholder

            loaded_pymars_results = load_results("pymars", "friedman1", str(params))
            compare_results(loaded_pymars_results, pyearth_results)
