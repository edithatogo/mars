import matplotlib.pyplot as plt


def plot_basis_functions(model, X):
    """Plot the transformed basis functions."""
    if not hasattr(model, 'basis_'):
        raise ValueError('Model is not fitted')
    X_proc, mask, _ = model.earth_._scrub_input_data(X, X[:,0] if X.ndim>1 else X)
    B = model.earth_._build_basis_matrix(X_proc, model.basis_, mask)
    for i in range(B.shape[1]):
        plt.plot(B[:, i], label=f'basis {i}')
    plt.legend()
    return plt.gca()


def plot_residuals(model, X, y):
    """Plot residuals of the fitted model."""
    preds = model.predict(X)
    plt.scatter(preds, y - preds)
    plt.xlabel('Predicted')
    plt.ylabel('Residual')
    return plt.gca()
