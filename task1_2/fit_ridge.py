import numpy as np


def fit_ridge(x_train, y_train, z):
    n_samples, n_features = x_train.shape

    if z == 0:
        return np.linalg.pinv(x_train) @ y_train

    if n_features <= n_samples:
        return np.linalg.solve(
            x_train.T @ x_train + z * n_samples * np.eye(n_features),
            x_train.T @ y_train,
        )

    return x_train.T @ np.linalg.solve(
        x_train @ x_train.T + z * n_samples * np.eye(n_samples),
        y_train,
    )
