import numpy as np


def calculate_metrics(x_test, y_test, beta_hat):
    predictions = x_test @ beta_hat
    timing_returns = predictions * y_test

    r2_paper = 1 - np.mean((y_test - predictions) ** 2) / np.mean(y_test**2)
    expected_timing_return = np.mean(timing_returns)
    sharpe_paper = expected_timing_return / np.sqrt(np.mean(timing_returns**2))
    sharpe_variance = expected_timing_return / np.std(timing_returns)
    beta_norm_squared = np.sum(beta_hat**2)

    return {
        "r2_paper": r2_paper,
        "expected_timing_return": expected_timing_return,
        "sharpe_paper": sharpe_paper,
        "sharpe_variance": sharpe_variance,
        "beta_norm_squared": beta_norm_squared,
    }
