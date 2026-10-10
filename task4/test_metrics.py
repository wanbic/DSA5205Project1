import numpy as np


def calculate_metrics(y, predictions, true_signal):
    errors = y - predictions
    timing_returns = predictions * y
    timing_second_moment = np.mean(timing_returns**2)
    timing_std = np.std(timing_returns)

    prediction_std = np.std(predictions)
    signal_std = np.std(true_signal)
    signal_alignment = (
        np.corrcoef(predictions, true_signal)[0, 1] ** 2
        if prediction_std > 0 and signal_std > 0
        else 0.0
    )

    return {
        "half_mse": 0.5 * np.mean(errors**2),
        "r2_paper": 1 - np.mean(errors**2) / np.mean(y**2),
        "expected_timing_return": np.mean(timing_returns),
        "sharpe_paper": (
            np.mean(timing_returns) / np.sqrt(timing_second_moment)
            if timing_second_moment > 0
            else 0.0
        ),
        "sharpe_variance": (
            np.mean(timing_returns) / timing_std
            if timing_std > 0
            else 0.0
        ),
        "signal_alignment": signal_alignment,
    }
