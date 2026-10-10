import numpy as np


def activation(z):
    return np.sqrt(0.9) * z + (z**3 - 3 * z) / 6


def activation_derivative(z):
    return np.sqrt(0.9) + (z**2 - 1) / 2


def generate_data(
    seed,
    t_train,
    t_validation,
    t_test,
    input_dimension,
    noise_std,
):
    rng = np.random.default_rng(seed)

    latent_direction = rng.normal(size=input_dimension)
    latent_direction /= np.linalg.norm(latent_direction)

    n_total = t_train + t_validation + t_test
    features = rng.normal(size=(n_total, input_dimension))
    signals = activation(features @ latent_direction)
    returns = signals + rng.normal(scale=noise_std, size=n_total)

    train_end = t_train
    validation_end = t_train + t_validation

    return (
        features[:train_end],
        returns[:train_end],
        features[train_end:validation_end],
        returns[train_end:validation_end],
        signals[train_end:validation_end],
        features[validation_end:],
        returns[validation_end:],
        signals[validation_end:],
        latent_direction,
    )
