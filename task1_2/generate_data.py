import numpy as np


def generate_data(seed, t_train, t_test, p, b_star, noise_std):
    rng = np.random.default_rng(seed)

    beta = rng.normal(size=p)
    beta = np.sqrt(b_star) * beta / np.linalg.norm(beta)

    signals = rng.normal(size=(t_train + t_test, p))
    returns = signals @ beta + rng.normal(
        scale=noise_std, size=t_train + t_test
    )
    permutation = rng.permutation(p)

    return (
        signals[:t_train],
        returns[:t_train],
        signals[t_train:],
        returns[t_train:],
        permutation,
    )
