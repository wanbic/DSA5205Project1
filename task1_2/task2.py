from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import lasso_path

from generate_data import generate_data
from plot_results import plot_lasso_results
from test_metrics import calculate_metrics


# Read the shared Task 1 and Task 2 configuration.
config_path = Path(__file__).with_name("task2.yaml")
with config_path.open("r", encoding="utf-8") as file:
    config = yaml.safe_load(file)


# Keep Task 2 outputs separate from the Ridge results.
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
result_dir = Path(__file__).parent / f"task2_result_{timestamp}"
result_dir.mkdir()


# Shared DGP, seeds, train/test split, feature permutation, and cq grid.
seed = config["seed"]
n_replications = config["n_replications"]
t_train = config["t_train"]
t_test = config["t_test"]
c = config["true_complexity"]
b_star = config["beta_norm_squared"]
noise_std = config["noise_std"]
cq_grid = config["observed_complexity_grid"]

# Lasso-specific settings. sklearn minimizes
# (1 / (2 * n)) * ||y - X beta||_2^2 + alpha * ||beta||_1.
alpha_grid = config["lasso_alpha_grid"]
max_iter = config["lasso_max_iter"]
tolerance = config["lasso_tolerance"]

p = int(c * t_train)
results = []


for replication in range(n_replications):
    current_seed = seed + replication

    # The identical generator call used by task1.py reproduces the same data,
    # split, true beta, and fixed feature permutation for this replication.
    x_train, y_train, x_test, y_test, permutation = generate_data(
        current_seed,
        t_train,
        t_test,
        p,
        b_star,
        noise_std,
    )

    for cq in cq_grid:
        p1 = int(cq * t_train)
        q = p1 / p
        selected_features = permutation[:p1]
        # Coordinate descent is fastest with a Fortran-contiguous design.
        # Construct it once rather than letting every alpha fit copy X.
        x_train_observed = np.asfortranarray(
            x_train[:, selected_features]
        )
        x_test_observed = x_test[:, selected_features]

        # Solve the complete regularization path in one call. The solver sorts
        # alpha from large to small and warm-starts each denser solution from
        # the preceding sparse solution. Avoid a P1-by-P1 Gram matrix because
        # P1 reaches 5,000 while T_train is only 500.
        path_alphas, path_coefficients, _ = lasso_path(
            x_train_observed,
            y_train,
            alphas=np.asarray(alpha_grid),
            precompute=False,
            copy_X=False,
            max_iter=max_iter,
            tol=tolerance,
            selection="random",
            random_state=current_seed,
        )

        for alpha, beta_hat in zip(
            path_alphas,
            path_coefficients.T,
        ):
            metrics = calculate_metrics(
                x_test_observed,
                y_test,
                beta_hat,
            )

            results.append(
                {
                    "replication": replication,
                    "seed": current_seed,
                    "cq": cq,
                    "q": q,
                    "p1": p1,
                    "alpha": alpha,
                    **metrics,
                }
            )


results = pd.DataFrame(results)
results["beta_norm"] = results["beta_norm_squared"] ** 0.5
results.to_csv(result_dir / "results.csv", index=False)
plot_lasso_results(results, alpha_grid, result_dir)
