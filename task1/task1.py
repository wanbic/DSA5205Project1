from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from generate_data import generate_data
from fit_ridge import fit_ridge
from plot_results import plot_results
from test_metrics import calculate_metrics

# read config
config_path = Path(__file__).with_name("task1.yaml")
with config_path.open("r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

# create a timestamped result dir
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
result_dir = Path(__file__).parent / f"result_{timestamp}"
result_dir.mkdir()

# parse parameters
seed = config["seed"]
n_replications = config["n_replications"]
t_train = config["t_train"]
t_test = config["t_test"]
c = config["true_complexity"]
b_star = config["beta_norm_squared"]
noise_std = config["noise_std"]
cq_grid = config["observed_complexity_grid"]
z_grid = config["ridge_z_grid"]

p = int(c * t_train)
results = []



# main experiment
for replication in range(n_replications):
    current_seed = seed + replication

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
        x_train_observed = x_train[:, selected_features]
        x_test_observed = x_test[:, selected_features]

        for z in z_grid:
            beta_hat = fit_ridge(x_train_observed, y_train, z)
            metrics = calculate_metrics(x_test_observed, y_test, beta_hat)

            results.append(
                {
                    "replication": replication,
                    "seed": current_seed,
                    "cq": cq,
                    "q": q,
                    "p1": p1,
                    "z": z,
                    **metrics,
                }
            )


# save results and plots
results = pd.DataFrame(results)
results["beta_norm"] = results["beta_norm_squared"] ** 0.5
results.to_csv(result_dir / "results.csv", index=False)
plot_results(results, z_grid, result_dir)
