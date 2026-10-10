from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import lasso_path

from plot_results import plot_results
from test_metrics import calculate_metrics


# read config
config_path = Path(__file__).with_name("lasso.yaml")
with config_path.open("r", encoding="utf-8") as file:
    config = yaml.safe_load(file)


# read selected data set
dataset = config["dataset"]
data_dir = Path(__file__).parent / "data"
train_data = pd.read_csv(data_dir / f"pair{dataset}_train.csv")
test_data = pd.read_csv(data_dir / f"pair{dataset}_test_features.csv")

feature_names = train_data.columns[1:-1].tolist()
if test_data.columns.tolist() != ["t", *feature_names]:
    raise ValueError("Train and test features do not match")

x = train_data[feature_names].to_numpy()
y = train_data["return"].to_numpy()
x_test = test_data[feature_names].to_numpy()


# use the last part of the training file for validation
split = int(len(y) * (1 - config["validation_fraction"]))
x_train = np.asfortranarray(x[:split])
y_train = y[:split]
x_validation = x[split:]
y_validation = y[split:]

alpha_grid = config["lasso_alpha_grid"]
max_iter = config["lasso_max_iter"]
tolerance = config["lasso_tolerance"]
seed = config["seed"]


# select Lasso alpha on the validation block
path_alphas, path_coefficients, _ = lasso_path(
    x_train,
    y_train,
    alphas=np.asarray(alpha_grid),
    precompute=False,
    copy_X=False,
    max_iter=max_iter,
    tol=tolerance,
    selection="random",
    random_state=seed,
)

results = []
for alpha, beta_hat in zip(path_alphas, path_coefficients.T):
    metrics = calculate_metrics(x_validation, y_validation, beta_hat)
    results.append({"alpha": alpha, **metrics})

results = pd.DataFrame(results)
results["beta_norm"] = results["beta_norm_squared"] ** 0.5
best_alpha = results.loc[
    results[config["selection_metric"]].idxmax(), "alpha"
]


# refit on all training rows and predict the public test data
_, coefficients, _ = lasso_path(
    np.asfortranarray(x),
    y,
    alphas=np.asarray([best_alpha]),
    precompute=False,
    copy_X=False,
    max_iter=max_iter,
    tol=tolerance,
    selection="random",
    random_state=seed,
)
predictions = x_test @ coefficients[:, 0]


# save results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
result_dir = Path(__file__).parent / f"{dataset}_lasso_result_{timestamp}"
result_dir.mkdir()

results.to_csv(result_dir / "validation_results.csv", index=False)
plot_results(results, "alpha", result_dir)

pd.DataFrame(
    {"t": test_data["t"], "yhat": predictions}
).to_csv(
    result_dir / f"predictions_{dataset}.csv",
    index=False,
)
