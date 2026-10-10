from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from fit_ridge import fit_ridge
from plot_results import plot_results
from test_metrics import calculate_metrics


# read config
config_path = Path(__file__).with_name("ridge.yaml")
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
x_train, x_validation = x[:split], x[split:]
y_train, y_validation = y[:split], y[split:]


# select Ridge z on the validation block
results = []
for z in config["ridge_z_grid"]:
    beta_hat = fit_ridge(x_train, y_train, z)
    metrics = calculate_metrics(x_validation, y_validation, beta_hat)
    results.append({"z": z, **metrics})

results = pd.DataFrame(results)
results["beta_norm"] = results["beta_norm_squared"] ** 0.5
best_z = results.loc[results[config["selection_metric"]].idxmax(), "z"]


# refit on all training rows and predict the public test data
beta_hat = fit_ridge(x, y, best_z)
predictions = x_test @ beta_hat


# save results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
result_dir = Path(__file__).parent / f"{dataset}_ridge_result_{timestamp}"
result_dir.mkdir()

results.to_csv(result_dir / "validation_results.csv", index=False)
plot_results(results, "z", result_dir)

pd.DataFrame(
    {"t": test_data["t"], "yhat": predictions}
).to_csv(
    result_dir / f"predictions_{dataset}.csv",
    index=False,
)
