from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.neural_network import MLPRegressor

from plot_results import plot_results


# read config
config_path = Path(__file__).with_name("nn.yaml")
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


# select the hidden-layer size on the validation block
results = []
for hidden_units in config["hidden_units_grid"]:
    model = MLPRegressor(
        hidden_layer_sizes=(hidden_units,),
        alpha=config["alpha"],
        learning_rate_init=config["learning_rate_init"],
        max_iter=config["max_iter"],
        random_state=config["seed"],
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_validation)
    timing_returns = predictions * y_validation

    results.append(
        {
            "hidden_units": hidden_units,
            "r2_paper": 1
            - np.mean((y_validation - predictions) ** 2)
            / np.mean(y_validation**2),
            "expected_timing_return": np.mean(timing_returns),
            "sharpe_paper": np.mean(timing_returns)
            / np.sqrt(np.mean(timing_returns**2)),
            "sharpe_variance": np.mean(timing_returns)
            / np.std(timing_returns),
            "beta_norm": np.sqrt(
                sum(np.sum(weights**2) for weights in model.coefs_)
            ),
        }
    )

results = pd.DataFrame(results)
best_hidden_units = int(
    results.loc[
        results[config["selection_metric"]].idxmax(),
        "hidden_units",
    ]
)


# refit on all training rows and predict the public test data
model = MLPRegressor(
    hidden_layer_sizes=(best_hidden_units,),
    alpha=config["alpha"],
    learning_rate_init=config["learning_rate_init"],
    max_iter=config["max_iter"],
    random_state=config["seed"],
)
model.fit(x, y)
predictions = model.predict(x_test)


# save results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
result_dir = Path(__file__).parent / f"{dataset}_nn_result_{timestamp}"
result_dir.mkdir()

results.to_csv(result_dir / "validation_results.csv", index=False)
plot_results(results, "hidden_units", result_dir)

pd.DataFrame(
    {"t": test_data["t"], "yhat": predictions}
).to_csv(
    result_dir / f"predictions_{dataset}.csv",
    index=False,
)
