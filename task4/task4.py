from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from generate_data import activation, activation_derivative, generate_data
from plot_results import plot_financial_results, plot_results
from test_metrics import calculate_metrics


# read config
config_path = Path(__file__).with_name("task4.yaml")
with config_path.open("r", encoding="utf-8") as file:
    config = yaml.safe_load(file)


# create a timestamped result dir
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
result_dir = Path(__file__).parent / f"task4_result_{timestamp}"
result_dir.mkdir()


# parse parameters
seed = config["seed"]
n_replications = config["n_replications"]
input_dimension = config["input_dimension"]
hidden_units = config["hidden_units"]
sample_ratio = config["sample_ratio"]
t_validation = config["t_validation"]
t_test = config["t_test"]
noise_std_grid = config["noise_std_grid"]
batch_size = config["batch_size"]
learning_rate = config["learning_rate"]
max_steps = config["max_steps"]
n_checkpoints = config["n_checkpoints"]

t_train = int(sample_ratio * hidden_units * input_dimension)
checkpoints = np.unique(
    np.rint(np.geomspace(1, max_steps, n_checkpoints)).astype(int)
)
checkpoints = np.insert(checkpoints, 0, 0)
checkpoint_set = set(checkpoints)
results = []


def predict(features, first_layer, second_layer):
    return activation(features @ first_layer.T) @ second_layer / hidden_units


def save_metrics(
    replication,
    current_seed,
    noise_std,
    step,
    first_layer,
    second_layer,
    x_train,
    y_train,
    train_signal,
    x_validation,
    y_validation,
    validation_signal,
    x_test,
    y_test,
    test_signal,
):
    train_metrics = calculate_metrics(
        y_train,
        predict(x_train, first_layer, second_layer),
        train_signal,
    )
    validation_metrics = calculate_metrics(
        y_validation,
        predict(x_validation, first_layer, second_layer),
        validation_signal,
    )
    test_metrics = calculate_metrics(
        y_test,
        predict(x_test, first_layer, second_layer),
        test_signal,
    )

    results.append(
        {
            "replication": replication,
            "seed": current_seed,
            "noise_std": noise_std,
            "input_dimension": input_dimension,
            "hidden_units": hidden_units,
            "sample_ratio": sample_ratio,
            "t_train": t_train,
            "step": step,
            "scaled_step": step / hidden_units,
            "train_half_mse": train_metrics["half_mse"],
            "validation_half_mse": validation_metrics["half_mse"],
            "test_half_mse": test_metrics["half_mse"],
            "generalization_gap": (
                test_metrics["half_mse"] - train_metrics["half_mse"]
            ),
            "second_layer_complexity": np.mean(np.abs(second_layer)),
            "signal_alignment": test_metrics["signal_alignment"],
            "test_r2_paper": test_metrics["r2_paper"],
            "test_expected_timing_return": test_metrics[
                "expected_timing_return"
            ],
            "test_sharpe_paper": test_metrics["sharpe_paper"],
            "test_sharpe_variance": test_metrics["sharpe_variance"],
        }
    )


# main experiment
for replication in range(n_replications):
    current_seed = seed + replication

    for noise_std in noise_std_grid:
        (
            x_train,
            y_train,
            x_validation,
            y_validation,
            validation_signal,
            x_test,
            y_test,
            test_signal,
            latent_direction,
        ) = generate_data(
            current_seed,
            t_train,
            t_validation,
            t_test,
            input_dimension,
            noise_std,
        )
        train_signal = activation(x_train @ latent_direction)

        rng = np.random.default_rng(current_seed + 10000)
        first_layer = rng.normal(
            scale=1 / np.sqrt(input_dimension),
            size=(hidden_units, input_dimension),
        )
        first_layer /= np.linalg.norm(first_layer, axis=1, keepdims=True)
        second_layer = np.ones(hidden_units)

        save_metrics(
            replication,
            current_seed,
            noise_std,
            0,
            first_layer,
            second_layer,
            x_train,
            y_train,
            train_signal,
            x_validation,
            y_validation,
            validation_signal,
            x_test,
            y_test,
            test_signal,
        )

        for step in range(1, max_steps + 1):
            batch_indices = rng.integers(0, t_train, size=batch_size)
            x_batch = x_train[batch_indices]
            y_batch = y_train[batch_indices]

            hidden_input = x_batch @ first_layer.T
            hidden_output = activation(hidden_input)
            predictions = hidden_output @ second_layer / hidden_units
            residuals = predictions - y_batch

            second_layer_gradient = (
                hidden_output.T @ residuals / (batch_size * hidden_units)
            )
            hidden_gradient = (
                residuals[:, None]
                * second_layer[None, :]
                * activation_derivative(hidden_input)
                / hidden_units
            )
            first_layer_gradient = hidden_gradient.T @ x_batch / batch_size

            second_layer -= learning_rate * second_layer_gradient
            first_layer -= learning_rate * first_layer_gradient
            first_layer /= np.linalg.norm(first_layer, axis=1, keepdims=True)

            if step in checkpoint_set:
                save_metrics(
                    replication,
                    current_seed,
                    noise_std,
                    step,
                    first_layer,
                    second_layer,
                    x_train,
                    y_train,
                    train_signal,
                    x_validation,
                    y_validation,
                    validation_signal,
                    x_test,
                    y_test,
                    test_signal,
                )


# save results and plots
results = pd.DataFrame(results)
results.to_csv(result_dir / "results.csv", index=False)

best_checkpoints = results.loc[
    results.groupby(["replication", "noise_std"])[
        "validation_half_mse"
    ].idxmin()
]
best_checkpoints.to_csv(result_dir / "best_checkpoints.csv", index=False)

plot_results(results, result_dir)
plot_financial_results(results, result_dir)
