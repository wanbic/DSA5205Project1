import matplotlib.pyplot as plt


PLOTS = {
    "r2_paper": (r"$R^2$", (-0.3, 0.3), [-0.3, -0.2, -0.1, 0, 0.1, 0.2]),
    "beta_norm": (r"$\|\hat{\beta}\|$", (0, 6), [0, 1, 2, 3, 4, 5, 6]),
    "expected_timing_return": (
        "Expected Return",
        (0, 0.02),
        [0, 0.01, 0.02],
    ),
    "sharpe_paper": (
        "Sharpe Ratio",
        (0, 0.06),
        [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
    ),
}


def _plot_results(
    results,
    parameter_name,
    parameter_grid,
    result_dir,
    label_parameter,
    ridgeless_parameter=None,
):
    mean_results = (
        results.groupby(["cq", parameter_name])[
            [
                "r2_paper",
                "expected_timing_return",
                "sharpe_paper",
                "beta_norm",
            ]
        ]
        .mean()
        .reset_index()
    )

    for metric, (title, y_limits, y_ticks) in PLOTS.items():
        plt.figure(figsize=(8, 5))

        for parameter_value in parameter_grid:
            curve = mean_results[
                mean_results[parameter_name] == parameter_value
            ]
            is_ridgeless = parameter_value == ridgeless_parameter
            label = (
                "Ridgeless"
                if is_ridgeless
                else f"{label_parameter} = {parameter_value:g}"
            )
            color = "black" if is_ridgeless else None
            x_values = [0, *curve["cq"].tolist()]
            y_values = [0, *curve[metric].tolist()]
            plt.plot(x_values, y_values, label=label, color=color)

        plt.axvline(
            1, color="lightgray", linestyle="--", linewidth=1, label="cq = 1"
        )
        plt.xlim(0, 10)
        plt.ylim(y_limits)
        plt.xticks([0, 2, 4, 6, 8, 10])
        plt.yticks(y_ticks)
        plt.xlabel(r"$cq$")
        plt.title(title)
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(result_dir / f"{metric}_vs_cq.png", dpi=300)
        plt.close()


def plot_results(results, z_grid, result_dir):
    _plot_results(
        results,
        parameter_name="z",
        parameter_grid=z_grid,
        result_dir=result_dir,
        label_parameter="z",
        ridgeless_parameter=0,
    )


def plot_lasso_results(results, alpha_grid, result_dir):
    _plot_results(
        results,
        parameter_name="alpha",
        parameter_grid=alpha_grid,
        result_dir=result_dir,
        label_parameter=r"$\alpha$",
    )
