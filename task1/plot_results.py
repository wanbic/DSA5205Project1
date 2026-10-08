import matplotlib.pyplot as plt


def plot_results(results, z_grid, result_dir):
    mean_results = (
        results.groupby(["cq", "z"])[
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

    plots = {
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

    for metric, (title, y_limits, y_ticks) in plots.items():
        plt.figure(figsize=(8, 5))

        for z in z_grid:
            curve = mean_results[mean_results["z"] == z]
            label = "Ridgeless" if z == 0 else f"z = {z}"
            color = "black" if z == 0 else None
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
