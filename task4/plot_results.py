import matplotlib.pyplot as plt


def plot_results(results, result_dir):
    mean_results = (
        results.groupby(["noise_std", "step"])[
            [
                "train_half_mse",
                "validation_half_mse",
                "test_half_mse",
                "generalization_gap",
                "second_layer_complexity",
                "signal_alignment",
            ]
        ]
        .mean()
        .reset_index()
    )

    figure, axes = plt.subplots(2, 2, figsize=(10, 7))

    for color_index, noise_std in enumerate(
        sorted(results["noise_std"].unique())
    ):
        curve = mean_results[mean_results["noise_std"] == noise_std]
        x = curve["step"] + 1
        label = rf"$\tau={noise_std:g}$"
        color = f"C{color_index}"

        axes[0, 0].plot(
            x,
            curve["train_half_mse"],
            color=color,
            label=f"Train, {label}",
        )
        axes[0, 0].plot(
            x,
            curve["test_half_mse"],
            color=color,
            linestyle="--",
            label=f"Test, {label}",
        )
        axes[0, 1].plot(
            x, curve["generalization_gap"], color=color, label=label
        )
        axes[1, 0].plot(
            x, curve["second_layer_complexity"], color=color, label=label
        )
        axes[1, 1].plot(
            x, curve["signal_alignment"], color=color, label=label
        )

    axes[0, 0].set_title("Train and test error")
    axes[0, 1].set_title("Generalization gap")
    axes[1, 0].set_title("Second-layer complexity")
    axes[1, 1].set_title("Signal alignment")

    axes[0, 0].set_ylabel("Half MSE")
    axes[0, 1].set_ylabel("Test half MSE - train half MSE")
    axes[1, 0].set_ylabel(r"$\|a\|_1/m$")
    axes[1, 1].set_ylabel(r"$\mathrm{Corr}^2(\hat y, f^*)$")

    for axis in axes.flat:
        axis.set_xscale("log")
        axis.set_xlabel("SGD updates + 1")
        axis.legend(fontsize=8)
        axis.grid(alpha=0.2)

    figure.tight_layout()
    figure.savefig(result_dir / "training_dynamics.png", dpi=300)
    plt.close(figure)


def plot_financial_results(results, result_dir):
    mean_results = (
        results.groupby(["noise_std", "step"])[
            ["test_r2_paper", "test_expected_timing_return", "test_sharpe_paper"]
        ]
        .mean()
        .reset_index()
    )

    figure, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    metrics = {
        "test_r2_paper": r"Test $R^2$",
        "test_expected_timing_return": "Expected timing return",
        "test_sharpe_paper": "Timing Sharpe",
    }

    for noise_std in sorted(results["noise_std"].unique()):
        curve = mean_results[mean_results["noise_std"] == noise_std]
        for axis, (metric, title) in zip(axes, metrics.items()):
            axis.plot(
                curve["step"] + 1,
                curve[metric],
                label=rf"$\tau={noise_std:g}$",
            )
            axis.set_xscale("log")
            axis.set_xlabel("SGD updates + 1")
            axis.set_title(title)
            axis.grid(alpha=0.2)

    axes[0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(result_dir / "financial_metrics.png", dpi=300)
    plt.close(figure)
