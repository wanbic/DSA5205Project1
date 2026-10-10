import matplotlib.pyplot as plt


PLOTS = {
    "r2_paper": r"Validation $R^2$",
    "expected_timing_return": "Validation Expected Return",
    "sharpe_paper": "Validation Sharpe Ratio",
    "beta_norm": r"Validation $\|\hat{\beta}\|$",
}


def plot_results(results, parameter_name, result_dir):
    results = results.sort_values(parameter_name)

    for metric, title in PLOTS.items():
        plt.figure(figsize=(8, 5))
        plt.plot(results[parameter_name], results[metric], marker="o")
        plt.xlabel(parameter_name)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(
            result_dir / f"validation_{metric}_vs_{parameter_name}.png",
            dpi=300,
        )
        plt.close()
