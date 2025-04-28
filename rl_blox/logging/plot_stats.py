import matplotlib.pyplot as plt

from .logger import StandardLogger


def plot_stats(
    logger: StandardLogger,
    y_key: str,
    x_key: str = "step",
    ax: plt.Axes | None = None,
    legend: bool = True,
):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

    x, y = logger.get_stat(key=y_key, x_key=x_key)
    ax.plot(x, y, label=f"{logger.algorithm_name} {logger.start_time:.2f}")
    ax.set_title(f"Environment: {logger.env_name}")
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    if legend:
        ax.legend(loc="best")
    plt.show()
