import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from tasks.cliff_problem.cliff_game import Action


def use_style(func):
    def wrapper(*args, style='seaborn-v0_8-darkgrid', **kwargs):
        with plt.style.context(style):
            res = func(*args, **kwargs)
        return res
    return wrapper


def plot_q_values(q_values: np.ndarray, actions: list[Action], title: str = ""):
    if q_values.ndim != 3:
        raise ValueError(f"Expected a 3D numpy array; got dimensions: {q_values.shape}")

    vmin = q_values.min()
    vmax = q_values.max()

    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    for i, action in enumerate(actions):
        ax = axes[i // 2, i % 2]
        im = ax.imshow(q_values[:, :, i], vmin=vmin, vmax=vmax, cmap='viridis')
        ax.set_title(action.name)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.83, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    fig.suptitle(title)
    return fig


@use_style
def plot_policy(u: npt.NDArray[np.floating], v: npt.NDArray[np.floating], trajectory: list[tuple[int, int]],
                color='navy', title: str = ""):
    fig, axes = plt.subplots(2, 1, sharex='all', sharey='all', figsize=(10, 6))

    ny, nx = u.shape

    for ax in axes:
        ax.yaxis.set_inverted(True)
        ax.set_aspect('equal')
        ax.set_xticks(np.arange(nx + 1) - 0.5)
        ax.set_yticks(np.arange(ny) - 0.5)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlim(-0.5, nx - 0.5)
        ax.set_ylim(ny - 0.5, -0.5)
        ax.tick_params(axis='both', which='major', length=0)

    axes[0].quiver(u, v, pivot='mid', color=color)
    axes[0].set_title("Policy")

    y_coords, x_coords = list(zip(*trajectory))
    axes[1].plot(x_coords, y_coords, marker='o', lw=0.5, color=color, alpha=0.5)

    mkw = dict(lw=0, markerfacecolor='none', markeredgecolor=color, markersize=10, zorder=0)
    axes[1].plot([x_coords[0]], [y_coords[0]], marker='D', **mkw, label="Start")
    axes[1].plot([x_coords[-1]], [y_coords[-1]], marker='s', **mkw, label="Finish")
    axes[1].legend(frameon=True, framealpha=0.5, fancybox=True)

    axes[1].set_title("Greedy trajectory")

    fig.suptitle(title)
