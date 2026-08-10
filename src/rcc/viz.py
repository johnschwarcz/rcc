"""Plotting. Every function returns a figure; none call ``show()``.
Needs the viz extra: pip install "rcc[viz]"
Colours are coggrid's, so a belief and a posterior look the same in both repos.
"""
from collections.abc import Mapping, Sequence
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

__all__ = ["CHAIN", "IDEAL", "MUTED", "TRUTH",
    "plot_belief_accumulation", "plot_policy", "plot_training"]

CHAIN = "#7b52ab"  # what the chain learned
IDEAL = "#1f77b4"  # what an exact observer would have concluded
TRUTH = "#78c765"  # the truth
MUTED = "#A9A9A9"  # everything else

def _numpy(x) -> np.ndarray:
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

# ---------------------------------------------------------------------- stage 2
def plot_belief_accumulation(belief, goal_ind, goal_value, posterior=None,
    episode: int = 0, *, fig: Figure | None = None, figsize=(9.5, 3.6)) -> Figure:
    """Draw one episode's belief filling in, step by step.
    belief, posterior: (n_episodes, n_steps, n_contexts, n_realizations)
    goal_ind, goal_value: (n_episodes,); posterior is optional, drawn alongside
    returns: Figure
    """
    belief = _numpy(belief)
    goal = int(_numpy(goal_ind)[episode])
    truth = int(_numpy(goal_value)[episode])

    panels = [("chain", belief[episode, :, goal], CHAIN)]
    if posterior is not None:
        panels.append(("exact posterior", _numpy(posterior)[episode, :, goal], IDEAL))

    fig = fig or plt.figure(figsize=figsize)
    axes = fig.subplots(1, len(panels) + 1, width_ratios=[1] * len(panels) + [0.85])

    for ax, (name, values, colour) in zip(axes[:-1], panels, strict=True):
        ax.imshow(values.T, aspect="auto", origin="lower", cmap="magma",
            vmin=0, vmax=1, interpolation="nearest")
        ax.axhline(truth, color=TRUTH, linewidth=1.8, linestyle="--")
        ax.set_xlabel("step")
        ax.set_ylabel("realization")
        ax.set_yticks(range(belief.shape[-1]))  # Realizations are categories.
        ax.set_title(name, fontsize=10, color=colour)

    ax = axes[-1]  # The image panels above, collapsed onto the true realization.
    for name, values, colour in panels:
        ax.plot(values[:, truth], color=colour, linewidth=1.8, label=name)
    ax.axhline(1 / belief.shape[-1], color=MUTED, linestyle=":", linewidth=1.2,
        label="chance")
    ax.set_xlabel("step")
    ax.set_ylabel("belief in the truth")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("evidence accumulating", fontsize=10)
    fig.tight_layout()
    return fig

# --------------------------------------------------------------------- training
_SERIES_STYLES = (("ideal", IDEAL, "--"), ("chance", MUTED, ":"))

def _series_style(name: str) -> tuple[str, str]:
    """Label substring --> (colour, linestyle). First match wins."""
    lowered = name.lower()
    for token, colour, dash in _SERIES_STYLES:
        if token in lowered:
            return colour, dash
    return CHAIN, "-"

def plot_training(history: Mapping[str, Sequence[float]], *, smooth: int = 1,
    ylabel: str = "goal accuracy", fig: Figure | None = None,
    figsize=(6.5, 4.0)) -> Figure:
    """Plot every series in history against training iteration.
    history: label --> one value per iteration
    smooth: width of a centred moving average, 1 disabling it
    returns: Figure
    """
    fig = fig or plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    for name, series in history.items():
        values = np.asarray(series, dtype=float)
        if smooth > 1 and values.size >= smooth:
            values = np.convolve(values, np.ones(smooth) / smooth, mode="valid")
        colour, dash = _series_style(name)
        ax.plot(values, label=name, linewidth=1.6, color=colour, linestyle=dash)

    ax.set_xlabel("training iteration")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig

# ---------------------------------------------------------------------- stage 4
def plot_policy(policy, landscape=None, episode: int = 0, *,
    fig: Figure | None = None, figsize=(7.0, 3.4)) -> Figure:
    """Compare the controller's policy with the value it was chasing.
    policy, landscape: (n_episodes, n_realizations, n_realizations), so only
    n_contexts == 2, where the joint realization grid is a plane.
    returns: Figure
    """
    policy = _numpy(policy)
    if policy.ndim != 3:
        raise ValueError("plot_policy draws a two-variable grid, so policy must have "
            f"shape (n_episodes, n_realizations, n_realizations); got {policy.shape}")

    panels = [("policy", policy[episode], "magma")]
    if landscape is not None:
        panels.append(("true value", _numpy(landscape)[episode], "viridis"))

    fig = fig or plt.figure(figsize=figsize)
    axes = fig.subplots(1, len(panels), squeeze=False)[0]
    for ax, (name, values, cmap) in zip(axes, panels, strict=True):
        image = ax.imshow(values, origin="lower", cmap=cmap, interpolation="nearest")
        best = np.unravel_index(values.argmax(), values.shape)
        ax.plot(best[1], best[0], marker="*", markersize=14, color=TRUTH,
            markeredgecolor="white", markeredgewidth=0.8, clip_on=False)
        ax.set_xlabel("realization of variable 1")
        ax.set_ylabel("realization of variable 0")
        ax.set_title(name, fontsize=10)
        fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig
