"""Plotting. Every function returns a figure; none call ``show()``.
Needs the viz extra: pip install "rcc[viz]"
Colours are coggrid's, so a belief and a posterior look the same in both repos.
"""
from collections.abc import Mapping, Sequence
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

__all__ = ["CHAIN", "IDEAL", "MUTED", "TRUTH", "plot_belief_accumulation",
    "plot_belief_average", "plot_generalization", "plot_losses", "plot_policy",
    "plot_training"]

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

def _goal_belief(belief, goal_ind) -> np.ndarray:
    """The belief over the one variable being asked about.
    belief: (n_episodes, n_steps, n_contexts, n_realizations)
    returns: (n_episodes, n_steps, n_realizations)
    """
    belief, goal_ind = _numpy(belief), _numpy(goal_ind)
    return belief[np.arange(belief.shape[0]), :, goal_ind]

def plot_belief_average(belief, goal_ind, goal_value, posterior=None, *,
    fig: Figure | None = None, figsize=(9.0, 3.6)) -> Figure:
    """Compare the chain with the exact observer across a whole batch.
    belief, posterior: (n_episodes, n_steps, n_contexts, n_realizations)
    goal_ind, goal_value: (n_episodes,); posterior is optional
    returns: Figure

    Two panels because the two say different things. Only accuracy is bounded by
    the exact observer: belief-in-truth is linear in the belief, so a sharper
    estimator scores higher on it whether or not it is more often right.
    """
    goal_value = _numpy(goal_value)
    episodes = np.arange(goal_value.shape[0])
    series = [("chain", _goal_belief(belief, goal_ind), CHAIN)]
    if posterior is not None:
        series.append(("exact posterior", _goal_belief(posterior, goal_ind), IDEAL))

    fig = fig or plt.figure(figsize=figsize)
    truth_ax, accuracy_ax = fig.subplots(1, 2)
    for name, goal, colour in series:
        truth_ax.plot(goal[episodes, :, goal_value].mean(0), color=colour,
            linewidth=1.8, label=name)
        accuracy_ax.plot((goal.argmax(-1) == goal_value[:, None]).mean(0), color=colour,
            linewidth=1.8, label=name)

    chance = 1 / series[0][1].shape[-1]
    panels = ((truth_ax, "mean belief in the truth"), (accuracy_ax, "accuracy"))
    for ax, title in panels:
        ax.axhline(chance, color=MUTED, linestyle=":", linewidth=1.2, label="chance")
        ax.set_xlabel("step")
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"averaged over {len(episodes)} episodes", fontsize=9, color=MUTED)
    fig.tight_layout()
    return fig

# --------------------------------------------------------------------- training
_SERIES_STYLES = (("ideal", IDEAL, "--"), ("chance", MUTED, ":"),
    ("naive", MUTED, "-."), ("held", CHAIN, "--"))

def _series_style(name: str) -> tuple[str, str]:
    """Label substring --> (colour, linestyle). First match wins."""
    lowered = name.lower()
    for token, colour, dash in _SERIES_STYLES:
        if token in lowered:
            return colour, dash
    return CHAIN, "-"

def _smoothed(series: Sequence[float], smooth: int) -> np.ndarray:
    """Centred moving average of width smooth.
    series: one value per iteration
    returns: (len(series) - smooth + 1,), or the series itself when it is shorter
    """
    values = np.asarray(series, dtype=float)
    if smooth > 1 and values.size >= smooth:
        values = np.convolve(values, np.ones(smooth) / smooth, mode="valid")
    return values

def plot_training(history: Mapping[str, Sequence[float]], *, smooth: int = 1,
    ylabel: str = "goal accuracy", fig: Figure | None = None,
    figsize=(6.5, 4.0)) -> Figure:
    """Plot every series in history against training iteration.
    history: label --> one value per iteration, or fewer measured at an even cadence
    smooth: width of a centred moving average, 1 disabling it
    returns: Figure
    """
    fig = fig or plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    iterations = max(len(series) for series in history.values())
    for name, series in history.items():
        # Only a per-iteration series is noisy enough to want smoothing, and only it
        # is indexed by iteration; one measured periodically is spread evenly across
        # the run rather than crushed against the left edge.
        values = _smoothed(series, smooth if len(series) == iterations else 1)
        colour, dash = _series_style(name)
        ax.plot(np.linspace(0, iterations - 1, len(values)), values, label=name,
            linewidth=1.6, color=colour, linestyle=dash)

    ax.set_xlabel("training iteration")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig

def plot_losses(losses: Mapping[str, Sequence[float]], *, smooth: int = 1,
    fig: Figure | None = None, figsize=(9.0, 3.6)) -> Figure:
    """Plot one objective per panel against training iteration.
    losses: label --> one value per iteration
    smooth: width of a centred moving average, 1 disabling it
    returns: Figure

    A panel each rather than one pair of axes: separate objectives sit on their own
    scale and share no floor, so a single y-axis would invite reading one against
    the other.
    """
    fig = fig or plt.figure(figsize=figsize)
    axes = fig.subplots(1, len(losses), squeeze=False)[0]
    for ax, (name, series) in zip(axes, losses.items(), strict=True):
        ax.plot(_smoothed(series, smooth), linewidth=1.6, color=CHAIN)
        ax.set_xlabel("training iteration")
        ax.set_ylabel("loss")
        ax.set_title(name, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig

# --------------------------------------------------------------- generalization
def plot_generalization(scores: Mapping[str, Mapping[str, float]], *,
    chance: float | None = None, fig: Figure | None = None,
    figsize=(6.5, 4.0)) -> Figure:
    """Bar per observer, group per split of the variable pool.
    scores: split --> observer label --> accuracy in [0, 1]
    chance: drawn across the panel when given
    returns: Figure
    """
    splits, labels = list(scores), list(next(iter(scores.values())))
    group = np.arange(len(splits))
    width = 0.8 / len(labels)

    fig = fig or plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    for offset, label in enumerate(labels):
        ax.bar(group + offset * width, [scores[split][label] for split in splits],
            width, label=label, color=_series_style(label)[0])
    if chance is not None:
        ax.axhline(chance, color=MUTED, linestyle=":", linewidth=1.2, label="chance")

    ax.set_xticks(group + width * (len(labels) - 1) / 2)
    ax.set_xticklabels(splits)
    ax.set_ylabel("goal accuracy")
    ax.set_ylim(0, 1)
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
