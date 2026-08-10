"""Plotting. Every function returns a figure; none call ``show()``.

Needs the ``viz`` extra::

    pip install "rcc[viz]"

Colours are shared with `coggrid <https://github.com/johnschwarcz/coggrid>`_ so
that a chain's belief and an ideal observer's posterior look the same in both
repositories: the exact observer in blue, the truth in green, and anything the
chain learned in purple.
"""

from collections.abc import Mapping, Sequence
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

__all__ = [
    "CHAIN",
    "IDEAL",
    "MUTED",
    "TRUTH",
    "plot_belief_accumulation",
    "plot_policy",
    "plot_training",
]

#: What the chain learned.
CHAIN = "#7b52ab"
#: What an exact observer would have concluded.
IDEAL = "#1f77b4"
#: The truth.
TRUTH = "#78c765"
#: Everything else.
MUTED = "#A9A9A9"


def _numpy(x) -> np.ndarray:
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


# --------------------------------------------------------------------------- #
# stage 2
# --------------------------------------------------------------------------- #
def plot_belief_accumulation(
    belief,
    goal_ind,
    goal_value,
    posterior=None,
    episode: int = 0,
    *,
    fig: Figure | None = None,
    figsize=(9.5, 3.6),
) -> Figure:
    """Show one episode's belief filling in, step by step.

    ``belief`` is ``(n_episodes, n_steps, n_contexts, n_realizations)`` from the
    chain, ``goal_ind`` says which active variable is the goal and
    ``goal_value`` is the truth, both ``(n_episodes,)``. An optional ``posterior``
    in the shape of ``belief`` is drawn alongside for comparison, and ``episode``
    picks which episode of the batch to draw.
    """
    belief = _numpy(belief)
    goal_ind = _numpy(goal_ind)
    goal_value = _numpy(goal_value)
    goal = int(goal_ind[episode])
    truth = int(goal_value[episode])

    chain_belief = belief[episode, :, goal]
    panels = [("chain", chain_belief, CHAIN)]
    if posterior is not None:
        panels.append(("exact posterior", _numpy(posterior)[episode, :, goal], IDEAL))

    fig = fig or plt.figure(figsize=figsize)
    axes = fig.subplots(1, len(panels) + 1, width_ratios=[1] * len(panels) + [0.85])

    # The last axis is the line plot below, so only the image panels pair up here.
    for ax, (name, values, colour) in zip(axes[:-1], panels, strict=True):
        ax.imshow(
            values.T, aspect="auto", origin="lower", cmap="magma",
            vmin=0, vmax=1, interpolation="nearest",
        )
        ax.axhline(truth, color=TRUTH, linewidth=1.8, linestyle="--")
        ax.set_xlabel("step")
        ax.set_ylabel("realization")
        # Realizations are categories, so only whole numbers name one.
        ax.set_yticks(range(belief.shape[-1]))
        ax.set_title(name, fontsize=10, color=colour)

    ax = axes[-1]
    for name, values, colour in panels:
        ax.plot(values[:, truth], color=colour, linewidth=1.8, label=name)
    ax.axhline(
        1 / belief.shape[-1], color=MUTED, linestyle=":", linewidth=1.2, label="chance"
    )
    ax.set_xlabel("step")
    ax.set_ylabel("belief in the truth")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("evidence accumulating", fontsize=10)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
#: Label substring to (colour, linestyle). First match wins.
_SERIES_STYLES = (("ideal", IDEAL, "--"), ("chance", MUTED, ":"))


def _series_style(name: str) -> tuple[str, str]:
    lowered = name.lower()
    for token, colour, dash in _SERIES_STYLES:
        if token in lowered:
            return colour, dash
    return CHAIN, "-"


def plot_training(
    history: Mapping[str, Sequence[float]],
    *,
    smooth: int = 1,
    ylabel: str = "goal accuracy",
    fig: Figure | None = None,
    figsize=(6.5, 4.0),
) -> Figure:
    """Plot every series in ``history`` against training iteration.

    ``history`` maps a label to one value per iteration. Labels containing
    ``"ideal"`` are drawn in the exact-observer colour and dashed, anything
    containing ``"chance"`` in grey and dotted, everything else in the chain's
    colour. ``smooth`` is the width of a centred moving average, ``1`` disabling it.
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


# --------------------------------------------------------------------------- #
# stage 4
# --------------------------------------------------------------------------- #
def plot_policy(
    policy,
    landscape=None,
    episode: int = 0,
    *,
    fig: Figure | None = None,
    figsize=(7.0, 3.4),
) -> Figure:
    """Compare the controller's policy with the value it was chasing.

    Only defined for two active variables, where the joint realization grid is a
    plane that can be drawn. ``policy`` and the optional ``landscape`` of true
    per-realization value are both ``(n_episodes, n_realizations, n_realizations)``;
    ``episode`` picks which episode of the batch to draw.
    """
    policy = _numpy(policy)
    if policy.ndim != 3:
        raise ValueError(
            "plot_policy draws a two-variable grid, so policy must have shape "
            f"(n_episodes, n_realizations, n_realizations); got {policy.shape}"
        )

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
