"""Plotting. Every function returns a figure; none call ``show()``.

Needs the ``viz`` extra::

    pip install "rcc[viz]"

Colours are shared with `coggrid <https://github.com/johnschwarcz/coggrid>`_ so
that a chain's belief and an ideal observer's posterior look the same in both
repositories: the exact observer in blue, the truth in green, and anything the
chain learned in purple.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

__all__ = [
    "CHAIN",
    "IDEAL",
    "TRUTH",
    "plot_architecture",
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
# the architecture
# --------------------------------------------------------------------------- #
def plot_architecture(*, fig: Figure | None = None, figsize=(10.5, 4.2)) -> Figure:
    """Draw the four stages and the seam between representation and inference.

    The one thing this figure exists to make obvious: the arrow into the
    classifier is dashed, because gradient does not travel back along it.

    Returns
    -------
    Figure
    """
    fig = fig or plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 10.6)
    ax.set_ylim(0, 4.3)
    ax.axis("off")

    # The interactions fan out to three consumers, which cannot be drawn as
    # three straight arrows without crossings. So they leave as one bus that
    # runs under the forward path, and each consumer taps it from below.
    boxes = [
        (0.10, 2.30, 1.35, 0.80, MUTED, "variables\n$v_1 \\ldots v_C$"),
        (1.95, 2.30, 2.05, 0.80, CHAIN, "1. interactions\n$Z_{ij} = ⟨K_i, Q_j⟩$"),
        (4.95, 2.30, 2.30, 0.80, IDEAL, "2. classifier\nsoftmax(cumsum)"),
        (8.05, 2.30, 1.45, 0.80, IDEAL, "belief\n$b(r)$"),
        (5.60, 0.40, 2.30, 0.80, CHAIN, "3. generator\n$\\hat{p}$(obs)"),
        (1.95, 0.40, 2.05, 0.80, TRUTH, "4. controller\naction"),
        (4.95, 3.55, 2.30, 0.62, MUTED, "observations $o_{1..T}$"),
    ]
    for x, y, w, h, colour, label in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h,
                boxstyle="round,pad=0.06",
                linewidth=1.6, edgecolor=colour, facecolor=colour + "1f",
            )
        )
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9)

    def arrow(start, end, *, dashed=False, colour="0.35", rad=0.0, width=1.4):
        ax.add_patch(
            FancyArrowPatch(
                start, end,
                arrowstyle="-|>", mutation_scale=13, linewidth=width, color=colour,
                linestyle=(0, (4, 3)) if dashed else "solid",
                connectionstyle=f"arc3,rad={rad}",
                shrinkA=0, shrinkB=0,
            )
        )

    arrow((1.45, 2.70), (1.95, 2.70))
    arrow((4.00, 2.70), (4.95, 2.70), dashed=True, colour=CHAIN, width=1.8)
    arrow((7.25, 2.70), (8.05, 2.70))
    arrow((6.10, 3.55), (6.10, 3.10))
    arrow((8.45, 2.30), (7.70, 1.20))

    # The interaction bus, and its two taps.
    ax.plot([2.98, 6.75], [1.75, 1.75], color=CHAIN, linewidth=1.4, zorder=1)
    ax.plot([2.98, 2.98], [2.30, 1.75], color=CHAIN, linewidth=1.4, zorder=1)
    arrow((2.98, 1.75), (2.98, 1.20), colour=CHAIN)
    arrow((6.75, 1.75), (6.75, 1.20), colour=CHAIN)

    # The one arrow that runs backwards: prediction error, into the embeddings.
    arrow((5.60, 0.80), (3.45, 2.30), colour=CHAIN, rad=0.32, dashed=True)
    ax.text(
        4.62, 1.05, "prediction error", fontsize=8, color=CHAIN,
        ha="center", va="center", style="italic",
    )
    ax.text(
        4.47, 2.92, "detached", fontsize=8.5, color=CHAIN,
        ha="center", style="italic",
    )
    ax.set_title(
        "A Representation Classification Chain", fontsize=12, pad=6, loc="left"
    )
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# stage 2
# --------------------------------------------------------------------------- #
def plot_belief_accumulation(
    belief,
    goal_index,
    goal_value,
    posterior=None,
    episode: int = 0,
    *,
    fig: Figure | None = None,
    figsize=(9.5, 3.6),
) -> Figure:
    """Show one episode's belief filling in, step by step.

    Parameters
    ----------
    belief:
        ``(n_episodes, n_steps, n_contexts, n_realizations)`` from the chain.
    goal_index:
        ``(n_episodes,)`` — which active variable is the goal.
    goal_value:
        ``(n_episodes,)`` — the truth.
    posterior:
        Optional exact posterior in the same shape, drawn alongside for
        comparison.
    episode:
        Which episode of the batch to draw.

    Returns
    -------
    Figure
    """
    belief = _numpy(belief)
    goal_index = _numpy(goal_index)
    goal_value = _numpy(goal_value)
    goal = int(goal_index[episode])
    truth = int(goal_value[episode])

    chain_belief = belief[episode, :, goal]
    panels = [("chain", chain_belief, CHAIN)]
    if posterior is not None:
        panels.append(("exact posterior", _numpy(posterior)[episode, :, goal], IDEAL))

    fig = fig or plt.figure(figsize=figsize)
    axes = fig.subplots(1, len(panels) + 1, width_ratios=[1] * len(panels) + [0.85])

    for ax, (name, values, colour) in zip(axes, panels, strict=False):
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
def plot_training(
    history: Mapping[str, Sequence[float]],
    *,
    smooth: int = 1,
    ylabel: str = "goal accuracy",
    fig: Figure | None = None,
    figsize=(6.5, 4.0),
) -> Figure:
    """Plot every series in ``history`` against training step.

    Parameters
    ----------
    history:
        Maps a label to a sequence of per-step values. Labels containing
        ``"ideal"`` are drawn in the exact-observer colour and dashed, anything
        containing ``"chance"`` in grey and dotted, everything else in the
        chain's colour.
    smooth:
        Width of a centred moving average. ``1`` disables smoothing.
    ylabel:
        Label for the vertical axis.

    Returns
    -------
    Figure
    """
    fig = fig or plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    for name, series in history.items():
        values = np.asarray(series, dtype=float)
        if smooth > 1 and values.size >= smooth:
            values = np.convolve(values, np.ones(smooth) / smooth, mode="valid")
        lowered = name.lower()
        style: dict[str, Any]
        if "ideal" in lowered:
            style = dict(color=IDEAL, linestyle="--")
        elif "chance" in lowered:
            style = dict(color=MUTED, linestyle=":")
        else:
            style = dict(color=CHAIN)
        ax.plot(values, label=name, linewidth=1.6, **style)

    ax.set_xlabel("training step")
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

    Only defined for two active variables, where the joint realization grid is
    a plane that can be drawn.

    Parameters
    ----------
    policy:
        ``(n_episodes, n_realizations, n_realizations)``.
    landscape:
        Optional ``(n_episodes, n_realizations, n_realizations)`` of the true
        value of each joint realization.
    episode:
        Which episode of the batch to draw.

    Returns
    -------
    Figure
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
