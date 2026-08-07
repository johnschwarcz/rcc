"""Objectives, one per stage, plus the divergences they are built from.

Which objectives you use is how you choose what the chain is being asked to do.
Nothing here is wired into a module, and nothing here assumes a task:

============================  ==============================================
:func:`distillation_loss`     match a belief you already trust
:func:`reward_loss`           learn from whether the committed answer was right
:func:`prediction_loss`       train the embeddings on the world, unsupervised
:func:`embedding_norm_penalty`  keep those embeddings on the unit sphere
:func:`controller_loss`       actor-critic, for stage 4
============================  ==============================================

Beliefs and rates are both probabilities, so every divergence below clamps
smoothly rather than hard: a hard clamp has zero gradient once it bites, which
silently kills learning exactly where the chain is most confident and most
wrong.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = [
    "controller_loss",
    "distillation_loss",
    "embedding_norm_penalty",
    "kl_divergence",
    "prediction_loss",
    "reward_loss",
    "soft_clip",
    "symmetric_kl",
]


def soft_clip(x: Tensor, eps: float = 1e-3) -> Tensor:
    """Squash ``x`` into ``[eps, 1 - eps]`` without flattening the gradient.

    Examples
    --------
    Values already inside the range pass through essentially untouched:

    >>> soft_clip(torch.tensor([0.5])).round(decimals=4)
    tensor([0.5000])

    Values outside it are pulled to the boundary, and still carry gradient:

    >>> out = soft_clip(torch.tensor([-5.0, 5.0], requires_grad=True))
    >>> [round(v, 3) for v in out.tolist()]
    [0.001, 0.999]
    >>> out.sum().backward()  # no error: the gradient exists
    """
    upper = 1.0 - eps
    x = upper - torch.sigmoid((upper - x) / eps) * (upper - x)
    return eps + torch.sigmoid((x - eps) / eps) * (x - eps)


def kl_divergence(x: Tensor, y: Tensor, eps: float = 1e-3) -> Tensor:
    """Elementwise ``x * log(x / y)``, both arguments soft-clipped.

    Returned per element rather than summed, because the objectives below
    reshape those elements before reducing them.
    """
    x = soft_clip(x, eps)
    y = soft_clip(y, eps)
    return x * torch.log(x / y)


def symmetric_kl(
    x: Tensor, y: Tensor, *, bernoulli: bool = True, eps: float = 1e-3
) -> Tensor:
    """Divergence between ``x`` and ``y`` that does not care which came first.

    Parameters
    ----------
    x, y:
        Probabilities, broadcastable against each other.
    bernoulli:
        Whether each element is its own Bernoulli distribution, in which case
        the complement ``1 - x`` carries information too and is included. Set
        ``False`` when the elements are categories of one shared distribution,
        so that only ``x`` itself counts.
    eps:
        Clamp passed to :func:`soft_clip`.

    Returns
    -------
    Tensor
        Elementwise, same broadcast shape as the inputs.

    Examples
    --------
    Symmetric in its arguments, unlike a plain KL:

    >>> a, b = torch.tensor([0.2, 0.7]), torch.tensor([0.5, 0.5])
    >>> bool(torch.allclose(symmetric_kl(a, b), symmetric_kl(b, a)))
    True

    Zero exactly when the two agree:

    >>> bool(symmetric_kl(a, a).abs().max() < 1e-6)
    True
    """
    forward = kl_divergence(x, y, eps)
    backward = kl_divergence(y, x, eps)
    if bernoulli:
        forward = forward + kl_divergence(1 - x, 1 - y, eps)
        backward = backward + kl_divergence(1 - y, 1 - x, eps)
    return (forward + backward) / 2


def distillation_loss(belief: Tensor, target: Tensor, eps: float = 1e-8) -> Tensor:
    """Pull the chain's belief towards one you already trust.

    The target can come from anywhere — an exact posterior, a slower model, a
    human annotation — which is what makes this the objective to reach for when
    a reference answer exists.

    The elementwise divergences are square-rooted before they are averaged, so a
    single badly-wrong category cannot dominate the batch. They are shifted by
    the batch minimum first, because an elementwise KL contribution can be
    negative and a square root cannot take that.

    Parameters
    ----------
    belief, target:
        Same shape, normalized over the last axis. Pass a single time slice
        (``belief[:, -1]``) to score only the chain's final answer.
    eps:
        Floor on the shifted divergence, keeping the square root differentiable
        at its minimum.

    Returns
    -------
    Tensor
        Scalar.

    Examples
    --------
    Minimized when the two agree, where it bottoms out at ``sqrt(eps)`` rather
    than at zero:

    >>> target = torch.tensor([[0.7, 0.2, 0.1]])
    >>> bool(distillation_loss(target, target) < 1e-3)
    True
    >>> bool(distillation_loss(torch.tensor([[0.1, 0.2, 0.7]]), target)
    ...      > distillation_loss(target, target))
    True
    """
    if belief.shape != target.shape:
        raise ValueError(
            f"belief {tuple(belief.shape)} and target {tuple(target.shape)} must "
            "have the same shape"
        )
    divergence = symmetric_kl(belief, target, bernoulli=False)
    shifted = divergence - divergence.detach().min() + eps
    return shifted.sqrt().mean()


def reward_loss(
    goal_belief: Tensor,
    selection: Tensor,
    correct: Tensor,
    *,
    entropy_bonus: float = 0.1,
) -> Tensor:
    """Learn from a verdict on the committed answer, with no target belief.

    The chain commits to one realization at the end of the episode and is told
    only whether that was right. That single bit is then applied to the belief
    in that realization at *every* step, so an answer that turned out correct
    pulls up the whole trajectory that led to it.

    Parameters
    ----------
    goal_belief:
        ``(n_episodes, n_steps, n_realizations)`` — the belief about the goal
        variable over time.
    selection:
        ``(n_episodes,)`` — the realization the chain committed to.
    correct:
        ``(n_episodes,)`` in [0, 1] — whether that commitment was right.
    entropy_bonus:
        Weight on an entropy term that resists collapsing to certainty.

    Returns
    -------
    Tensor
        Scalar.

    Examples
    --------
    Confidence is cheap when it was right and expensive when it was wrong:

    >>> belief = torch.tensor([[[0.9, 0.05, 0.05]]])
    >>> right = reward_loss(belief, torch.tensor([0]), torch.tensor([1.0]))
    >>> wrong = reward_loss(belief, torch.tensor([0]), torch.tensor([0.0]))
    >>> bool(wrong > right)
    True
    """
    if goal_belief.ndim != 3:
        raise ValueError(
            "goal_belief must have shape (n_episodes, n_steps, n_realizations), "
            f"got {tuple(goal_belief.shape)}"
        )
    n_episodes, n_steps, _ = goal_belief.shape
    for name, tensor in (("selection", selection), ("correct", correct)):
        if tensor.shape != (n_episodes,):
            raise ValueError(
                f"{name} must have shape ({n_episodes},), got {tuple(tensor.shape)}"
            )

    chosen = selection[:, None, None].expand(-1, n_steps, 1)
    belief = soft_clip(goal_belief.gather(-1, chosen).squeeze(-1))

    verdict = correct[:, None]
    reward = verdict * -belief.log()
    punishment = (1 - verdict) * -(1 - belief).log()
    entropy = -belief * belief.log() * entropy_bonus
    return (reward + punishment - entropy).mean()


def prediction_loss(
    predicted_rates: Tensor,
    observed_rates: Tensor,
    *,
    correct: Tensor | None = None,
    chance: float = 0.0,
) -> Tensor:
    """Score the generator against what the world actually did.

    This is the objective that trains the embeddings, and it needs no labels:
    the observations are their own target.

    When ``correct`` is supplied the score is a fixed blend of two averages —
    over all episodes, and over the episodes the chain got right. The blend
    weight is the chance rate, which is the honest one: while the chain is
    guessing, "the episodes it got right" is not a meaningful subset and the
    unconditional average carries the signal; as it improves, the correct
    episodes are the ones whose proposed realization was worth predicting from.

    Parameters
    ----------
    predicted_rates:
        ``(n_episodes, n_observations)`` from
        :class:`~rcc.generator.ObservationGenerator`.
    observed_rates:
        ``(n_episodes, n_observations)`` — the empirical rate per channel,
        normally ``observations.mean(1)``.
    correct:
        ``(n_episodes,)`` in [0, 1], or ``None`` to weight every episode alike.
    chance:
        Weight on the unconditional average, normally ``1 / n_realizations``.
        Ignored when ``correct`` is ``None``.

    Returns
    -------
    Tensor
        Scalar.

    Examples
    --------
    >>> observed = torch.tensor([[0.8, 0.2]])
    >>> float(prediction_loss(observed, observed).round(decimals=4))
    0.0
    >>> bool(prediction_loss(torch.tensor([[0.2, 0.8]]), observed) > 0.5)
    True
    """
    if predicted_rates.shape != observed_rates.shape:
        raise ValueError(
            f"predicted_rates {tuple(predicted_rates.shape)} and observed_rates "
            f"{tuple(observed_rates.shape)} must have the same shape"
        )
    error = symmetric_kl(predicted_rates, observed_rates)
    if correct is None:
        return error.mean()

    if correct.shape != (predicted_rates.shape[0],):
        raise ValueError(
            f"correct must have shape ({predicted_rates.shape[0]},), got "
            f"{tuple(correct.shape)}"
        )
    weight = correct[:, None]
    total = weight.sum()
    # Every episode wrong is a real state early in training, not an error.
    # Fall back to the unconditional average rather than dividing by zero.
    if float(total) == 0.0:
        return error.mean()
    when_correct = (weight * error).sum() / total
    return error.mean() * chance + (1 - chance) * when_correct


def embedding_norm_penalty(keys: Tensor, queries: Tensor) -> Tensor:
    """Hold the embeddings on the unit sphere.

    Without this the prediction loss can be reduced by inflating the embeddings
    instead of orienting them, and the interactions lose their scale.

    Parameters
    ----------
    keys, queries:
        ``(..., embedding_dim)``, from
        :class:`~rcc.interactions.Interaction`.

    Returns
    -------
    Tensor
        Scalar, zero when every embedding has unit norm.

    Examples
    --------
    >>> unit = torch.eye(3)
    >>> float(embedding_norm_penalty(unit, unit))
    0.0
    >>> bool(embedding_norm_penalty(unit * 3, unit) > 0)
    True
    """
    off_sphere = (keys.norm(dim=-1) - 1) ** 2 + (queries.norm(dim=-1) - 1) ** 2
    return off_sphere.mean()


def controller_loss(
    value: Tensor,
    predicted_value: Tensor,
    log_prob: Tensor,
    entropy: Tensor,
    *,
    entropy_bonus: float = 0.05,
) -> Tensor:
    """Advantage actor-critic for stage 4.

    Parameters
    ----------
    value:
        ``(n_episodes,)`` — what the chosen action actually returned, e.g. from
        :func:`~rcc.controller.intrinsic_value`.
    predicted_value:
        ``(n_episodes,)`` — the critic's estimate, from
        :class:`~rcc.controller.Control`.
    log_prob, entropy:
        ``(n_episodes,)`` from :meth:`~rcc.controller.Controller.act`.
    entropy_bonus:
        Weight on the entropy term that keeps the policy exploring.

    Returns
    -------
    Tensor
        Scalar combining the policy gradient, the critic's regression, and the
        entropy bonus.

    Examples
    --------
    The critic term alone punishes a bad value estimate:

    >>> value = torch.tensor([1.0])
    >>> log_prob, entropy = torch.tensor([-0.5]), torch.tensor([0.0])
    >>> close = controller_loss(value, torch.tensor([0.9]), log_prob, entropy)
    >>> far = controller_loss(value, torch.tensor([0.1]), log_prob, entropy)
    >>> bool(far > close)
    True
    """
    advantage = value - predicted_value
    policy_gradient = (-log_prob * advantage.detach()).mean()
    critic = (advantage**2).mean()
    return policy_gradient + critic - entropy_bonus * entropy.mean()
