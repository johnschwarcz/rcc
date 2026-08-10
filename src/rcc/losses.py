"""Objectives, one per stage, plus the divergences they are built from.
* :func:`supervised_loss` — match a belief you already trust
* :func:`reward_loss` — learn from whether an action was right
* :func:`prediction_loss` — learn embeddings to predict observations
* :func:`embedding_norm_penalty` — keep embeddings on unit sphere
* :func:`controller_loss` — actor-critic, for stage 4
"""
import torch
from torch import Tensor
from ._helpers import require_shape

__all__ = ["controller_loss", "embedding_norm_penalty", "kl_divergence",
    "prediction_loss", "reward_loss", "soft_clip", "supervised_loss", "symmetric_kl"]

def soft_clip(x: Tensor, eps: float = 1e-3) -> Tensor:
    """Softly squash x into [eps, 1 - eps] without zeroing the gradient."""
    upper = 1.0 - eps
    x = upper - torch.sigmoid((upper - x) / eps) * (upper - x)
    return eps + torch.sigmoid((x - eps) / eps) * (x - eps)

def kl_divergence(x: Tensor, y: Tensor, eps: float = 1e-3) -> Tensor:
    x = soft_clip(x, eps)
    y = soft_clip(y, eps)
    return x * torch.log(x / y)

def symmetric_kl(x: Tensor, y: Tensor, *, bernoulli: bool = True, eps: float = 1e-3) -> Tensor:
    forward = kl_divergence(x, y, eps)
    backward = kl_divergence(y, x, eps)
    if bernoulli:
        forward = forward + kl_divergence(1 - x, 1 - y, eps)
        backward = backward + kl_divergence(1 - y, 1 - x, eps)
    return (forward + backward) / 2

def supervised_loss(belief: Tensor, target: Tensor, *,  eps: float = 1e-8, root: bool = True) -> Tensor:
    require_shape("target", target, belief.shape)
    divergence = symmetric_kl(belief, target, bernoulli=False)
    shifted = divergence - divergence.detach().min() + eps
    shifted = shifted.sqrt() if root else shifted
    return shifted.mean()


def reward_loss(goal_belief: Tensor, selection: Tensor, correct: Tensor, *,
    entropy_bonus: float = 0.1) -> Tensor:
    """Learn from a verdict on the committed answer, with no target belief.
    goal_belief: (n_episodes, n_steps, n_realizations)
    selection, correct: (n_episodes,); the final verdict is applied at every step
    returns: scalar
    """
    if goal_belief.ndim != 3:
        raise ValueError("goal_belief must have shape (n_episodes, n_steps, "
            f"n_realizations), got {tuple(goal_belief.shape)}")
    n_episodes, n_steps, _ = goal_belief.shape
    require_shape("selection", selection, (n_episodes,))
    require_shape("correct", correct, (n_episodes,))

    chosen = selection[:, None, None].expand(-1, n_steps, 1)
    belief = soft_clip(goal_belief.gather(-1, chosen).squeeze(-1))
    verdict = correct[:, None]
    reward = verdict * -belief.log()
    punishment = (1 - verdict) * -(1 - belief).log()
    entropy = -belief * belief.log() * entropy_bonus
    return (reward + punishment - entropy).mean()


def prediction_loss(predicted_rates: Tensor, observed_rates: Tensor, *,
    correct: Tensor | None = None, chance: float = 0.0) -> Tensor:
    """Score the generator against what the world actually did.
    predicted_rates, observed_rates: (n_episodes, n_observations); observed is
    normally observations.mean(1)
    correct: (n_episodes,), blends the unconditional average with the average over
    correct episodes, weighted by chance, normally 1 / n_realizations
    returns: scalar

    >>> observed = torch.tensor([[0.8, 0.2]])
    >>> float(prediction_loss(observed, observed).round(decimals=4))
    0.0
    >>> bool(prediction_loss(torch.tensor([[0.2, 0.8]]), observed) > 0.5)
    True
    """
    require_shape("observed_rates", observed_rates, predicted_rates.shape)
    error = symmetric_kl(predicted_rates, observed_rates)
    if correct is None:
        return error.mean()

    require_shape("correct", correct, (predicted_rates.shape[0],))
    weight = correct[:, None]
    total = weight.sum()
    # Every episode wrong is a real state early in training, not an error. Fall
    # back to the unconditional average rather than dividing by zero.
    if float(total) == 0.0:
        return error.mean()
    when_correct = (weight * error).sum() / total
    return error.mean() * chance + (1 - chance) * when_correct


def embedding_norm_penalty(keys: Tensor, queries: Tensor) -> Tensor:
    """Hold the embeddings on the unit sphere.
    keys, queries: (..., embedding_dim), from rcc.interactions.Interaction
    returns: scalar, zero when every embedding has unit norm

    >>> unit = torch.eye(3)
    >>> float(embedding_norm_penalty(unit, unit))
    0.0
    >>> bool(embedding_norm_penalty(unit * 3, unit) > 0)
    True
    """
    off_sphere = (keys.norm(dim=-1) - 1) ** 2 + (queries.norm(dim=-1) - 1) ** 2
    return off_sphere.mean()


def controller_loss(value: Tensor, predicted_value: Tensor, log_prob: Tensor,
    entropy: Tensor, *, entropy_bonus: float = 0.05) -> Tensor:
    """Advantage actor-critic for stage 4.
    value, predicted_value, log_prob, entropy: (n_episodes,); value is e.g. from
    rcc.controller.intrinsic_value
    returns: scalar — policy gradient, critic regression, entropy bonus

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
