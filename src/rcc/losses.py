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

def supervised_loss(belief: Tensor, target: Tensor, *,
    eps: float = 1e-8, root: bool = True) -> Tensor:
    require_shape("target", target, belief.shape)
    divergence = symmetric_kl(belief, target, bernoulli=False)
    shifted = divergence - divergence.detach().min() + eps
    shifted = shifted.sqrt() if root else shifted
    return shifted.mean()

def reward_loss(goal_belief: Tensor, selection: Tensor, correct: Tensor, *,
    entropy_bonus: float = 0.1) -> Tensor:
    """
    goal_belief: (n_episodes, n_steps, n_realizations)
    selection, correct: (n_episodes,)
    the final verdict evaluated at the final step is applied at every step
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
    """Score the generator against the experienced observations.
    predicted_rates, observed_rates: (n_episodes, n_observations);
    observed is normally observations.mean(1)
    correct: (n_episodes,), biases loss to episodes with correct actions
    chance: weight on the unconditional term, so 0 is correct episodes only
    returns: scalar
    """
    require_shape("observed_rates", observed_rates, predicted_rates.shape)
    error = symmetric_kl(predicted_rates, observed_rates)
    mean_error = error.mean()
    if correct is None:
        return mean_error

    require_shape("correct", correct, (predicted_rates.shape[0],))
    total = correct.sum()
    if float(total) == 0.0:
        return mean_error
    weighted_error = (correct * error.mean(-1)).sum() / total
    return mean_error * chance + (1 - chance) * weighted_error

def embedding_norm_penalty(keys: Tensor, queries: Tensor) -> Tensor:
    """Hold the embeddings on the unit sphere."""
    off_sphere = (keys.norm(dim=-1) - 1) ** 2 + (queries.norm(dim=-1) - 1) ** 2
    return off_sphere.mean()

def controller_loss(value: Tensor, predicted_value: Tensor, log_prob: Tensor,
    entropy: Tensor, *, entropy_bonus: float = 0.05) -> Tensor:
    """Advantage actor-critic for stage 4."""
    advantage = value - predicted_value
    policy_gradient = (-log_prob * advantage.detach()).mean()
    critic = (advantage**2).mean()
    return policy_gradient + critic - entropy_bonus * entropy.mean()