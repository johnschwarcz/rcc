"""Stage 4 — interactions to an action.
An action is a full joint assignment, one realization per active variable, so the
policy is a distribution over n_realizations ** n_contexts outcomes, shaped as a
grid rather than a flat vector.
"""
import torch
from torch import Tensor, nn
from ._helpers import sample_categorical, seeded
from .config import RCCConfig

__all__ = ["Controller", "intrinsic_value"]

class Controller(nn.Module):
    """An actor and a critic utilizing learned interactions.
    cfg.n_joint_realizations sets the width of the actor's readout.
    """

    def __init__(self, cfg: RCCConfig) -> None:
        super().__init__()
        self.cfg = cfg
        width = cfg.n_observations * cfg.n_interactions
        with seeded(cfg.seed):
            self.actor = nn.Sequential(
                nn.Linear(width, cfg.hidden_dim),
                nn.ReLU(),
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
                nn.ReLU(),
                nn.Linear(cfg.hidden_dim, cfg.n_joint_realizations),
            )
            self.critic = nn.Sequential(
                nn.Linear(width, cfg.hidden_dim),
                nn.ReLU(),
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
                nn.ReLU(),
                nn.Linear(cfg.hidden_dim, 1),
            )

    def forward(self, interactions: Tensor) -> tuple[Tensor, Tensor]:
        """Score every joint realization from the interactions alone.
        interactions: (n_episodes, n_observations, n_interactions), detached here.
        returns: policy (n_episodes, *realization_shape), value (n_episodes,) 
        """
        expected = (self.cfg.n_observations, self.cfg.n_interactions)
        if interactions.ndim != 3 or tuple(interactions.shape[1:]) != expected:
            raise ValueError(
                f"interactions must have shape (n_episodes, {expected[0]}, "
                f"{expected[1]}), got {tuple(interactions.shape)}")
        flat = interactions.detach().reshape(interactions.shape[0], -1)
        policy = torch.softmax(self.actor(flat), -1)
        value = torch.sigmoid(self.critic(flat)).squeeze(-1)
        return policy.reshape(-1, *self.cfg.realization_shape), value

    def act(
        self, policy: Tensor, generator: torch.Generator | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Sample a latent state of the world.
        policy: (n_episodes, *realization_shape)
        returns: actions (n_episodes, n_contexts), log_prob and entropy (n_episodes,)
        """
        flat = policy.reshape(policy.shape[0], -1)
        chosen = sample_categorical(flat, generator)
        log_policy = flat.clamp_min(1e-12).log()
        log_prob = log_policy.gather(-1, chosen[:, None]).squeeze(-1)
        entropy = -(flat * log_policy).sum(-1)
        return self._unravel(chosen), log_prob, entropy

    def best(self, policy: Tensor) -> Tensor:
        """The joint realization the policy likes most.
        policy: (n_episodes, *realization_shape)
        returns: (n_episodes, n_contexts) of realization indices
        """
        return self._unravel(policy.reshape(policy.shape[0], -1).argmax(-1))

    def _unravel(self, chosen: Tensor) -> Tensor:
        """A flat action index back to one realization per active variable."""
        axes = torch.unravel_index(chosen, self.cfg.realization_shape)
        return torch.stack(axes, 1).long()


def intrinsic_value(rates: Tensor, preferences: Tensor, eps: float = 1e-6) -> Tensor:
    """How well a set of observation rates matches what the chain wants to see.
    The geometric mean over channels of rate where the channel is preferred and
    1 - rate where it is not, so one badly-missed channel cannot be bought back.
    rates: (n_episodes, n_observations) in [0, 1]
    preferences: (n_observations,) or (n_episodes, n_observations)
    returns: (n_episodes,) in [0, 1]
    """
    if preferences.ndim == 1:
        preferences = preferences[None, :]
    if preferences.shape[-1] != rates.shape[-1]:
        raise ValueError(
            f"preferences must cover {rates.shape[-1]} channels, got "
            f"{preferences.shape[-1]}"
        )
    rates = rates.clamp(eps, 1 - eps)
    wanted = rates.log() * preferences
    unwanted = (1 - rates).log() * (1 - preferences)
    return ((wanted + unwanted).sum(-1) / rates.shape[-1]).exp()