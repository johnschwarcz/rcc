"""Stage 4 — interactions to an action.

The controller does not see observations. It sees only the interactions, and
from them alone it picks a joint realization to put the world into. That is the
strongest test of stage 1: a representation good enough to act on has to encode
what the variables will *do* together, not merely what they were.

An action here is a full joint assignment — one realization for every active
variable — so the policy is a distribution over ``n_realizations ** n_contexts``
outcomes, shaped as a grid rather than a flat vector.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor, nn

from ._random import sample_categorical, seeded
from .config import RCCConfig

__all__ = ["Control", "Controller", "intrinsic_value"]


class Control(NamedTuple):
    """What stage 4 produces.

    Attributes
    ----------
    policy:
        ``(n_episodes, *realization_shape)`` — a distribution over joint
        realizations, laid out as a grid with one axis per active variable.
    value:
        ``(n_episodes,)`` in [0, 1] — the critic's estimate of the value the
        policy is about to collect.
    """

    policy: Tensor
    value: Tensor


class Controller(nn.Module):
    """An actor and a critic reading the interactions.

    Parameters
    ----------
    cfg:
        The chain's configuration. ``cfg.n_joint_realizations`` sets the width
        of the actor's readout, so ``n_contexts`` is the parameter that makes
        this module expensive.

    Examples
    --------
    >>> from rcc import RCCConfig
    >>> cfg = RCCConfig(n_contexts=2, n_realizations=4, n_observations=3,
    ...                 hidden_dim=16, seed=0)
    >>> controller = Controller(cfg)
    >>> interactions = torch.randn(6, 3, cfg.n_interactions)
    >>> policy, value = controller(interactions)
    >>> policy.shape, value.shape
    (torch.Size([6, 4, 4]), torch.Size([6]))

    The policy is a distribution over the grid, not per axis:

    >>> bool(torch.allclose(policy.flatten(1).sum(-1), torch.ones(6), atol=1e-5))
    True
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

    def forward(self, interactions: Tensor) -> Control:
        """Score every joint realization from the interactions alone.

        Parameters
        ----------
        interactions:
            ``(n_episodes, n_observations, n_interactions)``. Detached on the
            way in: the controller's return does not reshape the
            representation, the same way the classifier's accuracy does not.
        """
        expected = (self.cfg.n_observations, self.cfg.n_interactions)
        if interactions.ndim != 3 or tuple(interactions.shape[1:]) != expected:
            raise ValueError(
                f"interactions must have shape (n_episodes, {expected[0]}, "
                f"{expected[1]}), got {tuple(interactions.shape)}"
            )
        flat = interactions.detach().reshape(interactions.shape[0], -1)
        policy = torch.softmax(self.actor(flat), -1)
        value = torch.sigmoid(self.critic(flat)).squeeze(-1)
        return Control(policy.reshape(-1, *self.cfg.realization_shape), value)

    def act(
        self, policy: Tensor, generator: torch.Generator | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Sample a joint realization and report what it cost to choose it.

        Parameters
        ----------
        policy:
            ``(n_episodes, *realization_shape)`` from :meth:`forward`.
        generator:
            Draws from the global RNG when ``None``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            ``actions`` ``(n_episodes, n_contexts)`` — one realization per
            active variable; ``log_prob`` and ``entropy``, both
            ``(n_episodes,)``, which the policy-gradient loss needs.
        """
        flat = policy.reshape(policy.shape[0], -1)
        chosen = sample_categorical(flat, generator)
        log_prob = torch.log(flat.clamp_min(1e-12)).gather(-1, chosen[:, None])
        entropy = -(flat * torch.log(flat.clamp_min(1e-12))).sum(-1)
        actions = torch.stack(torch.unravel_index(chosen, self.cfg.realization_shape), 1)
        return actions.long(), log_prob.squeeze(-1), entropy

    def best(self, policy: Tensor) -> Tensor:
        """The joint realization the policy likes most.

        Parameters
        ----------
        policy:
            ``(n_episodes, *realization_shape)`` from :meth:`forward`.

        Returns
        -------
        Tensor
            ``(n_episodes, n_contexts)`` of realization indices.
        """
        flat = policy.reshape(policy.shape[0], -1)
        best = torch.unravel_index(flat.argmax(-1), self.cfg.realization_shape)
        return torch.stack(best, 1).long()


def intrinsic_value(rates: Tensor, preferences: Tensor, eps: float = 1e-6) -> Tensor:
    """How well a set of observation rates matches what the chain wants to see.

    The geometric mean over channels of ``rate`` where the channel is preferred
    and ``1 - rate`` where it is not. A geometric mean, rather than a sum, means
    one badly-missed channel cannot be bought back by the others — the chain has
    to satisfy every preference at once.

    Parameters
    ----------
    rates:
        ``(n_episodes, n_observations)`` in [0, 1]. Either predicted by
        :class:`~rcc.generator.ObservationGenerator` or read off a known
        likelihood.
    preferences:
        ``(n_observations,)`` or ``(n_episodes, n_observations)`` in [0, 1] —
        1 for a channel the chain wants on, 0 for one it wants off.
    eps:
        Clamp on ``rates``, so a rate of exactly 0 or 1 cannot produce an
        infinite log.

    Returns
    -------
    Tensor
        ``(n_episodes,)`` in [0, 1].

    Examples
    --------
    Rates that match the preferences exactly score near 1:

    >>> rates = torch.tensor([[0.999, 0.001], [0.5, 0.5]])
    >>> preferences = torch.tensor([1.0, 0.0])
    >>> intrinsic_value(rates, preferences).round(decimals=2).tolist()
    [1.0, 0.5]

    Missing one channel drags the whole score down:

    >>> intrinsic_value(torch.tensor([[0.999, 0.999]]), preferences).item() < 0.05
    True
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
