"""A minimal task with genuine interaction structure, for demonstrations.

This exists so that the package can be run, tested and shown off on its own.
It is *not* the environment from the paper — that is
`coggrid <https://github.com/johnschwarcz/coggrid>`_, which models the
generative process properly, ships the ideal-observer baselines, and is what
any real experiment should use. What is here is the smallest thing that gives a
chain something worth learning:

* every variable owns key and query embeddings, so pairs of them interact;
* an interaction sets the *phase* of the profile relating a variable's
  realization to the observation rates, so the rate for one variable depends on
  which other variable it was drawn with;
* observations are Bernoulli samples of those rates, so evidence accumulates
  over an episode and a sequential observer can beat a one-shot one.

Because the realization space is small enough to enumerate, the exact posterior
comes free, which is what :func:`ToyTask.sample` returns alongside the data.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import Tensor

from ._random import seeded
from .config import RCCConfig

__all__ = ["Episode", "ToyTask"]


class Episode(NamedTuple):
    """One batch of episodes, and the answers to them.

    Attributes
    ----------
    observations:
        ``(n_episodes, n_steps, n_observations)`` of 0/1 samples.
    var_ids:
        ``(n_episodes, n_contexts)`` — which variables were active.
    realizations:
        ``(n_episodes, n_contexts)`` — the value each took. The truth.
    goal_index:
        ``(n_episodes,)`` — which active variable is being asked about.
    goal_value:
        ``(n_episodes,)`` — that variable's true realization.
    posterior:
        ``(n_episodes, n_steps, n_contexts, n_realizations)`` — the exact
        marginal posterior after each step, from an observer that knows the
        interactions. The target :func:`~rcc.losses.distillation_loss` wants,
        and the ceiling a chain is trying to reach.
    rates:
        ``(n_episodes, n_observations)`` — the true Bernoulli rates the
        observations were drawn from.
    """

    observations: Tensor
    var_ids: Tensor
    realizations: Tensor
    goal_index: Tensor
    goal_value: Tensor
    posterior: Tensor
    rates: Tensor


class ToyTask:
    """A small latent-variable task whose likelihood does not factorize.

    Parameters
    ----------
    cfg:
        Supplies ``n_vars``, ``n_contexts``, ``n_realizations``,
        ``n_observations`` and ``embedding_dim``. Nothing else is read, and the
        chain that consumes the task need not share the config.
    temperature:
        Scales the potential before the sigmoid. Larger values push rates
        towards 0/1, making a single observation more informative.
    seed:
        Seeds the embeddings, which are fixed for the life of the task.

    Attributes
    ----------
    keys, queries:
        ``(n_vars, n_observations, embedding_dim)``. Hand these to
        :class:`~rcc.interactions.InteractionEncoder` with
        ``learn_embeddings=False`` to give a chain the true representation, or
        withhold them and make it find its own.

    Examples
    --------
    >>> from rcc import RCCConfig
    >>> cfg = RCCConfig(n_vars=8, n_contexts=2, n_realizations=3,
    ...                 n_observations=2, embedding_dim=4)
    >>> task = ToyTask(cfg, seed=0)
    >>> episode = task.sample(5, n_steps=6, generator=torch.Generator().manual_seed(0))
    >>> episode.observations.shape
    torch.Size([5, 6, 2])
    >>> episode.posterior.shape
    torch.Size([5, 6, 2, 3])

    The posterior is a distribution, and it knows the answer better than chance:

    >>> bool(torch.allclose(episode.posterior.sum(-1), torch.ones(5, 6, 2), atol=1e-5))
    True
    """

    def __init__(
        self, cfg: RCCConfig, *, temperature: float = 2.0, seed: int | None = None
    ) -> None:
        self.cfg = cfg
        self.temperature = temperature
        with seeded(seed):
            shape = (cfg.n_vars, cfg.n_observations, cfg.embedding_dim)
            self.keys = _unit_rows(torch.randn(shape))
            self.queries = _unit_rows(torch.randn(shape))

    def rate_table(self, var_ids: Tensor) -> Tensor:
        """Bernoulli rates for every joint realization of the active variables.

        Parameters
        ----------
        var_ids:
            ``(n_episodes, n_contexts)``.

        Returns
        -------
        Tensor
            ``(n_episodes, n_observations, *realization_shape)`` in (0, 1).
        """
        cfg = self.cfg
        n_episodes = var_ids.shape[0]
        keys = self.keys[var_ids]
        queries = self.queries[var_ids]
        grid = torch.arange(cfg.n_realizations, dtype=torch.float32)

        def profile(i: int, j: int) -> Tensor:
            """Variable ``i``'s potential over its own realizations, with the
            phase set by how its key meets variable ``j``'s query.

            Two harmonics, not one. A single sinusoid is two-to-one over a
            period, so some pairs of realizations would share a potential
            exactly and *no* observer could tell them apart; the second harmonic
            breaks that reflection and makes the profile injective.
            """
            strength = (keys[:, i] * queries[:, j]).sum(-1)
            phase = 2 * math.pi * (grid / cfg.n_realizations + strength[..., None])
            return torch.sin(phase) + 0.5 * torch.sin(2 * phase)

        logits = torch.zeros(n_episodes, cfg.n_observations, *cfg.realization_shape)
        if cfg.n_contexts == 1:
            logits = logits + profile(0, 0)
        else:
            for i in range(cfg.n_contexts):
                for j in range(i + 1, cfg.n_contexts):
                    # An outer *product* of the two profiles, not a sum. A sum
                    # would leave the log-odds additive across realizations, and
                    # a factorized observer would lose nothing by ignoring the
                    # coupling — which is exactly the effect worth studying.
                    pair = torch.einsum("bom,bon->bomn", profile(i, j), profile(j, i))
                    shape = [1] * cfg.n_contexts
                    shape[i] = shape[j] = cfg.n_realizations
                    logits = logits + pair.reshape(
                        n_episodes, cfg.n_observations, *shape
                    )
        return torch.sigmoid(self.temperature * logits)

    def sample(
        self,
        n_episodes: int,
        n_steps: int = 30,
        generator: torch.Generator | None = None,
    ) -> Episode:
        """Draw a batch of episodes, with their exact posteriors.

        Parameters
        ----------
        n_episodes:
            How many episodes to draw.
        n_steps:
            Observations per episode.
        generator:
            Draws from the global RNG when ``None``.
        """
        cfg = self.cfg
        shape = (n_episodes, cfg.n_contexts)
        var_ids = torch.randint(0, cfg.n_vars, shape, generator=generator)
        realizations = torch.randint(0, cfg.n_realizations, shape, generator=generator)
        goal_index = torch.randint(
            0, cfg.n_contexts, (n_episodes,), generator=generator
        )

        table = self.rate_table(var_ids)
        flat = table.reshape(n_episodes, cfg.n_observations, -1)
        chosen = _flatten_realizations(realizations, cfg.n_realizations)
        rates = flat.gather(-1, chosen[:, None, None].expand(-1, cfg.n_observations, 1))
        rates = rates.squeeze(-1)

        draws = torch.rand(
            n_episodes, n_steps, cfg.n_observations, generator=generator
        )
        observations = (draws < rates[:, None, :]).float()

        episodes = torch.arange(n_episodes)
        return Episode(
            observations=observations,
            var_ids=var_ids,
            realizations=realizations,
            goal_index=goal_index,
            goal_value=realizations[episodes, goal_index],
            posterior=self.posterior(observations, table),
            rates=rates,
        )

    def posterior(self, observations: Tensor, table: Tensor) -> Tensor:
        """The exact marginal posterior after each step.

        Enumerates every joint realization, which is affordable precisely
        because this task is small. A real environment cannot generally do this,
        which is why the chain is worth training at all.

        Parameters
        ----------
        observations:
            ``(n_episodes, n_steps, n_observations)``.
        table:
            ``(n_episodes, n_observations, *realization_shape)`` from
            :meth:`rate_table`.

        Returns
        -------
        Tensor
            ``(n_episodes, n_steps, n_contexts, n_realizations)``.
        """
        cfg = self.cfg
        n_episodes, n_steps, _ = observations.shape
        flat = table.reshape(n_episodes, cfg.n_observations, -1)

        # Log likelihood of each step under every joint realization, accumulated.
        on = torch.einsum("bto,boj->btj", observations, flat.clamp_min(1e-12).log())
        off = torch.einsum(
            "bto,boj->btj", 1 - observations, (1 - flat).clamp_min(1e-12).log()
        )
        joint = torch.softmax((on + off).cumsum(1), -1)
        joint = joint.reshape(n_episodes, n_steps, *cfg.realization_shape)

        # Marginalize onto each active variable in turn.
        axes = range(2, 2 + cfg.n_contexts)
        marginals = [joint.sum(dim=[a for a in axes if a != keep]) for keep in axes]
        return torch.stack(marginals, 2)


def _unit_rows(x: Tensor) -> Tensor:
    return x / x.norm(dim=-1, keepdim=True)


def _flatten_realizations(realizations: Tensor, n_realizations: int) -> Tensor:
    """Row-major index of a joint realization, matching ``Tensor.reshape``."""
    flat = torch.zeros(realizations.shape[0], dtype=torch.long)
    for context in range(realizations.shape[1]):
        flat = flat * n_realizations + realizations[:, context]
    return flat
