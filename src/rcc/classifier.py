"""Stage 2 — observations and interactions to a belief.

The readout does not emit a belief. It emits an *increment*, and the belief is
the running sum of increments passed through a softmax. Since a softmax over a
sum of logits is a product of the corresponding likelihood ratios, the network
accumulates evidence multiplicatively while only ever having to learn one step
of it. Nothing constrains it to be Bayesian; the architecture just makes the
Bayesian solution the easy one to represent.

The interactions arrive detached. That is the seam the paper's claim rests on:
whatever the classifier learns, it cannot reshape the representation it was
handed to make its own job easier.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ._random import sample_categorical, seeded
from .config import RCCConfig

__all__ = ["BeliefClassifier", "goal_accuracy", "sample_goal", "select_goal"]


class BeliefClassifier(nn.Module):
    """Read the observation stream and emit a belief per active variable.

    The belief is *factorized*: one distribution over realizations for each of
    the ``n_contexts`` active variables, rather than one distribution over their
    joint realization. An observer that models the interactions can still beat a
    factorized posterior, and the size of that gap is what the environment is
    built to expose.

    Parameters
    ----------
    cfg:
        The chain's configuration. ``cfg.recurrent`` selects the architecture
        and ``cfg.reservoir`` freezes the recurrent weights.

    Examples
    --------
    >>> from rcc import RCCConfig
    >>> cfg = RCCConfig(n_contexts=2, n_realizations=4, n_observations=3,
    ...                 hidden_dim=16, seed=0)
    >>> classifier = BeliefClassifier(cfg)
    >>> observations = torch.rand(8, 5, 3).round()
    >>> interactions = torch.randn(8, 3, cfg.n_interactions)
    >>> belief = classifier(observations, interactions)
    >>> belief.shape
    torch.Size([8, 5, 2, 4])

    Every row is a distribution:

    >>> bool(torch.allclose(belief.sum(-1), torch.ones(8, 5, 2), atol=1e-5))
    True

    A reservoir freezes the recurrence but not the layers around it:

    >>> reservoir = BeliefClassifier(cfg.replace(reservoir=True))
    >>> any(p.requires_grad for p in reservoir.rnn.parameters())
    False
    >>> all(p.requires_grad for p in reservoir.readin.parameters())
    True
    """

    def __init__(self, cfg: RCCConfig) -> None:
        super().__init__()
        self.cfg = cfg
        with seeded(cfg.seed):
            self.readin = nn.Linear(cfg.classifier_input_dim, cfg.hidden_dim)
            self.readout = nn.Linear(
                cfg.hidden_dim, cfg.n_contexts * cfg.n_realizations
            )
            if cfg.recurrent:
                self.rnn = nn.LSTM(cfg.hidden_dim, cfg.hidden_dim, batch_first=True)
                # Learned initial state: what the chain believes before it has
                # seen anything. Trained even in a reservoir.
                self.initial_short_term = nn.Parameter(torch.randn(1, 1, cfg.hidden_dim))
                self.initial_long_term = nn.Parameter(torch.randn(1, 1, cfg.hidden_dim))
                if cfg.reservoir:
                    self.rnn.requires_grad_(False)

    def forward(self, observations: Tensor, interactions: Tensor) -> Tensor:
        """Integrate ``observations`` in the context of ``interactions``.

        Parameters
        ----------
        observations:
            ``(n_episodes, n_steps, n_observations)``. One binary sample per
            channel per step.
        interactions:
            ``(n_episodes, n_observations, n_interactions)`` from
            :class:`~rcc.interactions.InteractionEncoder`. Detached on the way
            in, so no gradient reaches the embeddings through this path.

        Returns
        -------
        Tensor
            ``(n_episodes, n_steps, n_contexts, n_realizations)``, normalized
            over the last axis.
        """
        self._check(observations, interactions)
        n_episodes, n_steps, _ = observations.shape
        context = interactions.detach().reshape(n_episodes, 1, -1)

        if not self.cfg.recurrent:
            # No sequential integration: one readout of the episode mean, held
            # constant over time so the output stays shape-compatible.
            hidden = torch.relu(
                self.readin(torch.cat((observations.mean(1, keepdim=True), context), -1))
            )
            increments = self.readout(hidden).expand(-1, n_steps, -1)
        else:
            per_step = torch.cat((observations, context.expand(-1, n_steps, -1)), -1)
            hidden = torch.relu(self.readin(per_step))
            short_term = self.initial_short_term.expand(-1, n_episodes, -1).contiguous()
            long_term = self.initial_long_term.expand(-1, n_episodes, -1).contiguous()
            recurrent, _ = self.rnn(hidden, (short_term, long_term))
            increments = self.readout(recurrent)

        increments = increments.reshape(
            n_episodes, n_steps, self.cfg.n_contexts, self.cfg.n_realizations
        )
        # The whole architectural idea, in one line: accumulate, then normalize.
        logits = increments.cumsum(1) if self.cfg.recurrent else increments
        return torch.softmax(logits, -1)

    def _check(self, observations: Tensor, interactions: Tensor) -> None:
        cfg = self.cfg
        if observations.ndim != 3 or observations.shape[2] != cfg.n_observations:
            raise ValueError(
                "observations must have shape (n_episodes, n_steps, "
                f"{cfg.n_observations}), got {tuple(observations.shape)}"
            )
        expected = (observations.shape[0], cfg.n_observations, cfg.n_interactions)
        if tuple(interactions.shape) != expected:
            raise ValueError(
                f"interactions must have shape {expected}, got "
                f"{tuple(interactions.shape)}"
            )


def select_goal(belief: Tensor, goal_index: Tensor) -> Tensor:
    """Pick out the belief over the one variable that is being asked about.

    Parameters
    ----------
    belief:
        ``(n_episodes, n_steps, n_contexts, n_realizations)``.
    goal_index:
        ``(n_episodes,)`` — which of the active variables is the goal, per
        episode.

    Returns
    -------
    Tensor
        ``(n_episodes, n_steps, n_realizations)``.

    Examples
    --------
    >>> belief = torch.arange(2 * 1 * 2 * 3).float().reshape(2, 1, 2, 3)
    >>> select_goal(belief, torch.tensor([0, 1]))
    tensor([[[ 0.,  1.,  2.]],
    <BLANKLINE>
            [[ 9., 10., 11.]]])
    """
    if goal_index.ndim != 1 or goal_index.shape[0] != belief.shape[0]:
        raise ValueError(
            f"goal_index must have shape ({belief.shape[0]},), got "
            f"{tuple(goal_index.shape)}"
        )
    episodes = torch.arange(belief.shape[0], device=belief.device)
    return belief[episodes, :, goal_index]


def sample_goal(
    goal_belief: Tensor, generator: torch.Generator | None = None
) -> Tensor:
    """Draw a realization for the goal variable from the belief about it.

    The chain commits to a sample rather than an argmax, which is what lets the
    reward objective in :mod:`rcc.losses` shape the belief's *spread* and not
    just its peak.

    Parameters
    ----------
    goal_belief:
        ``(n_episodes, n_steps, n_realizations)`` from :func:`select_goal`.
    generator:
        Draws from the global RNG when ``None``.

    Returns
    -------
    Tensor
        ``(n_episodes, n_steps)`` of realization indices.

    Examples
    --------
    >>> certain = torch.zeros(2, 3, 4)
    >>> certain[..., 2] = 1.0
    >>> sample_goal(certain)
    tensor([[2, 2, 2],
            [2, 2, 2]])
    """
    return sample_categorical(goal_belief, generator)


def goal_accuracy(selection: Tensor, goal_value: Tensor) -> Tensor:
    """Whether each sampled realization matched the truth, as a float mask.

    Parameters
    ----------
    selection:
        ``(n_episodes, n_steps)`` from :func:`sample_goal`.
    goal_value:
        ``(n_episodes,)`` — the goal variable's true realization.

    Returns
    -------
    Tensor
        ``(n_episodes, n_steps)`` of 1.0 where the sample was right.

    Examples
    --------
    >>> goal_accuracy(torch.tensor([[1, 2], [0, 0]]), torch.tensor([2, 0]))
    tensor([[0., 1.],
            [1., 1.]])
    """
    if goal_value.ndim != 1 or goal_value.shape[0] != selection.shape[0]:
        raise ValueError(
            f"goal_value must have shape ({selection.shape[0]},), got "
            f"{tuple(goal_value.shape)}"
        )
    return (selection == goal_value[:, None]).float()
