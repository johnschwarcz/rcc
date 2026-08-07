"""Stage 3 — a realization back to observation rates.

The classifier maps observations to a belief. The generator maps a belief back
to the observations it would have produced, and the error in that round trip is
the *only* thing that trains the embeddings in stage 1. No labels are needed:
the chain checks its representation against the world it just saw.

The one place a label does enter is the goal variable. Where the chain has been
told whether it was right, that verdict replaces the confidence it would
otherwise have guessed — which is how a teaching signal on a handful of goal
variables ends up shaping a representation shared by all of them.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ._random import sample_categorical, seeded
from .config import RCCConfig

__all__ = ["ObservationGenerator", "query_from_belief"]


class ObservationGenerator(nn.Module):
    """Predict each channel's Bernoulli rate from a proposed joint realization.

    The realization and the confidence are shared across channels; the
    interactions are not. So a single hidden state is broadcast over channels
    and it is the interaction term that makes one channel's prediction differ
    from another's.

    Parameters
    ----------
    cfg:
        The chain's configuration.

    Examples
    --------
    >>> from rcc import RCCConfig
    >>> cfg = RCCConfig(n_contexts=2, n_realizations=4, n_observations=3,
    ...                 hidden_dim=16, seed=0)
    >>> generator = ObservationGenerator(cfg)
    >>> realizations = torch.tensor([[0, 3], [2, 1]])
    >>> confidence = torch.rand(2, 2)
    >>> interactions = torch.randn(2, 3, cfg.n_interactions)
    >>> rates = generator(realizations, confidence, interactions)
    >>> rates.shape
    torch.Size([2, 3])
    >>> bool(((rates > 0) & (rates < 1)).all())
    True
    """

    def __init__(self, cfg: RCCConfig) -> None:
        super().__init__()
        self.cfg = cfg
        with seeded(cfg.seed):
            self.realization_embedding = nn.Embedding(cfg.n_realizations, cfg.hidden_dim)
            self.realization_proj = nn.Linear(
                cfg.hidden_dim * cfg.n_contexts, cfg.hidden_dim
            )
            self.confidence_proj = nn.Linear(cfg.n_contexts, cfg.hidden_dim)
            self.interaction_proj = nn.Linear(cfg.n_interactions, cfg.hidden_dim)
            self.hidden = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
            self.rate = nn.Linear(cfg.hidden_dim, 1)

    def forward(
        self, realizations: Tensor, confidence: Tensor, interactions: Tensor
    ) -> Tensor:
        """Predict observation rates for a proposed realization.

        Parameters
        ----------
        realizations:
            ``(n_episodes, n_contexts)`` of realization indices — what the chain
            supposes the active variables are set to.
        confidence:
            ``(n_episodes, n_contexts)`` in [0, 1] — how much it believes that.
        interactions:
            ``(n_episodes, n_observations, n_interactions)``. **Not** detached:
            this is the path that trains the embeddings.

        Returns
        -------
        Tensor
            ``(n_episodes, n_observations)`` of Bernoulli rates.
        """
        self._check(realizations, confidence, interactions)
        n_episodes = realizations.shape[0]

        embedded = self.realization_embedding(realizations).reshape(n_episodes, -1)
        proposal = self.realization_proj(embedded).unsqueeze(1)
        believed = self.confidence_proj(confidence).unsqueeze(1)
        structure = self.interaction_proj(interactions)

        hidden = torch.relu(proposal + believed + structure)
        return torch.sigmoid(self.rate(torch.relu(self.hidden(hidden)))).squeeze(-1)

    def _check(
        self, realizations: Tensor, confidence: Tensor, interactions: Tensor
    ) -> None:
        cfg = self.cfg
        expected = (realizations.shape[0], cfg.n_contexts)
        for name, tensor in (("realizations", realizations), ("confidence", confidence)):
            if tuple(tensor.shape) != expected:
                raise ValueError(
                    f"{name} must have shape {expected}, got {tuple(tensor.shape)}"
                )
        wanted = (realizations.shape[0], cfg.n_observations, cfg.n_interactions)
        if tuple(interactions.shape) != wanted:
            raise ValueError(
                f"interactions must have shape {wanted}, got {tuple(interactions.shape)}"
            )


def query_from_belief(
    belief: Tensor,
    goal_index: Tensor,
    *,
    goal_selection: Tensor | None = None,
    goal_correct: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Turn a belief into the realization/confidence pair the generator wants.

    Each active variable's realization is *sampled* from the belief about it,
    and the confidence is the probability the belief assigned to whatever came
    up. Sampling rather than taking the mode is deliberate: it makes the
    generator see the whole belief over many episodes, not just its peak.

    Passing ``goal_selection`` and ``goal_correct`` supplies the teaching
    signal. For the goal variable only, the chain's committed answer replaces
    the sample and the verdict on that answer replaces the guessed confidence.
    Leave them out and the query is entirely self-supervised.

    Parameters
    ----------
    belief:
        ``(n_episodes, n_contexts, n_realizations)`` — one time slice, normally
        the last: ``belief[:, -1]``.
    goal_index:
        ``(n_episodes,)`` — which active variable is the goal.
    goal_selection:
        ``(n_episodes,)`` — the chain's committed realization for the goal.
    goal_correct:
        ``(n_episodes,)`` in [0, 1] — whether that answer was right.
    generator:
        Draws from the global RNG when ``None``.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``realizations`` and ``confidence``, both
        ``(n_episodes, n_contexts)``.

    Examples
    --------
    A belief certain of realization 2 everywhere samples 2 with confidence 1:

    >>> belief = torch.zeros(2, 2, 4)
    >>> belief[..., 2] = 1.0
    >>> realizations, confidence = query_from_belief(belief, torch.tensor([0, 1]))
    >>> realizations.tolist(), confidence.tolist()
    ([[2, 2], [2, 2]], [[1.0, 1.0], [1.0, 1.0]])

    The teaching signal overwrites the goal slot, and only the goal slot:

    >>> realizations, confidence = query_from_belief(
    ...     belief, torch.tensor([0, 1]),
    ...     goal_selection=torch.tensor([3, 3]),
    ...     goal_correct=torch.tensor([0.0, 0.0]))
    >>> realizations.tolist()
    [[3, 2], [2, 3]]
    >>> confidence.tolist()
    [[0.0, 1.0], [1.0, 0.0]]
    """
    if belief.ndim != 3:
        raise ValueError(
            "belief must have shape (n_episodes, n_contexts, n_realizations) — "
            f"index a single time step first. Got {tuple(belief.shape)}"
        )
    if (goal_selection is None) != (goal_correct is None):
        raise ValueError(
            "goal_selection and goal_correct go together: supply both to teach "
            "the goal variable, or neither to stay self-supervised."
        )

    realizations = sample_categorical(belief, generator)
    confidence = belief.gather(-1, realizations.unsqueeze(-1)).squeeze(-1)

    if goal_selection is not None and goal_correct is not None:
        episodes = torch.arange(belief.shape[0], device=belief.device)
        realizations = realizations.clone()
        confidence = confidence.clone()
        realizations[episodes, goal_index] = goal_selection.to(realizations.dtype)
        confidence[episodes, goal_index] = goal_correct.to(confidence.dtype)
    return realizations, confidence
