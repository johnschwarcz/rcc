"""The four stages, assembled.

:class:`RCC` is a container, not a framework. It holds the stages, runs the main
path, and knows which parameters belong to which objective. It does not own a
training loop, a device policy, or an optimizer — those are yours.

If you only want one stage, import it directly. Every stage is a plain
``nn.Module`` that takes and returns tensors, and none of them reference this
class.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import NamedTuple

import torch
from torch import Tensor, nn

from .classifier import BeliefClassifier
from .config import RCCConfig
from .controller import Controller
from .generator import ObservationGenerator, query_from_belief
from .interactions import Interaction, InteractionEncoder

__all__ = ["RCC", "ChainOutput"]


class ChainOutput(NamedTuple):
    """What a forward pass through stages 1-2 produces.

    Attributes
    ----------
    belief:
        ``(n_episodes, n_steps, n_contexts, n_realizations)``.
    interaction:
        The :class:`~rcc.interactions.Interaction` the belief was conditioned
        on. Stages 3 and 4 both need it, and it is returned rather than
        recomputed.
    """

    belief: Tensor
    interaction: Interaction


class RCC(nn.Module):
    """A Representation Classification Chain.

    Parameters
    ----------
    cfg:
        The chain's configuration.
    keys, queries:
        Fixed embeddings for stage 1, each
        ``(n_vars, n_observations, embedding_dim)``. Required when
        ``cfg.learn_embeddings`` is ``False``.

    Examples
    --------
    >>> from rcc import RCCConfig
    >>> cfg = RCCConfig(n_vars=50, n_contexts=2, n_realizations=4,
    ...                 n_observations=3, hidden_dim=16, seed=0)
    >>> chain = RCC(cfg)
    >>> observations = torch.rand(8, 5, 3).round()
    >>> var_ids = torch.randint(0, 50, (8, 2))
    >>> belief, interaction = chain(observations, var_ids)
    >>> belief.shape, interaction.strength.shape
    (torch.Size([8, 5, 2, 4]), torch.Size([8, 3, 2]))

    The three parameter groups partition the chain, which is what lets the
    objectives be optimized independently:

    >>> groups = [chain.estimation_parameters(), chain.inference_parameters(),
    ...           chain.control_parameters()]
    >>> counted = sum(len(list(g)) for g in groups)
    >>> counted == len(list(chain.parameters()))
    True
    """

    def __init__(
        self,
        cfg: RCCConfig,
        *,
        keys: Tensor | None = None,
        queries: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = InteractionEncoder(cfg, keys=keys, queries=queries)
        self.classifier = BeliefClassifier(cfg)
        self.generator = ObservationGenerator(cfg)
        self.controller = Controller(cfg)

    def forward(self, observations: Tensor, var_ids: Tensor) -> ChainOutput:
        """Run stages 1 and 2: variables to interactions to a belief.

        Parameters
        ----------
        observations:
            ``(n_episodes, n_steps, n_observations)``.
        var_ids:
            ``(n_episodes, n_contexts)`` of indices into the variable pool.
        """
        interaction = self.encoder(var_ids)
        belief = self.classifier(observations, interaction.strength)
        return ChainOutput(belief, interaction)

    def reconstruct(
        self,
        belief: Tensor,
        interaction: Interaction,
        goal_index: Tensor,
        *,
        goal_selection: Tensor | None = None,
        goal_correct: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Run stage 3: predict the rates the belief's final answer implies.

        Reads the last time slice of ``belief``, since that is the chain's
        settled answer — reconstructing from an earlier slice would score the
        representation against a belief the chain has already moved past.

        Parameters
        ----------
        belief:
            ``(n_episodes, n_steps, n_contexts, n_realizations)`` from
            :meth:`forward`.
        interaction:
            From :meth:`forward`. Passed to the generator *undetached*, so this
            is the call that trains the embeddings.
        goal_index:
            ``(n_episodes,)`` — which active variable is the goal.
        goal_selection, goal_correct:
            The teaching signal, supplied together or not at all. See
            :func:`~rcc.generator.query_from_belief`.
        generator:
            RNG for the realization sample.

        Returns
        -------
        Tensor
            ``(n_episodes, n_observations)`` of predicted rates.
        """
        realizations, confidence = query_from_belief(
            belief[:, -1].detach(),
            goal_index,
            goal_selection=goal_selection,
            goal_correct=goal_correct,
            generator=generator,
        )
        return self.generator(realizations, confidence, interaction.strength)

    # ------------------------------------------------------ parameter groups
    def estimation_parameters(self) -> Iterator[nn.Parameter]:
        """Parameters trained by the *prediction* objective — stages 1 and 3.

        The embeddings live here rather than with the classifier, and that
        placement is the architecture: what the chain represents is shaped by
        how well it predicts the world, never by how conveniently it classifies.
        """
        yield from self.encoder.parameters()
        yield from self.generator.parameters()

    def inference_parameters(self) -> Iterator[nn.Parameter]:
        """Parameters trained by a *classification* objective — stage 2.

        Under ``cfg.reservoir`` the recurrent weights are frozen, so they are
        absent from this group by way of ``requires_grad``, not by exclusion.
        """
        yield from self.classifier.parameters()

    def control_parameters(self) -> Iterator[nn.Parameter]:
        """Parameters trained by the *control* objective — stage 4."""
        yield from self.controller.parameters()
