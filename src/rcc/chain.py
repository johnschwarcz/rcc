"""The four stages, assembled. Each is a plain ``nn.Module``, importable alone."""
from collections.abc import Iterator
import torch
from torch import Tensor, nn
from .classifier import BeliefClassifier
from .config import RCCConfig
from .controller import Controller
from .generator import ObservationGenerator, query_from_belief
from .interactions import Interaction, InteractionEncoder

__all__ = ["RCC"]

class RCC(nn.Module):
    """A Representation Classification Chain.
    keys and queries supply embeddings of shape (n_vars, n_observations, embedding_dim) 
    """

    def __init__(self, cfg: RCCConfig, *,
        keys: Tensor | None = None, queries: Tensor | None = None,) -> None:

        super().__init__()
        self.cfg = cfg
        self.encoder = InteractionEncoder(cfg, keys=keys, queries=queries)
        self.classifier = BeliefClassifier(cfg)
        self.generator = ObservationGenerator(cfg)
        self.controller = Controller(cfg)

    def forward(self, observations: Tensor, ctx_inds: Tensor) -> tuple[Tensor, Interaction]:
        """Stages 1 and 2: variables to interactions to a belief.
        observations: (n_episodes, n_steps, n_observations)
        ctx_inds: (n_episodes, n_contexts) of active variable indices.
        Returns: (n_episodes, n_steps, n_contexts, n_realizations), Interaction
        """
        interaction = self.encoder(ctx_inds)
        belief = self.classifier(observations, interaction.score)
        return belief, interaction

    def reconstruct(self, belief: Tensor, interaction: Interaction, goal_ind: Tensor, *,
        goal_selection: Tensor | None = None, goal_correct: Tensor | None = None,
        generator: torch.Generator | None = None,) -> Tensor:
        """Stage 3: return the (n_episodes, n_observations) predicted rates."""
        ctx_vals, confidence = query_from_belief(
            belief[:, -1].detach(), 
            goal_ind,
            goal_selection=goal_selection,
            goal_correct=goal_correct,
            generator=generator,)
        return self.generator(ctx_vals, confidence, interaction.score)

    # ------------------------------------------------------ parameter groups
    def estimation_parameters(self) -> Iterator[nn.Parameter]:
        """Trained by the *prediction* objective — stages 1 and 3."""
        yield from self.encoder.parameters()
        yield from self.generator.parameters()

    def inference_parameters(self) -> Iterator[nn.Parameter]:
        """Trained by a *classification* objective — stage 2."""
        yield from self.classifier.parameters()

    def control_parameters(self) -> Iterator[nn.Parameter]:
        """Trained by the *control* objective — stage 4."""
        yield from self.controller.parameters()