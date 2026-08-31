"""The four stages, assembled. Each is a plain ``nn.Module``, importable alone."""
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
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

    @property
    def device(self) -> torch.device:
        """Where the chain's parameters actually are, which is worth checking against
        where you meant to put them.
        >>> RCC(RCCConfig(n_vars=8, hidden_dim=16, seed=0)).device
        device(type='cpu')
        """
        return next(self.parameters()).device

    # ------------------------------------------------------- saving and loading
    def save(self, path: str | Path) -> None:
        """Write the config beside the parameters, so ``load`` needs nothing else.
        Reproducing a run by reloading it beats reproducing it by seeding: the seed
        only reproduces a chain if every version between then and now agrees.
        """
        torch.save({"cfg": asdict(self.cfg), "state": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str | Path, map_location: str | None = None) -> "RCC":
        """Rebuild a chain that :meth:`save` wrote, fixed embeddings included.
        Those are buffers, so they come back with the rest of the state rather than
        having to be supplied again.
        """
        saved = torch.load(path, map_location=map_location, weights_only=True)
        cfg = RCCConfig(**saved["cfg"])
        placeholder = {}
        if not cfg.learn_embeddings:
            shape = (cfg.n_vars, cfg.n_observations, cfg.embedding_dim)
            placeholder = {"keys": torch.zeros(shape), "queries": torch.zeros(shape)}
        chain = cls(cfg, **placeholder)
        chain.load_state_dict(saved["state"])
        return chain

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