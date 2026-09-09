"""The four stages, assembled. Each is a plain ``nn.Module``, importable alone."""
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
import torch
from torch import Tensor, nn
from ._helpers import resolve_device
from .classifier import BeliefClassifier
from .config import RCCConfig
from .controller import Controller
from .generator import ObservationGenerator, query_from_belief
from .interactions import Interaction, InteractionEncoder

__all__ = ["RCC"]

class RCC(nn.Module):
    """ A Representation Classification Chain. 
    initialize with:
        chain = RCC(cfg) 
    trained with:
        Trainer(chain).step()
    """

    def __init__(self, cfg: RCCConfig, *, keys: Tensor | None = None, queries: Tensor | None = None) -> None:

        super().__init__()
        self.cfg = cfg
        self.encoder = InteractionEncoder(cfg, keys=keys, queries=queries)
        self.classifier = BeliefClassifier(cfg)
        self.generator = ObservationGenerator(cfg)
        self.controller = Controller(cfg)
        if cfg.device is not None:
            self.to(resolve_device(cfg.device))

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
            generator: torch.Generator | None = None) -> Tensor:
        """Stage 3: return the (n_episodes, n_observations) predicted rates."""
        ctx_vals, confidence = query_from_belief(belief[:, -1].detach(), goal_ind,
            goal_selection=goal_selection, goal_correct=goal_correct, generator=generator,)
        return self.generator(ctx_vals, confidence, interaction.score)

    @property
    def device(self) -> torch.device:
        """Where the chain's parameters actually are"""
        return next(self.parameters()).device

    # ------------------------------------------------------- saving and loading
    def save(self, path: str | Path) -> None:
        """Write the config beside the parameters, so ``load`` needs nothing else."""
        torch.save({"cfg": asdict(self.cfg), "state": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str | Path, map_location: str | None = None) -> "RCC":
        """Rebuild a chain that :meth:`save` wrote"""
        saved = torch.load(path, map_location="cpu", weights_only=True)
        return cls.from_saved(saved, map_location)

    @classmethod
    def from_saved(cls, saved: dict, map_location: str | None = None) -> "RCC":
        """Rebuild from an already-read payload of ``{"cfg": ..., "state": ...}``.
        Split out from :meth:`load` so that a checkpoint carrying more than a chain —
        see :func:`rcc.save_run` — reconstructs it exactly the same way.
        """
        cfg = RCCConfig(**saved["cfg"])
        placeholder = {}
        if not cfg.learn_embeddings:
            shape = (cfg.n_vars, cfg.n_observations, cfg.embedding_dim)
            placeholder = {"keys": torch.zeros(shape), "queries": torch.zeros(shape)}
        chain = cls(cfg.replace(device=None), **placeholder)
        chain.load_state_dict(saved["state"])
        chain.cfg = cfg
        return chain.to(map_location if map_location is not None else resolve_device(cfg.device))

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