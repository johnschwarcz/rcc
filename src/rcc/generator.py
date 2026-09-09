"""Stage 3 — a realization back to observation rates."""
import torch
from torch import Tensor, nn
from ._helpers import require_shape, sample_categorical, seeded
from .config import RCCConfig

__all__ = ["ObservationGenerator", "query_from_belief"]

class ObservationGenerator(nn.Module):
    """Predict each channel conditioned on a joint realization."""

    def __init__(self, cfg: RCCConfig) -> None:
        super().__init__()
        self.cfg = cfg
        with seeded(cfg.seed):
            self.realization_embedding = nn.Embedding(cfg.n_realizations, cfg.hidden_dim)
            self.realization_proj = nn.Linear(cfg.hidden_dim * cfg.n_contexts, cfg.hidden_dim)
            self.interaction_proj = nn.Linear(cfg.n_interactions, cfg.hidden_dim)
            self.confidence_proj = nn.Linear(cfg.n_contexts, cfg.hidden_dim)
            self.hidden = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
            self.rate = nn.Linear(cfg.hidden_dim, 1)

    def forward(self, ctx_vals: Tensor, confidence: Tensor, interactions: Tensor) -> Tensor:
        """Predict observation rates for a proposed realization.
        ctx_vals, confidence: (n_episodes, n_contexts), confidence in [0, 1]
        interactions: (n_episodes, n_observations, n_interactions), undetached
        returns: (n_episodes, n_observations) of Bernoulli rates
        """
        self._check(ctx_vals, confidence, interactions)
        n_episodes = ctx_vals.shape[0]

        embedded = self.realization_embedding(ctx_vals).reshape(n_episodes, -1)
        proposal = self.realization_proj(embedded).unsqueeze(1)
        believed = self.confidence_proj(confidence).unsqueeze(1)
        structure = self.interaction_proj(interactions)

        mixed = torch.relu(proposal + believed + structure)
        hidden = torch.relu(self.hidden(mixed))
        return torch.sigmoid(self.rate(hidden)).squeeze(-1)

    def _check(self, ctx_vals: Tensor, confidence: Tensor, interactions: Tensor) -> None:
        cfg = self.cfg
        n_episodes = ctx_vals.shape[0]
        require_shape("ctx_vals", ctx_vals, (n_episodes, cfg.n_contexts))
        require_shape("confidence", confidence, (n_episodes, cfg.n_contexts))
        require_shape("interactions", interactions,
            (n_episodes, cfg.n_observations, cfg.n_interactions))

def query_from_belief(belief: Tensor, goal_ind: Tensor, *, goal_selection: Tensor | None = None,
        goal_correct: Tensor | None = None, generator: torch.Generator | None = None) -> tuple[Tensor, Tensor]:
    """Turn a belief into the realization/confidence pair the generator wants.
    belief: (n_episodes, n_contexts, n_realizations) (normally final timestep)
    goal_ind, goal_selection, goal_correct: (n_episodes,)
    returns: ctx_vals, confidence, each of shape (n_episodes, n_contexts)
    """

    require_shape("belief", belief, (None, None, None))
    if (goal_selection is None) != (goal_correct is None):
        raise ValueError("supply both or neither goal_selection & goal_correct.")

    ctx_vals = sample_categorical(belief, generator)
    confidence = belief.gather(-1, ctx_vals.unsqueeze(-1)).squeeze(-1)

    if goal_selection is not None and goal_correct is not None:
        episodes = torch.arange(belief.shape[0], device=belief.device)
        ctx_vals = ctx_vals.clone()
        confidence = confidence.clone()
        ctx_vals[episodes, goal_ind] = goal_selection.to(ctx_vals.dtype)
        confidence[episodes, goal_ind] = goal_correct.to(confidence.dtype)
    return ctx_vals, confidence
