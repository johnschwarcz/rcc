"""Stage 1 — variables to interactions.
Each variable owns a key and a query per observation channel.
"""
from typing import NamedTuple
import torch
from torch import Tensor, nn
from ._helpers import require_shape, seeded
from .config import RCCConfig

__all__ = ["Interaction", "InteractionEncoder", "ordered_pairs"]

class Interaction(NamedTuple):
    """Stage 1. Unpacks as score, keys, queries.
    score: (n_episodes, n_observations, n_interactions)
    keys, queries: (n_episodes, n_contexts, n_observations, embedding_dim)
    """
    score: Tensor
    keys: Tensor
    queries: Tensor

def ordered_pairs(n_contexts: int) -> list[tuple[int, int]]:
    """Which (key, query) context pairs become interactions, in order.
    A single active variable has no pair, so it interacts with itself.
    """
    pairs = [pair
        for i in range(n_contexts)
        for j in range(i + 1, n_contexts)
        for pair in ((i, j), (j, i))]
    return pairs or [(0, 0)]

class InteractionEncoder(nn.Module):
    """Active variables' embeddings --> pairwise scores.
    learned keys, queries: (n_vars, n_observations, hidden_dim) are projected to embedding_dim.
    cfg.learn_embeddings = False --> supply keys, queries: (n_vars, n_observations, embedding_dim)
    """
    # Declared so that a type checker knows these are tensors.
    keys: Tensor
    queries: Tensor
    key_index: Tensor
    query_index: Tensor

    def __init__(self, cfg: RCCConfig, *, keys: Tensor | None = None, queries: Tensor | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        if cfg.learn_embeddings:
            if keys is not None or queries is not None:
                raise ValueError("keys/queries were supplied but cfg.learn_embeddings is True.")
            with seeded(cfg.seed):
                raw = (cfg.n_vars, cfg.n_observations, cfg.hidden_dim)
                self.keys = nn.Parameter(_unit_rows(torch.randn(raw)))
                self.queries = nn.Parameter(_unit_rows(torch.randn(raw)))
                self.key_proj = nn.Linear(cfg.hidden_dim, cfg.embedding_dim)
                self.query_proj = nn.Linear(cfg.hidden_dim, cfg.embedding_dim)
        else:
            if keys is None or queries is None:
                raise ValueError("keys/queries must be supplied when cfg.learn_embeddings is False.")
            expected = (cfg.n_vars, cfg.n_observations, cfg.embedding_dim)
            require_shape("keys", keys, expected)
            require_shape("queries", queries, expected)
            self.register_buffer("keys", keys.detach().clone())
            self.register_buffer("queries", queries.detach().clone())

        pairs = ordered_pairs(cfg.n_contexts)
        self.register_buffer("key_index", torch.tensor([k for k, _ in pairs]))
        self.register_buffer("query_index", torch.tensor([q for _, q in pairs]))

    def forward(self, ctx_inds: Tensor) -> Interaction:
        """Encode a batch of active variables.
        ctx_inds: (n_episodes, n_contexts) of variable indices; (repeats allowed).
        returns: Interaction {score, keys, queries}
        """
        require_shape("ctx_inds", ctx_inds, (None, self.cfg.n_contexts))

        keys = self.keys[ctx_inds]
        queries = self.queries[ctx_inds]
        if self.cfg.learn_embeddings:
            keys = self.key_proj(keys)
            queries = self.query_proj(queries)
        paired_keys = keys[:, self.key_index]
        paired_queries = queries[:, self.query_index]
        score = (paired_keys * paired_queries).sum(-1)
        return Interaction(score.transpose(1, 2), keys, queries)

    def extra_repr(self) -> str:
        source = "learned" if self.cfg.learn_embeddings else "fixed"
        return (f"n_vars={self.cfg.n_vars}, n_observations={self.cfg.n_observations}, "
            f"embedding_dim={self.cfg.embedding_dim}, {source}")

def _unit_rows(x: Tensor) -> Tensor:
    return x / x.norm(dim=-1, keepdim=True) # Scale to unit norm.
