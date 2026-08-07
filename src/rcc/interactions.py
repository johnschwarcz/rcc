"""Stage 1 — variables to interactions.

Each latent variable owns a *key* and a *query* embedding per observation
channel. Two active variables interact through the dot product of one's key with
the other's query, and because that product is not symmetric, the pair
contributes two numbers rather than one: ``<K_i, Q_j>`` and ``<K_j, Q_i>``.

Those numbers are the only thing stages 2-4 ever learn about how variables
combine. Everything downstream sees the interactions, never the variables.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor, nn

from ._random import seeded
from .config import RCCConfig

__all__ = ["Interaction", "InteractionEncoder", "ordered_pairs"]


class Interaction(NamedTuple):
    """What stage 1 produces.

    Unpacks as ``strength, keys, queries``.

    Attributes
    ----------
    strength:
        ``(n_episodes, n_observations, n_interactions)``. The interactions
        themselves — the input every later stage reads.
    keys, queries:
        ``(n_episodes, n_contexts, n_observations, embedding_dim)``. The
        embeddings the strengths were contracted from, after projection. Only
        the generator's norm penalty needs these; they are returned rather than
        recomputed because recomputing them means a second projection.
    """

    strength: Tensor
    keys: Tensor
    queries: Tensor


def ordered_pairs(n_contexts: int) -> list[tuple[int, int]]:
    """Which (key, query) context pairs become interactions, in order.

    Both directions of every distinct pair, the forward direction first. A
    single active variable has no pair, so it interacts with itself — otherwise
    stage 2 would receive an empty input.

    Examples
    --------
    >>> ordered_pairs(2)
    [(0, 1), (1, 0)]
    >>> ordered_pairs(3)
    [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)]
    >>> ordered_pairs(1)
    [(0, 0)]
    """
    pairs = [
        pair
        for i in range(n_contexts)
        for j in range(i + 1, n_contexts)
        for pair in ((i, j), (j, i))
    ]
    return pairs or [(0, 0)]


class InteractionEncoder(nn.Module):
    """Contract the active variables' embeddings into pairwise interactions.

    Parameters
    ----------
    cfg:
        The chain's configuration.
    keys, queries:
        Embeddings to use instead of learned ones, each
        ``(n_vars, n_observations, embedding_dim)``. Required when
        ``cfg.learn_embeddings`` is ``False``, rejected when it is ``True``.
        They are stored as buffers, so they move with ``.to(device)`` and are
        saved in the ``state_dict``, but no optimizer will ever see them.

    Notes
    -----
    Learned embeddings live in ``hidden_dim`` and are projected down to
    ``embedding_dim``; supplied ones are used as they arrive. The projection is
    what the generator's norm penalty acts on, which is why the projected
    embeddings — not the raw parameters — come back in :class:`Interaction`.

    Examples
    --------
    >>> from rcc import RCCConfig
    >>> cfg = RCCConfig(n_vars=20, n_contexts=2, n_observations=3, seed=0)
    >>> encoder = InteractionEncoder(cfg)
    >>> var_ids = torch.tensor([[3, 7], [7, 3]])
    >>> encoder(var_ids).strength.shape
    torch.Size([2, 3, 2])

    Swapping the two active variables swaps the two interactions, because the
    pair is ordered:

    >>> z = encoder(var_ids).strength
    >>> bool(torch.allclose(z[0, :, 0], z[1, :, 1], atol=1e-6))
    True
    """

    # Declared so that a type checker knows these are tensors. Without it,
    # anything registered as a buffer is inferred as ``Tensor | Module``, and
    # ``keys`` is a Parameter or a buffer depending on the config.
    keys: Tensor
    queries: Tensor
    key_index: Tensor
    query_index: Tensor

    def __init__(
        self,
        cfg: RCCConfig,
        *,
        keys: Tensor | None = None,
        queries: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        supplied = keys is not None or queries is not None
        if cfg.learn_embeddings and supplied:
            raise ValueError(
                "keys/queries were supplied but cfg.learn_embeddings is True. "
                "Set learn_embeddings=False to hold them fixed, or drop the "
                "arguments to learn them."
            )
        if not cfg.learn_embeddings and not (keys is not None and queries is not None):
            raise ValueError(
                "cfg.learn_embeddings is False, so both keys and queries must be "
                "supplied — there is nothing to learn them from otherwise."
            )

        expected = (cfg.n_vars, cfg.n_observations, cfg.embedding_dim)
        if cfg.learn_embeddings:
            with seeded(cfg.seed):
                raw = (cfg.n_vars, cfg.n_observations, cfg.hidden_dim)
                self.keys = nn.Parameter(_unit_rows(torch.randn(raw)))
                self.queries = nn.Parameter(_unit_rows(torch.randn(raw)))
                self.key_proj = nn.Linear(cfg.hidden_dim, cfg.embedding_dim)
                self.query_proj = nn.Linear(cfg.hidden_dim, cfg.embedding_dim)
        else:
            for name, tensor in (("keys", keys), ("queries", queries)):
                if tuple(tensor.shape) != expected:  # type: ignore[union-attr]
                    raise ValueError(
                        f"{name} must have shape {expected}, got "
                        f"{tuple(tensor.shape)}"  # type: ignore[union-attr]
                    )
            self.register_buffer("keys", keys.detach().clone())  # type: ignore[union-attr]
            self.register_buffer("queries", queries.detach().clone())  # type: ignore[union-attr]

        pairs = ordered_pairs(cfg.n_contexts)
        self.register_buffer("key_index", torch.tensor([k for k, _ in pairs]))
        self.register_buffer("query_index", torch.tensor([q for _, q in pairs]))

    def forward(self, var_ids: Tensor) -> Interaction:
        """Encode one batch of active variables.

        Parameters
        ----------
        var_ids:
            ``(n_episodes, n_contexts)`` of integer indices into the variable
            pool. Repeats are allowed: a variable may be active twice.
        """
        if var_ids.ndim != 2 or var_ids.shape[1] != self.cfg.n_contexts:
            raise ValueError(
                f"var_ids must have shape (n_episodes, {self.cfg.n_contexts}), "
                f"got {tuple(var_ids.shape)}"
            )

        keys = self.keys[var_ids]
        queries = self.queries[var_ids]
        if self.cfg.learn_embeddings:
            keys = self.key_proj(keys)
            queries = self.query_proj(queries)

        # (episodes, pairs, channels, dim) -> contract the embedding axis.
        paired_keys = keys[:, self.key_index]
        paired_queries = queries[:, self.query_index]
        strength = (paired_keys * paired_queries).sum(-1)
        return Interaction(strength.transpose(1, 2), keys, queries)

    def extra_repr(self) -> str:
        source = "learned" if self.cfg.learn_embeddings else "fixed"
        return (
            f"n_vars={self.cfg.n_vars}, n_observations={self.cfg.n_observations}, "
            f"embedding_dim={self.cfg.embedding_dim}, {source}"
        )


def _unit_rows(x: Tensor) -> Tensor:
    """Scale every embedding to unit norm."""
    return x / x.norm(dim=-1, keepdim=True)
