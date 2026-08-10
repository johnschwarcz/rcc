"""Every tunable of a Representation Classification Chain, in one dataclass.

Names follow `coggrid <https://github.com/johnschwarcz/coggrid>`
Two of coggrid's fields are deliberately absent: ``n_steps`` and ``n_episodes``.
For flexibility, modules read those from the shape of the tensor it is handed
"""

from dataclasses import dataclass, replace
from typing import Any

__all__ = ["RCCConfig"]

@dataclass(frozen=True, slots=True)
class RCCConfig:
    """Every tunable of a chain, validated on construction. See :mod:`rcc` for
    what the four stages these configure actually do.

    Attributes
    ----------
    n_vars:
        Size of the latent variable pool; one key and one query per channel each.
    n_contexts:
        How many latent variables are active at once. Sets the controller's action
        space, ``n_realizations ** n_contexts``.
    n_realizations:
        Discrete values each active variable can take.
    n_observations:
        Binary observation channels.
    embedding_dim:
        Width of the key/query embeddings contracted into interactions.
    hidden_dim:
        Width of every hidden layer, and of the recurrent state.
    learn_embeddings:
        Whether stage 1's embeddings are learned. ``False`` means you supply them,
        isolating the classifier by handing it a perfect representation.
    seed:
        Seed for parameter initialization. ``None`` means non-reproducible.
    """

    n_vars: int = 500
    n_contexts: int = 2
    n_realizations: int = 10
    n_observations: int = 5
    embedding_dim: int = 30
    hidden_dim: int = 1000
    learn_embeddings: bool = True
    seed: int | None = None

    # ------------------------------------------------------------------ setup
    def __post_init__(self) -> None:
        for name in ("n_vars", "n_contexts", "n_realizations", "n_observations",
                     "embedding_dim", "hidden_dim"):
            value = getattr(self, name)
            # bool is an int subclass, and RCCConfig(n_vars=True) is a mistake.
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive int, got {value!r}")

    # ------------------------------------------------------- derived quantities
    @property
    def n_interactions(self) -> int:
        """Ordered pairs of distinct active variables.
        With a single active variable the encoder falls back to self-interaction
        ``<K, Q>`` of that one variable.
        """
        return max(1, self.n_contexts * (self.n_contexts - 1))

    @property
    def realization_shape(self) -> tuple[int, ...]:
        """Shape of the joint realization axes: ``(n_realizations,) * n_contexts``."""
        return (self.n_realizations,) * self.n_contexts

    @property
    def n_joint_realizations(self) -> int:
        """Size of the controller's action space, ``n_realizations ** n_contexts``."""
        return self.n_realizations**self.n_contexts

    @property
    def classifier_input_dim(self) -> int:
        """Width of the classifier's readin: one observation vector plus the
        interactions of every channel."""
        return self.n_observations * (1 + self.n_interactions)

    # ---------------------------------------------------------------- helpers
    def replace(self, **changes: Any) -> "RCCConfig":
        """Return a copy with changes applied; validation re-runs.

        >>> RCCConfig().replace(n_contexts=1).n_interactions
        1
        """
        return replace(self, **changes)
