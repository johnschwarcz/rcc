"""Every tunable of a Representation Classification Chain, in one dataclass.

Nothing in this module allocates a tensor, touches a device, or builds a layer.

Names follow `coggrid <https://github.com/johnschwarcz/coggrid>`_ so that the two
can be read side by side. Two of coggrid's fields are deliberately absent:
``n_steps`` and ``n_episodes``. Every module here reads those from the shape of
the tensor it is handed, so storing them would only create a second source of
truth that could disagree with the data.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

__all__ = ["RCCConfig"]


@dataclass(frozen=True, slots=True)
class RCCConfig:
    """Immutable description of a chain.

    The chain has four stages, and the seam between the second and third is the
    architectural claim: *variable inference* is trained separately from
    *parameter estimation*.

    1. :class:`~rcc.interactions.InteractionEncoder` looks up a key and a query
       embedding for each active variable and contracts them into one scalar per
       ordered pair, per observation channel.
    2. :class:`~rcc.classifier.BeliefClassifier` reads those interactions
       alongside the observation stream and emits a belief over each active
       variable's realization.
    3. :class:`~rcc.generator.ObservationGenerator` runs the chain backwards:
       given a realization it predicts the observation rates. Its error is what
       trains the embeddings in stage 1.
    4. :class:`~rcc.controller.Controller` chooses which joint realization to
       put the world into, scoring candidates by the interactions alone.

    The classifier never propagates gradient into the embeddings — it consumes
    them detached. So the embeddings are shaped only by prediction error, and
    the classifier only by classification error, even when both train at once.

    Attributes
    ----------
    n_vars:
        Size of the latent variable pool. The encoder holds one key and one
        query embedding per variable per observation channel.
    n_contexts:
        Number of simultaneously active latent variables. Drives the width of
        the controller's action space, which is
        ``n_realizations ** n_contexts``.
    n_realizations:
        Number of discrete values each active variable can take.
    n_observations:
        Number of binary observation channels.
    embedding_dim:
        Dimensionality of the key/query embeddings that are contracted into
        interactions.
    hidden_dim:
        Width of every hidden layer, and of the recurrent state.
    recurrent:
        Whether the classifier integrates observations one step at a time. When
        ``False`` it sees only the observation mean and emits a belief that is
        constant over time — the ablation that removes sequential integration
        while holding the rest of the chain fixed.
    reservoir:
        Whether to freeze the recurrent weights at initialization. The readin,
        the readout and the initial states still train, so the recurrence
        supplies a fixed dynamical basis rather than a learned one. Requires
        ``recurrent=True``.
    learn_embeddings:
        Whether stage 1's embeddings are learned. When ``False`` you must supply
        them (see :class:`~rcc.interactions.InteractionEncoder`), which isolates
        the classifier by handing it a perfect representation.
    seed:
        Seed for parameter initialization. ``None`` means non-reproducible.

    Examples
    --------
    >>> cfg = RCCConfig(n_contexts=3, n_realizations=4)
    >>> cfg.n_interactions
    6
    >>> cfg.realization_shape
    (4, 4, 4)
    >>> cfg.n_joint_realizations
    64
    """

    n_vars: int = 500
    n_contexts: int = 2
    n_realizations: int = 10
    n_observations: int = 5
    embedding_dim: int = 30
    hidden_dim: int = 1000
    recurrent: bool = True
    reservoir: bool = False
    learn_embeddings: bool = True
    seed: int | None = None

    # ------------------------------------------------------------------ setup
    def __post_init__(self) -> None:
        positive = {
            "n_vars": self.n_vars,
            "n_contexts": self.n_contexts,
            "n_realizations": self.n_realizations,
            "n_observations": self.n_observations,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
        }
        for name, value in positive.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive int, got {value!r}")

        if self.reservoir and not self.recurrent:
            raise ValueError(
                "reservoir=True requires recurrent=True: there are no recurrent "
                "weights to freeze in the feedforward classifier."
            )

    # ------------------------------------------------------- derived quantities
    @property
    def n_interactions(self) -> int:
        """Ordered pairs of distinct active variables, and never fewer than one.

        With a single active variable there is no pair, but the classifier still
        needs an input, so the encoder falls back to the self-interaction
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
    def replace(self, **changes: Any) -> RCCConfig:
        """Return a copy with ``changes`` applied (validation re-runs).

        Examples
        --------
        >>> RCCConfig().replace(n_contexts=1).n_interactions
        1
        """
        return replace(self, **changes)
