# Every tunable of a Representation Classification Chain. See __init__ for more details.

from dataclasses import dataclass, replace
from typing import Any

__all__ = ["RCCConfig"]

@dataclass(frozen=True, slots=True)
class RCCConfig:
    """Every tunable of a chain and its training.
    ``chain = RCC(cfg)`` creates and places the architecture on ``device``,
    ``Trainer(chain)`` reads the rest off ``chain.cfg``. 

    Attributes
    ----------
    n_vars:
        Size of the latent variable pool; one key and one query per channel each.
    n_contexts:
        How many latent variables are active at once.
    n_realizations:
        Discrete values each active variable can take.
    n_observations:
        Binary observation channels.
    embedding_dim:
        Width of the key/query embeddings contracted into interactions.
    hidden_dim:
        Width of every hidden layer.
    learn_embeddings:
        Whether stage 1's embeddings are learned. ``False`` means you supply them.
    seed:
        Seed for parameter initialization and a Trainer's sampling, or ``None`` 
    device:
        ``None`` -> cpu, ``"auto"`` -> cuda when available, or make explicit -> ``"cuda:0"``
    classifier_lr:
        Adam learning rate for stage 2.
    generator_lr:
        Adam learning rate for stages 1 and 3. ``None`` -> ``classifier_lr``;
        see :attr:`estimation_lr`.
    control_lr:
        Adam learning rate for the stage 4.
    classifier_entropy_bonus:
        ``reward_loss``'s entropy bonus.
    controller_entropy_bonus:
        ``controller_loss``'s entropy bonus.
    micro_batch:
        Split each batch into slices, accumulating gradients before an optimization step,
        ``None`` runs the batch in one pass. — see :meth:`rcc.Trainer.step`.
    """

    # ------------------------------------------------------------ architecture
    n_vars: int = 500
    n_contexts: int = 2
    n_realizations: int = 10
    n_observations: int = 5
    embedding_dim: int = 30
    hidden_dim: int = 1000
    learn_embeddings: bool = True
    seed: int | None = None

    # ----------------------------------------------------- where and how it runs
    device: str | None = None
    classifier_lr: float = 1e-3
    generator_lr: float | None = None
    control_lr: float = 3e-3
    classifier_entropy_bonus: float = 0.1
    controller_entropy_bonus: float = 0.05
    micro_batch: int | None = None

    # ------------------------------------------------------------------ setup
    def __post_init__(self) -> None:
        for name in ("n_vars", "n_contexts", "n_realizations", "n_observations",
                     "embedding_dim", "hidden_dim"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive int, got {value!r}")

        for name in ("classifier_lr", "control_lr"):
            self._require_positive_float(name, getattr(self, name))
        if self.generator_lr is not None:
            self._require_positive_float("generator_lr", self.generator_lr)

        for name in ("classifier_entropy_bonus", "controller_entropy_bonus"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"{name} must be a non-negative float, got {value!r}")

        if self.micro_batch is not None and (
            not isinstance(self.micro_batch, int) or isinstance(self.micro_batch, bool)
            or self.micro_batch < 1
        ):
            raise ValueError(
                f"micro_batch must be a positive int or None, got {self.micro_batch!r}")

        if self.device is not None and not isinstance(self.device, str):
            raise ValueError(
                f"device must be a string like 'cuda:0', 'auto' or None, "
                f"got {self.device!r}")

    @staticmethod
    def _require_positive_float(name: str, value: Any) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"{name} must be a positive float, got {value!r}")

    # ------------------------------------------------------- derived quantities
    @property
    def n_interactions(self) -> int:
        """Ordered pairs of distinct active variables.
        With a single active variable falls back to self-interaction ``<K, Q>``.
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
        """Width of the classifier's readin: one observation vector plus every interaction."""
        return self.n_observations * (1 + self.n_interactions)

    @property
    def estimation_lr(self) -> float:
        return self.classifier_lr if self.generator_lr is None else self.generator_lr

    # ---------------------------------------------------------------- helpers
    def replace(self, **changes: Any) -> "RCCConfig":
        """Return a copy with changes applied; validation re-runs."""
        return replace(self, **changes)
