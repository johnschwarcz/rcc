"""rcc — Representation Classification Chains.

A chain separates *what the world is made of* from *what is happening right
now*. Stage 1 turns latent variables into pairwise interactions; stage 2 reads
those interactions alongside an observation stream and accumulates a belief;
stage 3 runs the map backwards to predict observations, which is what trains
stage 1; stage 4 acts on the interactions alone.

The seam is that stage 2 consumes the representation *detached*. Classification
error can never reshape the representation to make classification easier, so a
teaching signal on a few goal variables leaves behind a representation that
transfers to variables it was never given labels for.

Quick start
-----------
>>> import torch
>>> from rcc import RCC, RCCConfig
>>> cfg = RCCConfig(n_vars=50, n_contexts=2, n_realizations=4,
...                 n_observations=3, hidden_dim=32, seed=0)
>>> chain = RCC(cfg)
>>> observations = torch.rand(16, 10, 3).round()
>>> ctx_inds = torch.randint(0, cfg.n_vars, (16, cfg.n_contexts))
>>> belief, interaction = chain(observations, ctx_inds)
>>> belief.shape
torch.Size([16, 10, 2, 4])

Nothing above is specific to a task. Supply your own ``observations`` and
``ctx_inds`` and the chain applies unchanged.

Layout
------
``config``        :class:`RCCConfig` — every tunable, validated, immutable.
``interactions``  Stage 1. Variables to pairwise interactions.
``classifier``    Stage 2. Observations to an accumulated belief.
``generator``     Stage 3. A belief back to observation rates.
``controller``    Stage 4. Interactions to an action.
``losses``        One objective per stage. The primitives they are built from
                  (``symmetric_kl``, ``kl_divergence``, ``soft_clip``) stay in
                  the module rather than at the top level; import them from
                  ``rcc.losses`` if you are writing an objective of your own.
``chain``         :class:`RCC`, which holds all four.
``training``      :class:`Trainer` — those objectives, each with its own
                  optimizer, and the order they are stepped in.
``checkpoint``    :func:`save_run` and :func:`load_run` — a finished run on disk,
                  the chain and its optimizers beside what the loop measured, so
                  re-plotting never means retraining.
``viz``           Plotting. Every function returns a figure; none call ``show()``.

Names follow `coggrid <https://github.com/johnschwarcz/coggrid>`_, the
environment the paper used, so the two can be read side by side — though nothing
here imports it. Config fields, ``ctx_inds``, ``ctx_vals``, ``goal_ind``,
``goal_value``, ``observations`` and ``rates`` are all coggrid's own names, kept
rather than reinvented. *Realization* stays the word for what a ``ctx_vals``
holds, exactly as coggrid pairs the field with ``n_realizations``.
"""

from ._helpers import resolve_device, sample_categorical
from .chain import RCC
from .checkpoint import Run, load_run, save_run
from .classifier import BeliefClassifier, goal_accuracy, select_goal
from .config import RCCConfig
from .controller import Controller, intrinsic_value
from .generator import ObservationGenerator, query_from_belief
from .interactions import Interaction, InteractionEncoder, ordered_pairs
from .losses import (
    controller_loss,
    embedding_norm_penalty,
    prediction_loss,
    reward_loss,
    supervised_loss,
)
from .training import ControlStep, Trainer, TrainingStep

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # configuration
    "RCCConfig",
    # the chain
    "RCC",
    # stage 1 — representation
    "InteractionEncoder",
    "Interaction",
    "ordered_pairs",
    # stage 2 — classification
    "BeliefClassifier",
    "select_goal",
    "sample_categorical",
    "goal_accuracy",
    # stage 3 — generation
    "ObservationGenerator",
    "query_from_belief",
    # stage 4 — control
    "Controller",
    "intrinsic_value",
    # objectives
    "supervised_loss",
    "reward_loss",
    "prediction_loss",
    "embedding_norm_penalty",
    "controller_loss",
    # training
    "Trainer",
    "TrainingStep",
    "ControlStep",
    # where a run happens, and how one is kept
    "resolve_device",
    "save_run",
    "load_run",
    "Run",
]
