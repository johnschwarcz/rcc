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
>>> var_ids = torch.randint(0, cfg.n_vars, (16, cfg.n_contexts))
>>> belief, interaction = chain(observations, var_ids)
>>> belief.shape
torch.Size([16, 10, 2, 4])

Nothing above is specific to a task. Supply your own ``observations`` and
``var_ids`` and the chain applies unchanged.

Layout
------
``config``        :class:`RCCConfig` — every tunable, validated, immutable.
``interactions``  Stage 1. Variables to pairwise interactions.
``classifier``    Stage 2. Observations to an accumulated belief.
``generator``     Stage 3. A belief back to observation rates.
``controller``    Stage 4. Interactions to an action.
``losses``        One objective per stage, and the divergences behind them.
``chain``         :class:`RCC`, which holds all four.
``toy``           A small demonstration task, so the package runs on its own.
``viz``           Plotting. Every function returns a figure; none call ``show()``.

Names follow `coggrid <https://github.com/johnschwarcz/coggrid>`_, the
environment the paper used, but nothing here imports it.
"""

from __future__ import annotations

from .chain import RCC, ChainOutput
from .classifier import BeliefClassifier, goal_accuracy, sample_goal, select_goal
from .config import RCCConfig
from .controller import Control, Controller, intrinsic_value
from .generator import ObservationGenerator, query_from_belief
from .interactions import Interaction, InteractionEncoder, ordered_pairs
from .losses import (
    controller_loss,
    distillation_loss,
    embedding_norm_penalty,
    kl_divergence,
    prediction_loss,
    reward_loss,
    soft_clip,
    symmetric_kl,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # configuration
    "RCCConfig",
    # the chain
    "RCC",
    "ChainOutput",
    # stage 1 — representation
    "InteractionEncoder",
    "Interaction",
    "ordered_pairs",
    # stage 2 — classification
    "BeliefClassifier",
    "select_goal",
    "sample_goal",
    "goal_accuracy",
    # stage 3 — generation
    "ObservationGenerator",
    "query_from_belief",
    # stage 4 — control
    "Controller",
    "Control",
    "intrinsic_value",
    # objectives
    "distillation_loss",
    "reward_loss",
    "prediction_loss",
    "embedding_norm_penalty",
    "controller_loss",
    "symmetric_kl",
    "kl_divergence",
    "soft_clip",
]
