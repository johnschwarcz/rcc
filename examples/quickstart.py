"""The whole project, as short as it goes.

    python examples/quickstart.py

Every stage trained, nothing printed and nothing plotted, so this reads as the API
rather than as an experiment: a world, a chain, a trainer, and the two loops. See
``docs/make_assets.py`` for the same run instrumented — losses, figures, and how
the chain does on variables it never trained on.

Every task and architecture parameter is a flag; ``--help`` lists them.
"""

import torch
from _common import (
    arguments,
    draw,
    embeddings,
    random_preferences,
    value_landscape,
    world_and_config,
)
from rcc import RCC, Trainer

args = arguments(learn_embeddings=True, lr=3e-3)
world, cfg = world_and_config(args)
# Supplying the world's embeddings and learning them are mutually exclusive.
chain = RCC(cfg, **({} if cfg.learn_embeddings else embeddings(world)))
trainer = Trainer(chain, lr=args.lr, control_lr=args.control_lr,
    generator=None if args.seed is None else torch.Generator().manual_seed(args.seed))

# Stages 1 to 3: a belief distilled from the exact posterior, and a representation
# learned from the generator's prediction error.
for _ in range(args.iterations):
    batch = draw(world, args.batch_size)
    trainer.step(batch.observations, batch.ctx_inds, batch.posterior,
        batch.goal_ind, batch.goal_value)

# Stage 4: act on the interactions alone, chasing the preferred observations.
preferences = random_preferences(cfg)
for _ in range(args.control_iterations):
    acting = draw(world, args.batch_size)
    trainer.control_step(acting.ctx_inds, value_landscape(acting.rates, preferences))
