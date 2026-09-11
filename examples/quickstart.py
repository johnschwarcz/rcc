# A world, a chain, a trainer, and the training loops. 
# Every parameter is a flag; ``--help`` lists them.

import sys
from pathlib import Path
import rcc

# Run as a script, examples/ is already on sys.path; run from a console it is not.
sys.path.append(str(Path(rcc.__file__).parents[2] / "examples"))
from _common import (  # noqa: E402
    HELD_OUT,
    TRAIN,
    arguments,
    draw,
    evaluate,
    progress,
    random_preferences,
    value_landscape,
    world_and_config,
)
from rcc import RCC, Trainer, save_run  # noqa: E402

args = arguments(
    # world and RCC:
    n_vars= 1000, # 500,
    n_contexts=2,
    n_realizations=10,
    n_observations=5,
    embedding_dim=30,
    n_steps=30,
    # world only:
    n_held_out_vars=100, # 50,
    likelihood_temp=2.0,
    likelihood_freq=1.0,
    # RCC only:
    hidden_dim=1000,
    learn_embeddings=True,  # False hands RCC the world's embeddings
    # training, read off cfg by RCC and Trainer:
    seed=None,
    classifier_lr= 5e-4,  # 1e-4,
    generator_lr= 5e-4,  # 1e-4,
    classifier_entropy_bonus=0.1,
    control_lr=5e-3,
    controller_entropy_bonus=0.05,
    device='cuda:0',  # 'auto', or None for cpu
    micro_batch=5000,  # or None for one pass
    # the loops:
    iterations= 100000, # 50000,
    batch_size=20000,
    control_iterations=2000,
    evaluate_every=250,
    eval_episodes=5000,
)

world, cfg, given = world_and_config(args)
chain = RCC(cfg, **given)
trainer = Trainer(chain)

evaluation = {
    "train": draw(world, args.eval_episodes, split=TRAIN, rng=cfg.seed, device=chain.device),
    "test": draw(world, args.eval_episodes, split=HELD_OUT, rng=cfg.seed, device=chain.device),}

# Stages 1 to 3
loop = progress(args.iterations, "stages 1-3", args.progress)
for iteration in loop:
    batch = draw(world, args.batch_size, device=chain.device)
    step = trainer.step(batch.observations, batch.ctx_inds, batch.goal_ind, batch.goal_value)
    if iteration % args.evaluate_every == 0:
        test = evaluate(chain, evaluation["test"])
        loop.set_postfix({"train": f"{step.accuracy:.3f}",
            "test": f"{test['chain']:.3f}", "ideal": f"{test['ideal']:.3f}"})

# Stage 4
preferences = random_preferences(cfg, cfg.seed, device=chain.device)
loop = progress(args.control_iterations, "stage 4  ", args.progress)
for _ in loop:
    acting = draw(world, args.batch_size, device=chain.device)
    control = trainer.control_step(acting.ctx_inds, value_landscape(acting.rates, preferences))
    loop.set_postfix({"value": f"{control.value:.3f}"})

scores = {split: evaluate(chain, episodes) for split, episodes in evaluation.items()}
for split, score in scores.items():
    print(f"{split:>6}  chain {score['chain']:.3f}  ideal {score['ideal']:.3f}  "
          f"naive {score['naive']:.3f}")
print(f"controller value {control.value:.3f}")

checkpoint = args.checkpoint or Path(__file__).resolve().parents[1] / "runs" / "quickstart.pt"
save_run(checkpoint, chain, trainer=trainer, scores=scores)
print(f"saved {checkpoint}")
