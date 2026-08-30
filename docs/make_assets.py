"""The whole project in one run, and the plots the README embeds.
    python docs/make_assets.py
The training curve, both objectives' losses, one episode's belief accumulation,
that comparison averaged over the batch, how the chain does on variables it never
trained on, and the controller's policy. architecture.png is hand-drawn, not
produced here. seed=0 is pinned so a rebuild does not move the figures.
"""
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import rcc


def _repo_root() -> Path:
    """The checkout holding examples/_common.py.
    Searched for rather than derived: an interactive cell has no __file__ and an
    arbitrary cwd, and an editable install points back at the checkout.
    """
    seeds = [Path.cwd(), Path(rcc.__file__).parent]
    if globals().get("__file__") is not None:
        seeds.insert(0, Path(__file__).parent)
    for seed in seeds:
        for path in (seed.resolve(), *seed.resolve().parents):
            if (path / "examples" / "_common.py").exists():
                return path
    raise SystemExit("make_assets.py has to run from a checkout of the rcc repository.")

ROOT = _repo_root()
sys.path.insert(0, str(ROOT / "examples"))

from _common import (  # noqa: E402
    arguments,
    describe,
    draw,
    embeddings,
    evaluate,
    random_preferences,
    value_landscape,
    world_and_config,
)
from rcc import RCC, Trainer  # noqa: E402
from rcc.viz import (  # noqa: E402
    plot_belief_accumulation,
    plot_belief_average,
    plot_generalization,
    plot_losses,
    plot_policy,
    plot_training,
)

if __name__ == "__main__":
    # Every knob, spelled out so a figure can be reshaped from right here. The
    # command line still wins: `python docs/make_assets.py --hidden-dim 512`.
    args = arguments(
        # task - reaches both the world and the chain
        n_vars=500,
        n_contexts=2,  
        n_realizations=10,
        n_observations=5,
        embedding_dim=30,  # coggrid orthogonalizes per channel, so >= n_observations
        n_steps=30,
        # architecture - the chain alone
        hidden_dim=1000,
        learn_embeddings=True,  # False hands stage 1 the world's true embeddings
        # run
        iterations=50000,
        batch_size=5000,
        lr=1e-4,
        control_iterations=400,
        control_lr=3e-3,
        seed=0,
    )
    out = args.out or ROOT / "docs" / "images"
    out.mkdir(parents=True, exist_ok=True)

    world, cfg = world_and_config(args)
    print(describe(args, cfg))
    # Supplying the world's embeddings and learning them are mutually exclusive.
    chain = RCC(cfg, **({} if cfg.learn_embeddings else embeddings(world)))
    trainer = Trainer(chain, lr=args.lr, control_lr=args.control_lr,
        generator=torch.Generator().manual_seed(0))

    # Generalization. coggrid holds a third of the variable pool back, so held_out is
    # variables the chain was never trained on. Both batches are drawn once and held
    # fixed, so the held-out curve below tracks the same episodes all the way through.
    evaluation = {split: draw(world, args.batch_size, split=split, rng=args.seed)
        for split in ("train", "held_out")}
    # Each measurement is a forward pass over the whole batch, so not every iteration.
    evaluate_every = 250

    history: dict[str, list[float]] = {
        "chain accuracy": [], "held out": [], "ideal": [], "chance": []}
    losses: dict[str, list[float]] = {"classifier": [], "generator": []}
    for iteration in range(args.iterations):
        batch = draw(world, args.batch_size)
        step = trainer.step(batch.observations, batch.ctx_inds, batch.posterior,
            batch.goal_ind, batch.goal_value)
        history["chain accuracy"].append(step.accuracy)
        history["ideal"].append(batch.ideal)
        history["chance"].append(1 / cfg.n_realizations)
        losses["classifier"].append(step.classifier_loss)
        losses["generator"].append(step.generator_loss)
        if iteration % evaluate_every == 0:
            history["held out"].append(evaluate(chain, evaluation["held_out"])["chain"])
        if iteration % 200 == 0:
            print(f"  iter {iteration:>4}  acc {step.accuracy:.3f}"
                f"  held out {history['held out'][-1]:.3f}"
                f"  classifier {step.classifier_loss:.4f}"
                f"  generator {step.generator_loss:.4f}")

    scores = {split: evaluate(chain, episodes) for split, episodes in evaluation.items()}
    print(f"\n{'':10}" + "".join(f"{name:>8}" for name in scores["train"]))
    for split, row in scores.items():
        print(f"{split:10}" + "".join(f"{value:>8.3f}" for value in row.values()))
    print(f"{'chance':10}{1 / cfg.n_realizations:>8.3f}")

    # Stage 4 gets its own short run: an untrained policy is near-uniform, and a
    # figure of one would say nothing about the controller.
    preferences = random_preferences(cfg)
    for iteration in range(args.control_iterations):
        acting = draw(world, args.batch_size)
        control = trainer.control_step(
            acting.ctx_inds, value_landscape(acting.rates, preferences))
        if iteration % 50 == 0:
            print(f"  control iter {iteration:>4}  value {control.value:.3f}")

    # A fresh batch to draw, so the figure shows the trained policy on episodes it
    # was not just updated on.
    acting = draw(world, args.batch_size)
    with torch.no_grad():
        policy, _ = chain.controller(chain.encoder(acting.ctx_inds).score)

    for name, fig in (
        ("training", plot_training(history, smooth=25)),
        ("losses", plot_losses(losses, smooth=25)),
        ("belief_accumulation", plot_belief_accumulation(
            step.belief, batch.goal_ind, batch.goal_value, batch.posterior)),
        ("belief_average", plot_belief_average(
            step.belief, batch.goal_ind, batch.goal_value, batch.posterior)),
        ("generalization", plot_generalization(scores, chance=1 / cfg.n_realizations)),
        ("policy", plot_policy(policy, value_landscape(acting.rates, preferences))),
    ):
        path = out / f"{name}.png"
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)  # These are pyplot-managed, so they stay alive until closed.
        print("wrote", path, f"({path.stat().st_size / 1e3:.0f} kB)")
