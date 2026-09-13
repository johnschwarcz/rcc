"""The whole project in one run, and the plots the README embeds.
The training curve, both objectives' losses, one episode's belief accumulation,
that comparison averaged over the batch, how the chain does on variables it never
trained on, and the controller's policy.

This is ``examples/quickstart.py`` with the instrumentation added and nothing else:
the same config, the same two loops, the same ``RCC(cfg)`` and ``Trainer(chain)``.
What it adds is a record — accuracy and both losses at every iteration, a held-out
measurement periodically, and the last batch's belief — and the figures drawn from
it.

That record is saved beside the chain, so ``--replot`` redraws every figure from
the finished run without training anything. Adjusting a colour or a smoothing
window costs a second rather than the length of the run.
"""
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import rcc


def _repo_root() -> Path:
    """ The checkout holding examples/_common.py. """
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
    HELD_OUT,
    TRAIN,
    arguments,
    describe,
    draw,
    evaluate,
    progress,
    random_preferences,
    value_landscape,
    world_and_config,
)
from rcc import RCC, Trainer, load_run, save_run  # noqa: E402
from rcc.viz import (  # noqa: E402
    CHAIN,
    IDEAL,
    plot_belief_accumulation,
    plot_belief_average,
    plot_generalization,
    plot_losses,
    plot_policy,
    plot_training,
)

if __name__ == "__main__":
    args = arguments(
        # world and RCC:
        n_vars=500,
        n_contexts=2,
        n_realizations=10,
        n_observations=5,
        embedding_dim=30,
        n_steps=30,
        # world only:
        n_held_out_vars=50,
        likelihood_temp=2.0,
        likelihood_freq=1.0,
        # RCC only:
        hidden_dim=1000,
        learn_embeddings=True,  # False hands RCC the world's embeddings
        # training, read off cfg by RCC and Trainer:
        seed=0,  # or None
        classifier_lr=1e-4,
        generator_lr=1e-4,
        classifier_entropy_bonus=0.1,
        control_lr=5e-3,
        controller_entropy_bonus=0.05,
        device='cuda:0',  # 'auto', or None for cpu
        micro_batch=5000,  # or None for one pass
        # the loops and the measuring:
        iterations=50000,
        batch_size=20000,
        control_iterations=2000,
        evaluate_every=250,
        eval_episodes=5000,  # episodes per evaluation
        replot=False, 
        checkpoint = None,
    )
    
    out = args.out or ROOT / "docs" / "images"
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint or ROOT / "runs" / "make_assets.pt"

    if args.replot:
        if not checkpoint.exists():
            raise SystemExit(f"--replot needs a saved run, and {checkpoint} is not "
                             f"there. Run without --replot first.")
        run = load_run(checkpoint)
        chain, figures = run.chain, run.artifacts
        print(f"replotting {checkpoint}, trained {figures['iterations']} iterations")
    else:
        world, cfg, given = world_and_config(args)
        chain = RCC(cfg, **given)
        trainer = Trainer(chain)
        print(describe(args, cfg)  + f"\ndevice: asked for {cfg.device or 'cpu'}, running on {chain.device}")

        evaluation = {
            "train": draw(world, args.eval_episodes, split=TRAIN, rng=cfg.seed, device=chain.device),
            "test": draw(world, args.eval_episodes, split=HELD_OUT, rng=cfg.seed, device=chain.device), }
        perfs = {"train": [], "test": [], "ideal": []}
        losses = {"classifier": [], "generator": []}

        loop = progress(args.iterations, "stages 1-3", args.progress)
        for iteration in loop:
            batch = draw(world, args.batch_size, device=chain.device)
            step = trainer.step(batch.observations, batch.ctx_inds, batch.goal_ind, batch.goal_value)
            perfs["train"].append(step.accuracy)
            perfs["ideal"].append(batch.ideal)
            losses["classifier"].append(step.classifier_loss)
            losses["generator"].append(step.generator_loss)
            if iteration % args.evaluate_every == 0:
                test_acc = evaluate(chain, evaluation["test"])["chain"]
                perfs["test"].append(test_acc)
                loop.set_postfix({"train": f"{step.accuracy:.3f}", "test": f"{test_acc:.3f}",
                    "classifier": f"{step.classifier_loss:.4f}", "generator": f"{step.generator_loss:.4f}"})
                if loop.disable:
                    print(f"  iter {iteration:>6}  train {step.accuracy:.3f}  "
                          f"test {test_acc:.3f}  classifier {step.classifier_loss:.4f}"
                          f"  generator {step.generator_loss:.4f}", flush=True)
                    
        scores = {split: evaluate(chain, episodes)
                  for split, episodes in evaluation.items()}

        preferences = random_preferences(cfg, cfg.seed, device=chain.device)
        loop = progress(args.control_iterations, "stage 4  ", args.progress)
        for iteration in loop:
            acting = draw(world, args.batch_size, device=chain.device)
            control = trainer.control_step(acting.ctx_inds, value_landscape(acting.rates, preferences))
            loop.set_postfix({"value": f"{control.value:.3f}"})
            if loop.disable and iteration % 50 == 0:
                print(f"  control iter {iteration:>5}  value {control.value:.3f}", flush=True)

        acting = draw(world, args.batch_size, device=chain.device)
        landscape = value_landscape(acting.rates, preferences)
        with torch.no_grad():
            policy, _ = chain.controller(chain.encoder(acting.ctx_inds).score)

        figures = {
            "perfs": perfs,
            "losses": losses,
            "belief": step.belief,
            "goal_ind": batch.goal_ind,
            "goal_value": batch.goal_value,
            "posterior": batch.posterior,
            "scores": scores,
            "policy": policy,
            "landscape": landscape,
            "iterations": args.iterations,
        }
        save_run(checkpoint, chain, trainer=trainer, **figures)
        print(f"wrote {checkpoint} "
              f"({checkpoint.stat().st_size / 1e6:.0f} MB) — rerun with --replot "
              f"to redraw these figures without training")

    chance = 1 / chain.cfg.n_realizations
    for name, fig in (
        ("training", plot_training(figures["perfs"], smooth=25, styles={"train": (CHAIN, "-"), "test": (CHAIN, "--"), "ideal": (IDEAL, "--")})),
        ("losses", plot_losses(figures["losses"], smooth=25)),
        ("belief_accumulation", plot_belief_accumulation(figures["belief"], figures["goal_ind"], figures["goal_value"], figures["posterior"])),
        ("belief_average", plot_belief_average(figures["belief"], figures["goal_ind"], figures["goal_value"], figures["posterior"])),
        ("generalization", plot_generalization(figures["scores"], chance=chance)),
        ("policy", plot_policy(figures["policy"], figures["landscape"])),
    ):
        path = out / f"{name}.png"
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)  # These are pyplot-managed, so they stay alive until closed.
        print("wrote", path, f"({path.stat().st_size / 1e3:.0f} kB)")
