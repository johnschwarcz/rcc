"""The whole project in one run, and the plots the README embeds.
The training curve, both objectives' losses, one episode's belief accumulation,
that comparison averaged over the batch, how the chain does on variables it never
trained on, and the controller's policy. 
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
    embeddings,
    evaluate,
    random_preferences,
    resolve_device,
    value_landscape,
    world_and_config,
)
from rcc import RCC, Trainer  # noqa: E402
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
    seed = 0 # or None
    args = arguments(
        # reaches both the world and the RCC:
        n_vars=500,
        n_contexts=2,  
        n_realizations=10,
        n_observations=5,
        embedding_dim=30,  
        n_steps=30,
        # reaches the world alone:
        n_held_out_vars=50,
        likelihood_temp=2.0, 
        likelihood_freq=1.0, 
        # reaches the RCC alone:
        hidden_dim=1000,
        learn_embeddings=True,  # False hands RCC the world's true embeddings
        # run
        iterations=50000,
        batch_size=20000,
        seed=seed,
        classifier_lr=1e-4, 
        generator_lr=1e-4, 
        classifier_entropy_bonus=0.01,
        control_iterations=2000,  
        control_lr=5e-3, 
        controller_entropy_bonus=0.05,
        device=None,
        micro_batch= 1000, # or None
        evaluate_every=250,
        eval_episodes=2000, # batch size for evaluation
    )
    out = args.out or ROOT / "docs" / "images"
    out.mkdir(parents=True, exist_ok=True)

    world, cfg = world_and_config(args)
    device = resolve_device(args.device)
    evaluation = {"train": draw(world, args.eval_episodes, split=TRAIN, rng=args.seed, device=device), "test": draw(world, args.eval_episodes, split=HELD_OUT, rng=args.seed, device=device)}
    chain = RCC(cfg, **({} if cfg.learn_embeddings else embeddings(world))).to(device)    
    trainer = Trainer(chain, classifier_lr=args.classifier_lr, generator_lr=args.generator_lr, control_lr=args.control_lr, 
        classifier_entropy_bonus=args.classifier_entropy_bonus, controller_entropy_bonus=args.controller_entropy_bonus,
        micro_batch=args.micro_batch,
        generator=None if seed is None else torch.Generator(device=device).manual_seed(seed))
    print(describe(args, cfg) + "\n" + f"device: {chain.device}")


    log: dict[str, dict] = {"perfs": {"train": [], "test": [], "ideal": []}, "losses": {"classifier": [], "generator": []}}

    for iteration in range(args.iterations):
        batch = draw(world, args.batch_size, device=device)
        step = trainer.step(batch.observations, batch.ctx_inds, batch.goal_ind, batch.goal_value)
        log["perfs"]["train"].append(step.accuracy)
        log["perfs"]["ideal"].append(batch.ideal)
        log["losses"]["classifier"].append(step.classifier_loss)
        log["losses"]["generator"].append(step.generator_loss)
        if iteration % args.evaluate_every == 0:
            test_acc = evaluate(chain, evaluation["test"])["chain"]
            log["perfs"]["test"].append(test_acc)
            print(f"  iter {iteration:>4}  train {step.accuracy:.3f} test {test_acc:.3f}"
                f"  classifier {step.classifier_loss:.4f}  generator {step.generator_loss:.4f}")
    scores = {split: evaluate(chain, episodes) for split, episodes in evaluation.items()}
    
    preferences = random_preferences(cfg, device=device)
    for iteration in range(args.control_iterations):
        acting = draw(world, args.batch_size, device=device)
        control = trainer.control_step(acting.ctx_inds, value_landscape(acting.rates, preferences))
        if iteration % 50 == 0:
            print(f"  control iter {iteration:>4}  value {control.value:.3f}")
    acting = draw(world, args.batch_size, device=device)

    with torch.no_grad():
        policy, _ = chain.controller(chain.encoder(acting.ctx_inds).score)

    for name, fig in (
        ("training", plot_training(log["perfs"], smooth=25, styles={"train": (CHAIN, "-"), "test": (CHAIN, "--"), "ideal": (IDEAL, "--")})),
        ("losses", plot_losses(log["losses"], smooth=25)),
        ("belief_accumulation", plot_belief_accumulation(step.belief, batch.goal_ind, batch.goal_value, batch.posterior)),
        ("belief_average", plot_belief_average(step.belief, batch.goal_ind, batch.goal_value, batch.posterior)),
        ("generalization", plot_generalization(scores, chance=1 / cfg.n_realizations)),
        ("policy", plot_policy(policy, value_landscape(acting.rates, preferences))),
    ):
        path = out / f"{name}.png"
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)  # These are pyplot-managed, so they stay alive until closed.
        print("wrote", path, f"({path.stat().st_size / 1e3:.0f} kB)")
