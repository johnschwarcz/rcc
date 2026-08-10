"""Regenerate the plots the README embeds.

    python docs/make_assets.py

``architecture.png`` is not produced here — it is a hand-drawn schematic. This
script regenerates the three figures that come out of a run: the training curve,
the belief accumulation panel, and the controller's policy.

Seeds are pinned here deliberately. Everywhere else in this repository the default
is ``seed=None`` so each run explores fresh randomness; these are the one
exception, because a README figure should not change every time it is rebuilt.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import torch

# Resolve the repository root without assuming __file__ exists: VS Code's "run in
# interactive window" pastes this source into a cell rather than executing the
# file, and a pasted cell has no __file__ at all.
if globals().get("__file__") is not None:
    ROOT = Path(__file__).resolve().parent.parent
else:
    _CWD = Path.cwd().resolve()
    _ROOTS = (p for p in (_CWD, *_CWD.parents) if (p / "pyproject.toml").exists())
    ROOT = next(_ROOTS, _CWD)

# The examples already know how to build a world and convert a batch; this script
# draws the same figures they do, so it borrows their plumbing rather than
# restating it.
sys.path.insert(0, str(ROOT / "examples"))

from _common import (  # noqa: E402
    accuracy,
    arguments,
    describe,
    draw,
    embeddings,
    world_and_config,
)
from rcc import (  # noqa: E402
    RCC,
    controller_loss,
    intrinsic_value,
    select_goal,
    supervised_loss,
)
from rcc.viz import plot_belief_accumulation, plot_policy, plot_training  # noqa: E402

if __name__ == "__main__":
    # The same flags as the examples, so a figure can be rebuilt at a different
    # shape: `python docs/make_assets.py --n-realizations 8`. Pin one here to
    # change the default — arguments(..., n_realizations=8) — and the command line
    # still wins. Only seed 0 and the --out fallback differ from the examples.
    args = arguments(iterations=3000, episodes=528, seed=0, n_steps = 10)
    out = args.out or ROOT / "docs" / "images"
    out.mkdir(parents=True, exist_ok=True)

    world, cfg = world_and_config(args, learn_embeddings=False)
    print(describe(args, cfg))
    chain = RCC(cfg, **embeddings(world))
    optimizer = torch.optim.Adam(chain.inference_parameters(), lr=1e-3)

    history: dict[str, list[float]] = {"chain accuracy": [], "ideal": [], "chance": []}
    for iteration in range(args.iterations):
        batch = draw(world, args.episodes)
        belief, _ = chain(batch.observations, batch.ctx_inds)
        loss = supervised_loss(belief, batch.posterior)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            goal = select_goal(belief, batch.goal_ind)
        history["chain accuracy"].append(accuracy(goal, batch.goal_value))
        history["ideal"].append(batch.ideal)
        history["chance"].append(1 / cfg.n_realizations)
        if iteration % 200 == 0:
            print(f"  iter {iteration:>4}  accuracy {history['chain accuracy'][-1]:.3f}")

    # Stage 4 gets its own short run: an untrained policy is near-uniform, and a
    # figure of one would say nothing about the controller.
    rng = torch.Generator().manual_seed(0)
    preferences = (torch.rand(cfg.n_observations, generator=rng) > 0.5).float()

    def value_landscape(rates):
        """What every joint realization is worth, given the preferences.

        ``rates`` is coggrid's likelihood table, one Bernoulli rate per channel per
        hypothetical joint realization.
        """
        n_episodes = rates.shape[0]
        per_action = rates.reshape(n_episodes, cfg.n_observations, -1).transpose(1, 2)
        value = intrinsic_value(per_action.reshape(-1, cfg.n_observations), preferences)
        return value.reshape(n_episodes, *cfg.realization_shape)

    control = torch.optim.Adam(chain.control_parameters(), lr=3e-3)
    for iteration in range(400):
        acting = draw(world, args.episodes)
        policy, predicted = chain.controller(chain.encoder(acting.ctx_inds).score)
        actions, log_prob, entropy = chain.controller.act(policy, rng)
        taken = value_landscape(acting.rates)[
            torch.arange(args.episodes), actions[:, 0], actions[:, 1]
        ]
        loss = controller_loss(taken, predicted, log_prob, entropy)
        control.zero_grad()
        loss.backward()
        control.step()
        if iteration % 50 == 0:
            print(f"  control iter {iteration:>4}  value {taken.mean().item():.3f}")

    # A fresh batch to draw, so the figure shows the trained policy on episodes it
    # was not just updated on.
    acting = draw(world, args.episodes)
    with torch.no_grad():
        policy, _ = chain.controller(chain.encoder(acting.ctx_inds).score)

    for name, fig in (
        ("training", plot_training(history, smooth=25)),
        (
            "belief_accumulation",
            plot_belief_accumulation(
                belief, batch.goal_ind, batch.goal_value, batch.posterior
            ),
        ),
        ("policy", plot_policy(policy, value_landscape(acting.rates))),
    ):
        path = out / f"{name}.png"
        fig.savefig(path, dpi=110, bbox_inches="tight")
        # These are pyplot-managed, so they stay alive until closed.
        plt.close(fig)
        print("wrote", path, f"({path.stat().st_size / 1e3:.0f} kB)")
