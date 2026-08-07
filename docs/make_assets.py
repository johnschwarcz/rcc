"""Regenerate the images the README embeds.

    python docs/make_assets.py

Seeds are pinned here deliberately. Everywhere else in this repository the
default is ``seed=None`` so each run explores fresh randomness; these are the one
exception, because a figure in the README should not change every time someone
regenerates it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from rcc import (
    RCC,
    RCCConfig,
    controller_loss,
    distillation_loss,
    intrinsic_value,
    select_goal,
)
from rcc.toy import ToyTask
from rcc.viz import (
    plot_architecture,
    plot_belief_accumulation,
    plot_policy,
    plot_training,
)

# Resolve the repository root without assuming __file__ exists: VS Code's
# "run in interactive window" pastes this source into a cell rather than
# executing the file, and a pasted cell has no __file__ at all.
if globals().get("__file__") is not None:
    ROOT = Path(__file__).resolve().parent.parent
else:
    _CWD = Path.cwd().resolve()
    _CANDIDATES = (p for p in (_CWD, *_CWD.parents) if (p / "pyproject.toml").exists())
    ROOT = next(_CANDIDATES, _CWD)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "images")
    parser.add_argument("--steps", type=int, default=1500)
    args, _ignored = parser.parse_known_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cfg = RCCConfig(
        n_vars=4,
        n_contexts=2,
        n_realizations=4,
        n_observations=5,
        embedding_dim=6,
        hidden_dim=64,
        learn_embeddings=False,
        seed=0,
    )
    task = ToyTask(cfg, seed=0)
    chain = RCC(cfg, keys=task.keys, queries=task.queries)
    optimizer = torch.optim.Adam(chain.inference_parameters(), lr=3e-3)
    rng = torch.Generator().manual_seed(0)

    history: dict[str, list[float]] = {"chain accuracy": [], "ideal": [], "chance": []}
    for step in range(args.steps):
        episode = task.sample(128, n_steps=20, generator=rng)
        belief, interaction = chain(episode.observations, episode.var_ids)
        loss = distillation_loss(belief, episode.posterior)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            chain_goal = select_goal(belief, episode.goal_index)
            ideal_goal = select_goal(episode.posterior, episode.goal_index)
            history["chain accuracy"].append(
                (chain_goal[:, -1].argmax(-1) == episode.goal_value)
                .float().mean().item()
            )
            history["ideal"].append(
                (ideal_goal[:, -1].argmax(-1) == episode.goal_value)
                .float().mean().item()
            )
            history["chance"].append(1 / cfg.n_realizations)
        if step % 250 == 0:
            print(f"  step {step:>4}  accuracy {history['chain accuracy'][-1]:.3f}")

    # Stage 4 gets its own short run: an untrained policy is near-uniform, and a
    # figure of one would say nothing about the controller.
    preferences = (torch.rand(cfg.n_observations, generator=rng) > 0.5).float()

    def value_landscape(var_ids):
        """What every joint realization is worth, given the preferences."""
        rates = task.rate_table(var_ids)
        n_episodes = rates.shape[0]
        flat = rates.reshape(n_episodes, cfg.n_observations, -1).transpose(1, 2)
        value = intrinsic_value(flat.reshape(-1, cfg.n_observations), preferences)
        return value.reshape(n_episodes, *cfg.realization_shape)

    control = torch.optim.Adam(chain.control_parameters(), lr=3e-3)
    for step in range(400):
        acting = task.sample(128, n_steps=1, generator=rng)
        policy, predicted = chain.controller(chain.encoder(acting.var_ids).strength)
        actions, log_prob, entropy = chain.controller.act(policy, rng)

        taken = value_landscape(acting.var_ids)[
            torch.arange(128), actions[:, 0], actions[:, 1]
        ]
        loss = controller_loss(taken, predicted, log_prob, entropy)
        control.zero_grad()
        loss.backward()
        control.step()
        if step % 200 == 0:
            print(f"  control step {step:>4}  value {taken.mean().item():.3f}")

    # A fresh batch to draw, so the figure shows the trained policy on episodes
    # it was not just updated on.
    acting = task.sample(128, n_steps=1, generator=rng)
    with torch.no_grad():
        policy, _ = chain.controller(chain.encoder(acting.var_ids).strength)
    landscape = value_landscape(acting.var_ids)

    for name, fig in (
        ("architecture", plot_architecture()),
        ("training", plot_training(history, smooth=25)),
        (
            "belief_accumulation",
            plot_belief_accumulation(
                belief, episode.goal_index, episode.goal_value, episode.posterior
            ),
        ),
        ("policy", plot_policy(policy, landscape)),
    ):
        path = args.out / f"{name}.png"
        fig.savefig(path, dpi=110, bbox_inches="tight")
        # These are pyplot-managed, so they stay alive until closed.
        plt.close(fig)
        print("wrote", path, f"({path.stat().st_size / 1e3:.0f} kB)")
