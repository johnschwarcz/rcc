"""Train on familiar variables, test on variables the chain has never seen.

    pip install git+https://github.com/johnschwarcz/coggrid
    python examples/coggrid_transfer.py

This is the experiment the architecture was built for, and it needs a real
environment: `coggrid <https://github.com/johnschwarcz/coggrid>`_ holds out a
slice of its variable pool, so "novel combination of known variables" is
something you can actually measure rather than assert.

Nothing in ``src/rcc`` imports coggrid. This example does, and skips if it is
not installed.

The comparison worth watching is the third column. The joint observer knows the
interactions; the naive one assumes the active variables are independent. The
gap between them is the factorization regret, and it is the part of the task a
chain can only get right by having represented the interaction structure.

This is a demonstration of the setup, not a replication. A run of this length
beats the naive observer on familiar variables and does *not* transfer to
held-out ones; the paper's transfer result comes from a reinforcement-learning
run over far more episodes, with the embeddings learned rather than supplied.
Treat the held-out row as a baseline to improve on.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    from coggrid import CogGridConfig, World, run_observers
except ImportError:
    print(
        "This example needs coggrid:\n"
        "    pip install git+https://github.com/johnschwarcz/coggrid",
        file=sys.stderr,
    )
    raise SystemExit(0) from None

from rcc import RCC, RCCConfig, distillation_loss, select_goal

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("--out", type=Path, default=None, help="save figures here")
parser.add_argument("--steps", type=int, default=2500)
parser.add_argument("--episodes", type=int, default=128)
parser.add_argument(
    "--seed", type=int, default=None,
    help="pin the run; the default explores fresh randomness",
)
args, _ignored = parser.parse_known_args()

# One config, read by both packages. The field names are shared deliberately.
SHAPE = dict(
    n_vars=12,
    n_contexts=2,
    n_realizations=4,
    n_observations=5,
    embedding_dim=8,
)
# n_held_out_vars is explicit: coggrid's default of n_vars // 10 would leave
# a single held-out variable, and a 'novel pair' drawn from a pool of one is
# the same variable twice.
world = World(CogGridConfig(**SHAPE, n_steps=20, n_held_out_vars=4, seed=args.seed))
cfg = RCCConfig(**SHAPE, hidden_dim=64, learn_embeddings=False, seed=args.seed)

# coggrid's embeddings *are* the representation stage 1 would otherwise learn,
# so handing them over isolates the classifier.
chain = RCC(
    cfg,
    keys=torch.as_tensor(world.keys, dtype=torch.float32),
    queries=torch.as_tensor(world.queries, dtype=torch.float32),
)
optimizer = torch.optim.Adam(chain.inference_parameters(), lr=3e-3)


def as_tensors(batch, traces):
    """coggrid speaks numpy; the chain speaks torch."""
    return (
        torch.as_tensor(batch.observations, dtype=torch.float32),
        torch.as_tensor(batch.ctx_inds, dtype=torch.long),
        torch.as_tensor(batch.goal_ind, dtype=torch.long),
        torch.as_tensor(batch.goal_value, dtype=torch.long),
        torch.as_tensor(traces["joint"].belief, dtype=torch.float32),
    )


def evaluate(split, n_episodes=1024):
    batch = world.sample_episodes(n_episodes, split=split, rng=args.seed)
    traces = run_observers(batch)
    observations, var_ids, goal_index, goal_value, _ = as_tensors(batch, traces)
    with torch.no_grad():
        belief, _ = chain(observations, var_ids)
    chain_goal = select_goal(belief, goal_index)
    return {
        "chain": (chain_goal[:, -1].argmax(-1) == goal_value).float().mean().item(),
        "joint": float(traces["joint"].accuracy[:, -1].mean()),
        "naive": float(traces["naive"].accuracy[:, -1].mean()),
    }


history: dict[str, list[float]] = {"chain accuracy": [], "ideal": [], "chance": []}
for step in range(args.steps):
    batch = world.sample_episodes(args.episodes, split="train")
    traces = run_observers(batch)
    observations, var_ids, goal_index, goal_value, target = as_tensors(batch, traces)

    belief, _ = chain(observations, var_ids)
    loss = distillation_loss(belief, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        chain_goal = select_goal(belief, goal_index)
        correct = (chain_goal[:, -1].argmax(-1) == goal_value).float().mean()
    history["chain accuracy"].append(correct.item())
    history["ideal"].append(float(traces["joint"].accuracy[:, -1].mean()))
    history["chance"].append(1 / cfg.n_realizations)

    if step % 250 == 0:
        print(f"step {step:>4}  loss {loss.item():.4f}  accuracy {correct.item():.3f}")

print(f"\n{'':10}  {'chain':>7} {'joint':>7} {'naive':>7}")
for split in ("train", "held_out"):
    scores = evaluate(split)
    print(
        f"{split:10}  {scores['chain']:>7.3f} {scores['joint']:>7.3f} "
        f"{scores['naive']:>7.3f}"
    )
print(f"{'chance':10}  {1 / cfg.n_realizations:>7.3f}")

from rcc.viz import plot_training  # noqa: E402

figure = plot_training(history, smooth=25)
if args.out is not None:
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "coggrid_transfer.png"
    figure.savefig(path, dpi=110, bbox_inches="tight")
    print("wrote", path)
else:
    import matplotlib.pyplot as plt

    plt.show()
