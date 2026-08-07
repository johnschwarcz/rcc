"""Train a chain to match an exact posterior, and watch it accumulate evidence.

    python examples/quickstart.py

Uses :class:`~rcc.toy.ToyTask` so it runs anywhere, with no environment to
install. The chain is handed the task's true embeddings, which isolates stage 2:
the only thing being learned here is how to turn an observation stream into a
belief. See ``self_supervised.py`` for the run where stage 1 has to find the
representation for itself.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from rcc import RCC, RCCConfig, distillation_loss, select_goal
from rcc.toy import ToyTask

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("--out", type=Path, default=None, help="save figures here")
parser.add_argument("--steps", type=int, default=1500)
parser.add_argument("--episodes", type=int, default=128)
parser.add_argument(
    "--seed", type=int, default=None,
    help="pin the run; the default explores fresh randomness",
)
args, _ignored = parser.parse_known_args()

cfg = RCCConfig(
    n_vars=4,
    n_contexts=2,
    n_realizations=4,
    n_observations=5,
    embedding_dim=6,
    hidden_dim=64,
    learn_embeddings=False,
    seed=args.seed,
)
task = ToyTask(cfg, seed=args.seed)
chain = RCC(cfg, keys=task.keys, queries=task.queries)
optimizer = torch.optim.Adam(chain.inference_parameters(), lr=3e-3)
rng = None if args.seed is None else torch.Generator().manual_seed(args.seed)

history: dict[str, list[float]] = {"chain accuracy": [], "ideal": [], "chance": []}
for step in range(args.steps):
    episode = task.sample(args.episodes, n_steps=20, generator=rng)
    belief, _ = chain(episode.observations, episode.var_ids)

    loss = distillation_loss(belief, episode.posterior)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        chain_goal = select_goal(belief, episode.goal_index)
        ideal_goal = select_goal(episode.posterior, episode.goal_index)
        correct = (chain_goal[:, -1].argmax(-1) == episode.goal_value).float().mean()
        ideal = (ideal_goal[:, -1].argmax(-1) == episode.goal_value).float().mean()
    history["chain accuracy"].append(correct.item())
    history["ideal"].append(ideal.item())
    history["chance"].append(1 / cfg.n_realizations)

    if step % 100 == 0:
        print(
            f"step {step:>4}  loss {loss.item():.4f}  "
            f"accuracy {correct.item():.3f}  ideal {ideal.item():.3f}"
        )


def mean(values, n=50):
    """Average of the last ``n`` values, or of all of them if there are fewer.

    Dividing by ``n`` regardless would silently scale down every number a short
    run reports.
    """
    tail = values[-n:]
    return sum(tail) / len(tail)


print(
    f"\nfinal   accuracy {mean(history['chain accuracy']):.3f}"
    f"   ideal {mean(history['ideal']):.3f}"
    f"   chance {1 / cfg.n_realizations:.3f}"
)

# Plotting is optional, so it is imported only once the training has finished.
from rcc.viz import plot_belief_accumulation, plot_training  # noqa: E402

figures = {
    "training": plot_training(history, smooth=25),
    "belief_accumulation": plot_belief_accumulation(
        belief, episode.goal_index, episode.goal_value, episode.posterior
    ),
}
if args.out is not None:
    args.out.mkdir(parents=True, exist_ok=True)
    for name, figure in figures.items():
        path = args.out / f"{name}.png"
        figure.savefig(path, dpi=110, bbox_inches="tight")
        print("wrote", path)
else:
    import matplotlib.pyplot as plt

    plt.show()
