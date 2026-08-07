"""Learn the representation from prediction error, with no target belief.

    python examples/self_supervised.py

This is the run the architecture exists for. Stage 1's embeddings start random,
and nothing supervises them: the generator predicts observation rates from the
belief's answer, and its error is the only gradient the embeddings ever see. The
classifier trains at the same time and in the same loop, but the two never touch
— the interactions reach the classifier detached.

Both objectives are stepped with their own optimizer, which is why
:meth:`~rcc.RCC.estimation_parameters` and
:meth:`~rcc.RCC.inference_parameters` exist.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from rcc import (
    RCC,
    RCCConfig,
    distillation_loss,
    embedding_norm_penalty,
    goal_accuracy,
    prediction_loss,
    sample_goal,
    select_goal,
)
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
    learn_embeddings=True,
    seed=args.seed,
)
task = ToyTask(cfg, seed=args.seed)
chain = RCC(cfg)

# One optimizer per objective. They step the same chain and never the same
# parameter, which is what makes "disentangled" a fact rather than a hope.
estimation = torch.optim.Adam(chain.estimation_parameters(), lr=3e-3)
inference = torch.optim.Adam(chain.inference_parameters(), lr=3e-3)
rng = None if args.seed is None else torch.Generator().manual_seed(args.seed)

history: dict[str, list[float]] = {
    "chain accuracy": [],
    "ideal": [],
    "chance": [],
}
prediction_error: list[float] = []

for step in range(args.steps):
    episode = task.sample(args.episodes, n_steps=20, generator=rng)
    belief, interaction = chain(episode.observations, episode.var_ids)

    # Stage 2: match the exact posterior. Touches only the classifier.
    inference.zero_grad()
    distillation_loss(belief, episode.posterior).backward(retain_graph=True)

    # Stages 1 and 3: predict the observations. Touches only the embeddings
    # and the generator. The teaching signal is the verdict on the goal.
    selection = sample_goal(select_goal(belief, episode.goal_index), rng)[:, -1]
    correct = goal_accuracy(selection[:, None], episode.goal_value)[:, 0]
    rates = chain.reconstruct(
        belief,
        interaction,
        episode.goal_index,
        goal_selection=selection,
        goal_correct=correct,
        generator=rng,
    )
    estimation.zero_grad()
    generator_loss = prediction_loss(
        rates,
        episode.observations.mean(1),
        correct=correct,
        chance=1 / cfg.n_realizations,
    ) + embedding_norm_penalty(interaction.keys, interaction.queries)
    generator_loss.backward()

    inference.step()
    estimation.step()

    with torch.no_grad():
        ideal_goal = select_goal(episode.posterior, episode.goal_index)
        ideal = (ideal_goal[:, -1].argmax(-1) == episode.goal_value).float().mean()
    history["chain accuracy"].append(correct.mean().item())
    history["ideal"].append(ideal.item())
    history["chance"].append(1 / cfg.n_realizations)
    prediction_error.append(generator_loss.item())

    if step % 100 == 0:
        print(
            f"step {step:>4}  prediction {generator_loss.item():.4f}  "
            f"accuracy {correct.mean().item():.3f}  ideal {ideal.item():.3f}"
        )


def mean(values, n=50):
    """Average of the last ``n`` values, or of all of them if there are fewer.

    Dividing by ``n`` regardless would silently scale down every number a short
    run reports.
    """
    tail = values[-n:]
    return sum(tail) / len(tail)


print(
    f"\nprediction error  {mean(prediction_error[:50]):.3f} -> "
    f"{mean(prediction_error):.3f}"
)
print(
    f"accuracy          {mean(history['chain accuracy'][:50]):.3f} -> "
    f"{mean(history['chain accuracy']):.3f}"
    f"   (ideal {mean(history['ideal']):.3f}, chance {1 / cfg.n_realizations:.3f})"
)

from rcc.viz import plot_training  # noqa: E402

figures = {
    "self_supervised": plot_training(history, smooth=25),
    "prediction_error": plot_training(
        {"prediction error": prediction_error}, smooth=25, ylabel="prediction error"
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
