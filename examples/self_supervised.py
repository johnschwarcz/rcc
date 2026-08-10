"""Learn the representation from prediction error, with no target belief.

    python examples/self_supervised.py

This is the run the architecture exists for. Stage 1's embeddings start random and
nothing supervises them: the generator predicts observation rates from the
belief's answer, and its error is the only gradient the embeddings ever see. The
classifier trains in the same loop and the two never touch, because the
interactions reach the classifier detached.

Both objectives are stepped with their own optimizer, which is why
:meth:`~rcc.RCC.estimation_parameters` and :meth:`~rcc.RCC.inference_parameters`
exist.
"""

import torch
from _common import (
    arguments,
    describe,
    draw,
    save_or_show,
    tail,
    world_and_config,
)
from rcc import (
    RCC,
    embedding_norm_penalty,
    goal_accuracy,
    prediction_loss,
    sample_goal,
    select_goal,
    supervised_loss,
)

# Every task and architecture parameter is a flag; `--help` lists them.
# Pin one here and the command line still overrides it, e.g.
#     args = arguments(n_realizations=8, hidden_dim=512)
args = arguments()
world, cfg = world_and_config(args, learn_embeddings=True)
chance = 1 / cfg.n_realizations
print(describe(args, cfg))
chain = RCC(cfg)

# One optimizer per objective. They step the same chain and never the same
# parameter, which is what makes "disentangled" a fact rather than a hope.
estimation = torch.optim.Adam(chain.estimation_parameters(), lr=3e-3)
inference = torch.optim.Adam(chain.inference_parameters(), lr=3e-3)
rng = None if args.seed is None else torch.Generator().manual_seed(args.seed)

history: dict[str, list[float]] = {"chain accuracy": [], "ideal": [], "chance": []}
prediction_error: list[float] = []

for iteration in range(args.iterations):
    batch = draw(world, args.episodes)
    belief, interaction = chain(batch.observations, batch.ctx_inds)

    # Stage 2: match the exact posterior. Touches only the classifier.
    inference.zero_grad()
    supervised_loss(belief, batch.posterior).backward(retain_graph=True)

    # Stages 1 and 3: predict the observations. Touches only the embeddings and
    # the generator. The teaching signal is the verdict on the goal.
    selection = sample_goal(select_goal(belief, batch.goal_ind), rng)[:, -1]
    correct = goal_accuracy(selection[:, None], batch.goal_value)[:, 0]
    rates = chain.reconstruct(
        belief, interaction, batch.goal_ind,
        goal_selection=selection, goal_correct=correct, generator=rng,
    )
    estimation.zero_grad()
    generator_loss = prediction_loss(
        rates, batch.observations.mean(1), correct=correct, chance=chance
    ) + embedding_norm_penalty(interaction.keys, interaction.queries)
    generator_loss.backward()

    inference.step()
    estimation.step()

    history["chain accuracy"].append(correct.mean().item())
    history["ideal"].append(batch.ideal)
    history["chance"].append(chance)
    prediction_error.append(generator_loss.item())

    if iteration % 100 == 0:
        print(
            f"iter {iteration:>4}  prediction {generator_loss.item():.4f}  "
            f"accuracy {correct.mean().item():.3f}  ideal {batch.ideal:.3f}"
        )

print(
    f"\nprediction error  {tail(prediction_error[:50]):.3f} -> "
    f"{tail(prediction_error):.3f}"
)
print(
    f"accuracy          {tail(history['chain accuracy'][:50]):.3f} -> "
    f"{tail(history['chain accuracy']):.3f}"
    f"   (ideal {tail(history['ideal']):.3f}, chance {chance:.3f})"
)

from rcc.viz import plot_training  # noqa: E402

save_or_show(
    {
        "self_supervised": plot_training(history, smooth=25),
        "prediction_error": plot_training(
            {"prediction error": prediction_error}, smooth=25, ylabel="prediction error"
        ),
    },
    args.out,
)
