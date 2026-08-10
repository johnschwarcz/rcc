"""Train a chain to match an exact posterior, and watch it accumulate evidence.

    python examples/quickstart.py

The chain is handed the world's true embeddings, which isolates stage 2: the only
thing learned here is how to turn an observation stream into a belief. See
``self_supervised.py`` for the run where stage 1 has to find the representation
for itself.

Episodes are drawn fresh every step from the world's own generator, so nothing
can be memorized; ``--seed`` pins that generator and the chain's initialization.
"""

import torch
from _common import (
    accuracy,
    arguments,
    describe,
    draw,
    embeddings,
    save_or_show,
    tail,
    world_and_config,
)
from rcc import RCC, select_goal, supervised_loss

# Every task and architecture parameter is a flag; `--help` lists them.
# Pin one here and the command line still overrides it, e.g.
#     args = arguments(n_realizations=8, hidden_dim=512)
args = arguments()
world, cfg = world_and_config(args, learn_embeddings=False)
print(describe(args, cfg))
chain = RCC(cfg, **embeddings(world))
optimizer = torch.optim.Adam(chain.inference_parameters(), lr=3e-3)

history: dict[str, list[float]] = {"chain accuracy": [], "ideal": [], "chance": []}
for iteration in range(args.iterations):
    batch = draw(world, args.episodes)
    belief, _ = chain(batch.observations, batch.ctx_inds)

    loss = supervised_loss(belief, batch.posterior)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        correct = accuracy(select_goal(belief, batch.goal_ind), batch.goal_value)
    history["chain accuracy"].append(correct)
    history["ideal"].append(batch.ideal)
    history["chance"].append(1 / cfg.n_realizations)

    if iteration % 100 == 0:
        print(
            f"iter {iteration:>4}  loss {loss.item():.4f}  "
            f"accuracy {correct:.3f}  ideal {batch.ideal:.3f}"
        )

print(
    f"\nfinal   accuracy {tail(history['chain accuracy']):.3f}"
    f"   ideal {tail(history['ideal']):.3f}"
    f"   chance {1 / cfg.n_realizations:.3f}"
)

# Plotting is optional, so it is imported only once the training has finished.
from rcc.viz import plot_belief_accumulation, plot_training  # noqa: E402

save_or_show(
    {
        "training": plot_training(history, smooth=25),
        "belief_accumulation": plot_belief_accumulation(
            belief, batch.goal_ind, batch.goal_value, batch.posterior
        ),
    },
    args.out,
)
