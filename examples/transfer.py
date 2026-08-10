"""Train on familiar variables, test on variables the chain has never seen.

    python examples/transfer.py

This is the experiment the architecture was built for. coggrid holds out a slice
of its variable pool, so "novel combination of known variables" is something you
can measure rather than assert.

The comparison worth watching is the last two columns. The joint observer knows
the interactions; the naive one assumes the active variables are independent. The
gap between them is the factorization regret, and it is the part of the task a
chain can only get right by having represented the interaction structure.

This demonstrates the setup; it is not a replication. A run of this length beats
the naive observer on familiar variables and does *not* transfer to held-out ones.
The paper's transfer result comes from a reinforcement-learning run over far more
episodes, with the embeddings learned rather than supplied. Treat the held-out row
as a baseline to improve on.
"""

import torch
from _common import (
    accuracy,
    arguments,
    describe,
    draw,
    embeddings,
    save_or_show,
    world_and_config,
)
from rcc import RCC, select_goal, supervised_loss

# Every task and architecture parameter is a flag; `--help` lists them.
# Pin one here and the command line still overrides it, e.g.
#     args = arguments(n_realizations=8, hidden_dim=512)
args = arguments(iterations=2500)
world, cfg = world_and_config(args, learn_embeddings=False)

# coggrid's embeddings *are* the representation stage 1 would otherwise learn, so
# handing them over isolates the classifier.
print(describe(args, cfg))
chain = RCC(cfg, **embeddings(world))
optimizer = torch.optim.Adam(chain.inference_parameters(), lr=3e-3)


def evaluate(split, n_episodes=1024):
    """The chain against both ideal observers, on a fixed batch of ``split``."""
    batch = draw(world, n_episodes, split=split, rng=args.seed)
    with torch.no_grad():
        belief, _ = chain(batch.observations, batch.ctx_inds)
    goal = select_goal(belief, batch.goal_ind)
    return accuracy(goal, batch.goal_value), batch.ideal, batch.naive


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

    if iteration % 250 == 0:
        print(f"iter {iteration:>4}  loss {loss.item():.4f}  accuracy {correct:.3f}")

print(f"\n{'':10}  {'chain':>7} {'joint':>7} {'naive':>7}")
for split in ("train", "held_out"):
    chain_score, joint, naive = evaluate(split)
    print(f"{split:10}  {chain_score:>7.3f} {joint:>7.3f} {naive:>7.3f}")
print(f"{'chance':10}  {1 / cfg.n_realizations:>7.3f}")

from rcc.viz import plot_training  # noqa: E402

save_or_show({"transfer": plot_training(history, smooth=25)}, args.out)
