"""The chain has to actually learn, not merely produce correctly-shaped tensors.

Every test here trains a real chain against a real environment — coggrid, a
development dependency; nothing in ``src/rcc`` imports it — and asserts that
something improved. The budgets are tiny, a few hundred steps against the paper's
hundred thousand plus, so the thresholds test *progress*, not mastery. They sit
well inside what was measured across several seeds, so a failure here means
something broke rather than that a run was unlucky.
"""

import torch
from _common import accuracy, draw, embeddings, world_and_config
from rcc import (
    RCC,
    controller_loss,
    embedding_norm_penalty,
    intrinsic_value,
    prediction_loss,
    select_goal,
    supervised_loss,
)


def taught_chain(seed=0, **overrides):
    """A chain handed the world's true embeddings, so only stage 2 has to learn."""
    world, cfg = world_and_config(seed=seed, learn_embeddings=False, **overrides)
    return world, cfg, RCC(cfg, **embeddings(world))


def learning_chain(seed=0):
    """A chain that has to find the representation for itself."""
    world, cfg = world_and_config(seed=seed, learn_embeddings=True)
    return world, cfg, RCC(cfg)


def goal_accuracy_of(belief, batch):
    return accuracy(select_goal(belief.detach(), batch.goal_ind), batch.goal_value)


# --------------------------------------------------------------------------- #
# stage 2 learns
# --------------------------------------------------------------------------- #
def test_distillation_fits_a_batch_and_approaches_the_ideal_observer():
    """The strongest statement a fast test can make: the architecture can
    represent the exact posterior, not just move towards it."""
    world, cfg, chain = taught_chain()
    batch = draw(world, 128, rng=0)
    optimizer = torch.optim.Adam(chain.inference_parameters(), lr=3e-3)

    first = None
    for step in range(300):
        belief, _ = chain(batch.observations, batch.ctx_inds)
        loss = supervised_loss(belief, batch.posterior)
        if step == 0:
            first = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    chance = 1 / cfg.n_realizations
    assert loss.item() < first / 2
    assert goal_accuracy_of(belief, batch) > chance + 0.25
    assert goal_accuracy_of(belief, batch) > batch.ideal - 0.25


def test_a_streaming_chain_improves_on_unseen_episodes():
    """Fresh episodes every step, so nothing can be memorized."""
    world, cfg, chain = taught_chain()
    optimizer = torch.optim.Adam(chain.inference_parameters(), lr=3e-3)

    scores = []
    for _ in range(400):
        batch = draw(world, 128)
        belief, _ = chain(batch.observations, batch.ctx_inds)
        loss = supervised_loss(belief, batch.posterior)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scores.append(goal_accuracy_of(belief, batch))

    early = sum(scores[:50]) / 50
    late = sum(scores[-50:]) / 50
    assert late > early
    assert late > 1 / cfg.n_realizations


def test_a_trained_belief_sharpens_as_evidence_arrives():
    """What the accumulator is for: later steps should be more certain."""
    world, cfg, chain = taught_chain()
    batch = draw(world, 128, rng=0)
    optimizer = torch.optim.Adam(chain.inference_parameters(), lr=3e-3)
    for _ in range(300):
        belief, _ = chain(batch.observations, batch.ctx_inds)
        loss = supervised_loss(belief, batch.posterior)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    goal_belief = select_goal(belief.detach(), batch.goal_ind)
    entropy = -(goal_belief * goal_belief.clamp_min(1e-12).log()).sum(-1).mean(0)
    assert entropy[-1] < entropy[0]

    correct = (goal_belief.argmax(-1) == batch.goal_value[:, None]).float().mean(0)
    assert correct[-1] > correct[0]


# --------------------------------------------------------------------------- #
# stages 1 and 3 learn, without labels
# --------------------------------------------------------------------------- #
def test_prediction_error_trains_the_embeddings():
    world, cfg, chain = learning_chain()
    optimizer = torch.optim.Adam(chain.estimation_parameters(), lr=3e-3)
    before = chain.encoder.keys.detach().clone()

    losses = []
    for _ in range(150):
        batch = draw(world, 128)
        belief, interaction = chain(batch.observations, batch.ctx_inds)
        rates = chain.reconstruct(belief, interaction, batch.goal_ind)
        loss = prediction_loss(
            rates, batch.observations.mean(1)
        ) + embedding_norm_penalty(interaction.keys, interaction.queries)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert sum(losses[-20:]) / 20 < sum(losses[:20]) / 20 * 0.6
    assert not torch.allclose(chain.encoder.keys.detach(), before, atol=1e-4)


def test_the_teaching_signal_does_not_break_prediction():
    """The goal slot is overwritten with a verdict; training must still descend."""
    world, cfg, chain = learning_chain()
    optimizer = torch.optim.Adam(chain.estimation_parameters(), lr=3e-3)

    losses = []
    for _ in range(150):
        batch = draw(world, 128)
        belief, interaction = chain(batch.observations, batch.ctx_inds)
        selection = select_goal(belief, batch.goal_ind)[:, -1].argmax(-1)
        correct = (selection == batch.goal_value).float()
        rates = chain.reconstruct(
            belief, interaction, batch.goal_ind,
            goal_selection=selection, goal_correct=correct,
        )
        loss = prediction_loss(
            rates,
            batch.observations.mean(1),
            correct=correct,
            chance=1 / cfg.n_realizations,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert all(torch.isfinite(torch.tensor(losses)))
    assert sum(losses[-20:]) / 20 < sum(losses[:20]) / 20


# --------------------------------------------------------------------------- #
# stage 4 learns
# --------------------------------------------------------------------------- #
def test_the_controller_learns_to_prefer_valuable_actions():
    world, cfg, chain = taught_chain()
    optimizer = torch.optim.Adam(chain.control_parameters(), lr=3e-3)
    rng = torch.Generator().manual_seed(0)
    preferences = (torch.rand(cfg.n_observations, generator=rng) > 0.5).float()

    rewards = []
    for _ in range(200):
        batch = draw(world, 128)
        policy, value = chain.controller(chain.encoder(batch.ctx_inds).score)
        actions, log_prob, entropy = chain.controller.act(policy, rng)

        # coggrid's likelihood table, indexed at the joint action just taken.
        table = batch.rates.reshape(128, cfg.n_observations, -1)
        flat = actions[:, 0] * cfg.n_realizations + actions[:, 1]
        rates = table.gather(
            -1, flat[:, None, None].expand(-1, cfg.n_observations, 1)
        ).squeeze(-1)

        reward = intrinsic_value(rates, preferences)
        loss = controller_loss(reward, value, log_prob, entropy)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        rewards.append(reward.mean().item())

    assert sum(rewards[-20:]) / 20 > sum(rewards[:20]) / 20 + 0.03


def test_training_does_not_blow_up():
    world, cfg, chain = taught_chain()
    optimizer = torch.optim.Adam(chain.inference_parameters(), lr=3e-3)

    for _ in range(50):
        batch = draw(world, 64)
        belief, _ = chain(batch.observations, batch.ctx_inds)
        loss = supervised_loss(belief, batch.posterior)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    assert torch.isfinite(loss)
    assert all(torch.isfinite(p).all() for p in chain.inference_parameters())
