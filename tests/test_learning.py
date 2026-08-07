"""The chain has to actually learn, not merely produce correctly-shaped tensors.

Every test here trains a real chain on :class:`~rcc.toy.ToyTask` and asserts that
something improved. The budgets are tiny — a few hundred steps against the
paper's hundred thousand plus — so the thresholds test *progress*, not mastery.
They are set well inside what was measured across several seeds, so a failure
here means something broke rather than that a run was unlucky.
"""

from __future__ import annotations

import pytest
import torch

from rcc import (
    RCC,
    RCCConfig,
    controller_loss,
    distillation_loss,
    embedding_norm_penalty,
    intrinsic_value,
    prediction_loss,
    select_goal,
)
from rcc.toy import ToyTask

SHAPE = dict(
    n_vars=8,
    n_contexts=2,
    n_realizations=4,
    n_observations=5,
    embedding_dim=6,
    hidden_dim=64,
)
CHANCE = 1 / SHAPE["n_realizations"]


def taught_chain(seed=0):
    """A chain handed the task's true embeddings, so only stage 2 has to learn."""
    cfg = RCCConfig(**SHAPE, learn_embeddings=False, seed=seed)
    task = ToyTask(cfg, seed=seed)
    return cfg, task, RCC(cfg, keys=task.keys, queries=task.queries)


def goal_accuracy_of(belief, episode):
    goal_belief = select_goal(belief.detach(), episode.goal_index)
    return (goal_belief[:, -1].argmax(-1) == episode.goal_value).float().mean().item()


# --------------------------------------------------------------------------- #
# stage 2 learns
# --------------------------------------------------------------------------- #
def test_distillation_fits_a_batch_and_approaches_the_ideal_observer():
    """The strongest statement a fast test can make: the architecture can
    represent the exact posterior, not just move towards it."""
    cfg, task, chain = taught_chain()
    episode = task.sample(128, n_steps=20, generator=torch.Generator().manual_seed(0))
    optimizer = torch.optim.Adam(chain.inference_parameters(), lr=3e-3)

    first = None
    for step in range(300):
        belief, _ = chain(episode.observations, episode.var_ids)
        loss = distillation_loss(belief, episode.posterior)
        if step == 0:
            first = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    ideal = select_goal(episode.posterior, episode.goal_index)
    ideal_accuracy = (
        (ideal[:, -1].argmax(-1) == episode.goal_value).float().mean().item()
    )
    accuracy = goal_accuracy_of(belief, episode)

    assert loss.item() < first / 2
    assert accuracy > CHANCE + 0.25
    assert accuracy > ideal_accuracy - 0.25


def test_a_streaming_chain_improves_on_unseen_episodes():
    """Fresh episodes every step, so nothing can be memorized."""
    cfg, task, chain = taught_chain()
    optimizer = torch.optim.Adam(chain.inference_parameters(), lr=3e-3)
    rng = torch.Generator().manual_seed(0)

    accuracy = []
    for _ in range(400):
        episode = task.sample(128, n_steps=20, generator=rng)
        belief, _ = chain(episode.observations, episode.var_ids)
        loss = distillation_loss(belief, episode.posterior)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        accuracy.append(goal_accuracy_of(belief, episode))

    early = sum(accuracy[:50]) / 50
    late = sum(accuracy[-50:]) / 50
    assert late > early
    assert late > CHANCE


def test_a_trained_belief_sharpens_as_evidence_arrives():
    """What the accumulator is for: later steps should be more certain."""
    cfg, task, chain = taught_chain()
    episode = task.sample(128, n_steps=20, generator=torch.Generator().manual_seed(0))
    optimizer = torch.optim.Adam(chain.inference_parameters(), lr=3e-3)
    for _ in range(300):
        belief, _ = chain(episode.observations, episode.var_ids)
        loss = distillation_loss(belief, episode.posterior)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    goal_belief = select_goal(belief.detach(), episode.goal_index)
    entropy = -(goal_belief * goal_belief.clamp_min(1e-12).log()).sum(-1).mean(0)
    assert entropy[-1] < entropy[0]

    correct = (goal_belief.argmax(-1) == episode.goal_value[:, None]).float().mean(0)
    assert correct[-1] > correct[0]


# --------------------------------------------------------------------------- #
# stages 1 and 3 learn, without labels
# --------------------------------------------------------------------------- #
def test_prediction_error_trains_the_embeddings():
    cfg = RCCConfig(**SHAPE, learn_embeddings=True, seed=0)
    task = ToyTask(cfg, seed=0)
    chain = RCC(cfg)
    optimizer = torch.optim.Adam(chain.estimation_parameters(), lr=3e-3)
    rng = torch.Generator().manual_seed(0)
    before = chain.encoder.keys.detach().clone()

    losses = []
    for _ in range(150):
        episode = task.sample(128, n_steps=20, generator=rng)
        belief, interaction = chain(episode.observations, episode.var_ids)
        rates = chain.reconstruct(belief, interaction, episode.goal_index)
        loss = prediction_loss(
            rates, episode.observations.mean(1)
        ) + embedding_norm_penalty(interaction.keys, interaction.queries)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert sum(losses[-20:]) / 20 < sum(losses[:20]) / 20 * 0.6
    assert not torch.allclose(chain.encoder.keys.detach(), before, atol=1e-4)


def test_the_teaching_signal_does_not_break_prediction():
    """The goal slot is overwritten with a verdict; training must still descend."""
    cfg = RCCConfig(**SHAPE, learn_embeddings=True, seed=0)
    task = ToyTask(cfg, seed=0)
    chain = RCC(cfg)
    optimizer = torch.optim.Adam(chain.estimation_parameters(), lr=3e-3)
    rng = torch.Generator().manual_seed(0)

    losses = []
    for _ in range(150):
        episode = task.sample(128, n_steps=20, generator=rng)
        belief, interaction = chain(episode.observations, episode.var_ids)
        goal_belief = select_goal(belief, episode.goal_index)
        selection = goal_belief[:, -1].argmax(-1)
        correct = (selection == episode.goal_value).float()
        rates = chain.reconstruct(
            belief,
            interaction,
            episode.goal_index,
            goal_selection=selection,
            goal_correct=correct,
        )
        loss = prediction_loss(
            rates,
            episode.observations.mean(1),
            correct=correct,
            chance=CHANCE,
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
    cfg, task, chain = taught_chain()
    optimizer = torch.optim.Adam(chain.control_parameters(), lr=3e-3)
    rng = torch.Generator().manual_seed(0)
    preferences = (torch.rand(cfg.n_observations, generator=rng) > 0.5).float()

    rewards = []
    for _ in range(200):
        episode = task.sample(128, n_steps=1, generator=rng)
        interaction = chain.encoder(episode.var_ids)
        policy, value = chain.controller(interaction.strength)
        actions, log_prob, entropy = chain.controller.act(policy, rng)

        table = task.rate_table(episode.var_ids).reshape(128, cfg.n_observations, -1)
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


@pytest.mark.parametrize("recurrent", [True, False])
def test_both_architectures_train_without_blowing_up(recurrent):
    cfg = RCCConfig(**SHAPE, learn_embeddings=False, recurrent=recurrent, seed=0)
    task = ToyTask(cfg, seed=0)
    chain = RCC(cfg, keys=task.keys, queries=task.queries)
    optimizer = torch.optim.Adam(chain.inference_parameters(), lr=3e-3)
    rng = torch.Generator().manual_seed(0)

    for _ in range(50):
        episode = task.sample(64, n_steps=10, generator=rng)
        belief, _ = chain(episode.observations, episode.var_ids)
        loss = distillation_loss(belief, episode.posterior)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    assert torch.isfinite(loss)
    assert all(torch.isfinite(p).all() for p in chain.inference_parameters())
