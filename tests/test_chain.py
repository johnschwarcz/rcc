"""The assembled chain, and the separation it exists to enforce.

The claim in the paper is that variable inference and parameter estimation are
disentangled. In this code that claim is not a comment, it is a fact about where
gradient can flow, so it is tested as one.
"""

import pytest
import torch
from rcc import (
    RCC,
    RCCConfig,
    embedding_norm_penalty,
    goal_accuracy,
    intrinsic_value,
    prediction_loss,
    reward_loss,
    sample_categorical,
    select_goal,
    supervised_loss,
)
from rcc.losses import controller_loss

CFG = RCCConfig(
    n_vars=13,
    n_contexts=2,
    n_realizations=4,
    n_observations=3,
    embedding_dim=5,
    hidden_dim=16,
    seed=0,
)
N_EPISODES, N_STEPS = 6, 8


@pytest.fixture
def chain():
    return RCC(CFG)


@pytest.fixture
def inputs():
    torch.manual_seed(0)
    return (
        torch.rand(N_EPISODES, N_STEPS, CFG.n_observations).round(),
        torch.randint(0, CFG.n_vars, (N_EPISODES, CFG.n_contexts)),
    )


def goal_ind():
    return torch.randint(0, CFG.n_contexts, (N_EPISODES,))


# --------------------------------------------------------------------------- #
# the seam
# --------------------------------------------------------------------------- #
def test_classification_error_never_reaches_the_embeddings(chain, inputs):
    """The architecture's central claim, as a property of the gradient graph."""
    belief, _ = chain(*inputs)
    target = torch.softmax(torch.randn_like(belief[:, -1]), -1)
    supervised_loss(belief[:, -1], target).backward()

    assert chain.encoder.keys.grad is None
    assert chain.encoder.queries.grad is None
    assert any(p.grad is not None for p in chain.classifier.parameters())


def test_reward_error_never_reaches_the_embeddings_either(chain, inputs):
    belief, _ = chain(*inputs)
    goal_belief = select_goal(belief, goal_ind())
    selection = sample_categorical(goal_belief, torch.Generator().manual_seed(0))[:, -1]
    correct = goal_accuracy(
        selection[:, None], torch.randint(0, CFG.n_realizations, (N_EPISODES,))
    )[:, 0]
    reward_loss(goal_belief, selection, correct).backward()

    assert chain.encoder.keys.grad is None
    assert chain.encoder.queries.grad is None


def test_prediction_error_does_reach_the_embeddings(chain, inputs):
    """The other half of the claim: something has to train stage 1."""
    belief, interaction = chain(*inputs)
    rates = chain.reconstruct(belief, interaction, goal_ind())
    prediction_loss(rates, inputs[0].mean(1)).backward()

    assert chain.encoder.keys.grad is not None
    assert chain.encoder.keys.grad.abs().sum() > 0
    assert all(p.grad is None for p in chain.classifier.parameters())


def test_control_error_never_reaches_the_embeddings(chain, inputs):
    """The controller reads the representation; it does not get to rewrite it."""
    _, interaction = chain(*inputs)
    policy, value = chain.controller(interaction.score)
    _, log_prob, entropy = chain.controller.act(policy, torch.Generator().manual_seed(0))
    controller_loss(torch.rand(N_EPISODES), value, log_prob, entropy).backward()

    assert chain.encoder.keys.grad is None
    assert any(p.grad is not None for p in chain.controller.parameters())


def test_reconstruction_does_not_backpropagate_through_the_sampled_belief(chain, inputs):
    """Stage 3 scores the representation, not the belief that queried it."""
    belief, interaction = chain(*inputs)
    rates = chain.reconstruct(belief, interaction, goal_ind())
    prediction_loss(rates, inputs[0].mean(1)).backward()
    assert all(p.grad is None for p in chain.classifier.parameters())


# --------------------------------------------------------------------------- #
# parameter groups
# --------------------------------------------------------------------------- #
def test_the_three_groups_partition_the_chain(chain):
    groups = {
        "estimation": {id(p) for p in chain.estimation_parameters()},
        "inference": {id(p) for p in chain.inference_parameters()},
        "control": {id(p) for p in chain.control_parameters()},
    }
    names = list(groups)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            assert not groups[first] & groups[second], f"{first} overlaps {second}"
    assert set().union(*groups.values()) == {id(p) for p in chain.parameters()}


def test_embeddings_belong_to_the_estimation_group(chain):
    """Not to the classifier's — which is the whole point of the separation."""
    assert id(chain.encoder.keys) in {id(p) for p in chain.estimation_parameters()}
    assert id(chain.encoder.keys) not in {id(p) for p in chain.inference_parameters()}


def test_optimizing_one_group_leaves_the_others_untouched(chain, inputs):
    optimizer = torch.optim.Adam(chain.inference_parameters(), lr=0.1)
    before = [p.clone() for p in chain.estimation_parameters()]

    belief, _ = chain(*inputs)
    target = torch.softmax(torch.randn_like(belief[:, -1]), -1)
    loss = supervised_loss(belief[:, -1], target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    after = list(chain.estimation_parameters())
    assert all(torch.equal(a, b) for a, b in zip(after, before, strict=True))


# --------------------------------------------------------------------------- #
# assembly
# --------------------------------------------------------------------------- #
def test_forward_shapes(chain, inputs):
    belief, interaction = chain(*inputs)
    assert belief.shape == (N_EPISODES, N_STEPS, CFG.n_contexts, CFG.n_realizations)
    assert interaction.score.shape == (
        N_EPISODES,
        CFG.n_observations,
        CFG.n_interactions,
    )
    assert interaction.keys.shape == (
        N_EPISODES,
        CFG.n_contexts,
        CFG.n_observations,
        CFG.embedding_dim,
    )


def test_fixed_embeddings_flow_through_the_chain(inputs):
    shape = (CFG.n_vars, CFG.n_observations, CFG.embedding_dim)
    keys, queries = torch.randn(shape), torch.randn(shape)
    chain = RCC(CFG.replace(learn_embeddings=False), keys=keys, queries=queries)
    belief, _ = chain(*inputs)
    assert torch.isfinite(belief).all()
    assert not any(p.requires_grad for p in chain.encoder.parameters())


def test_state_dict_round_trips(chain, inputs):
    twin = RCC(CFG.replace(seed=999))
    belief_before, _ = twin(*inputs)
    twin.load_state_dict(chain.state_dict())
    belief_after, _ = twin(*inputs)
    expected, _ = chain(*inputs)

    assert not torch.allclose(belief_before, expected, atol=1e-4)
    assert torch.allclose(belief_after, expected, atol=1e-6)


def test_seeding_makes_a_chain_reproducible(inputs):
    first, _ = RCC(CFG)(*inputs)
    second, _ = RCC(CFG)(*inputs)
    assert torch.allclose(first, second, atol=1e-6)


def test_an_unseeded_chain_does_not_disturb_global_randomness():
    torch.manual_seed(42)
    expected = torch.randn(4)
    torch.manual_seed(42)
    RCC(CFG.replace(seed=None))
    assert not torch.equal(expected, torch.randn(4))

    # ...whereas a seeded one puts the stream back exactly as it found it.
    torch.manual_seed(42)
    RCC(CFG)
    assert torch.equal(expected, torch.randn(4))


def test_the_chain_moves_to_a_device_in_one_call(chain, inputs):
    """``.to()`` has to carry the fixed buffers too, not just the parameters."""
    moved = chain.to(torch.float64)
    belief, interaction = moved(inputs[0].double(), inputs[1])
    assert belief.dtype == torch.float64
    assert interaction.score.dtype == torch.float64


def test_reconstruct_accepts_the_teaching_signal(chain, inputs):
    belief, interaction = chain(*inputs)
    rates = chain.reconstruct(
        belief,
        interaction,
        goal_ind(),
        goal_selection=torch.randint(0, CFG.n_realizations, (N_EPISODES,)),
        goal_correct=torch.rand(N_EPISODES).round(),
    )
    assert rates.shape == (N_EPISODES, CFG.n_observations)


def test_a_full_training_step_touches_every_stage(chain, inputs):
    """One step of all three objectives at once, as a real loop would run it."""
    observations, ctx_inds = inputs
    optimizers = {
        "estimation": torch.optim.Adam(chain.estimation_parameters(), lr=1e-3),
        "inference": torch.optim.Adam(chain.inference_parameters(), lr=1e-3),
        "control": torch.optim.Adam(chain.control_parameters(), lr=1e-3),
    }
    for optimizer in optimizers.values():
        optimizer.zero_grad()

    belief, interaction = chain(observations, ctx_inds)
    goals = goal_ind()

    target = torch.softmax(torch.randn_like(belief[:, -1]), -1)
    supervised_loss(belief[:, -1], target).backward(retain_graph=True)

    rates = chain.reconstruct(belief, interaction, goals)
    generator_loss = prediction_loss(
        rates, observations.mean(1)
    ) + embedding_norm_penalty(interaction.keys, interaction.queries)
    generator_loss.backward(retain_graph=True)

    policy, value = chain.controller(interaction.score)
    actions, log_prob, entropy = chain.controller.act(policy)
    reward = intrinsic_value(rates.detach(), torch.rand(CFG.n_observations).round())
    controller_loss(reward, value, log_prob, entropy).backward()

    for optimizer in optimizers.values():
        optimizer.step()

    assert actions.shape == (N_EPISODES, CFG.n_contexts)
    assert all(torch.isfinite(p).all() for p in chain.parameters())
