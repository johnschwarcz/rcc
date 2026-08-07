"""Each stage on its own: shapes, invariants, and the errors it raises.

These are the tests that matter for using a stage outside a chain, which is the
supported way to use one.
"""

from __future__ import annotations

import pytest
import torch

from rcc import (
    BeliefClassifier,
    Controller,
    InteractionEncoder,
    ObservationGenerator,
    RCCConfig,
    goal_accuracy,
    intrinsic_value,
    ordered_pairs,
    query_from_belief,
    sample_goal,
    select_goal,
)

CFG = RCCConfig(
    n_vars=11,
    n_contexts=2,
    n_realizations=4,
    n_observations=3,
    embedding_dim=5,
    hidden_dim=16,
    seed=0,
)
N_EPISODES, N_STEPS = 6, 9


@pytest.fixture
def interactions():
    torch.manual_seed(0)
    return torch.randn(N_EPISODES, CFG.n_observations, CFG.n_interactions)


@pytest.fixture
def observations():
    torch.manual_seed(1)
    return torch.rand(N_EPISODES, N_STEPS, CFG.n_observations).round()


def fixed_embeddings(cfg=CFG):
    torch.manual_seed(2)
    shape = (cfg.n_vars, cfg.n_observations, cfg.embedding_dim)
    return torch.randn(shape), torch.randn(shape)


# --------------------------------------------------------------------------- #
# stage 1
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_contexts", [1, 2, 3, 4])
def test_ordered_pairs_covers_both_directions_of_every_pair(n_contexts):
    pairs = ordered_pairs(n_contexts)
    assert len(pairs) == RCCConfig(n_contexts=n_contexts).n_interactions
    assert len(set(pairs)) == len(pairs)
    if n_contexts > 1:
        assert all((q, k) in pairs for k, q in pairs)
        assert not any(k == q for k, q in pairs)


def test_encoder_rejects_supplied_embeddings_when_it_should_learn():
    keys, queries = fixed_embeddings()
    with pytest.raises(ValueError, match="learn_embeddings is True"):
        InteractionEncoder(CFG, keys=keys, queries=queries)


def test_encoder_demands_embeddings_when_it_cannot_learn():
    with pytest.raises(ValueError, match="both keys and queries"):
        InteractionEncoder(CFG.replace(learn_embeddings=False))


def test_encoder_rejects_misshapen_embeddings():
    keys, queries = fixed_embeddings()
    with pytest.raises(ValueError, match="queries must have shape"):
        InteractionEncoder(
            CFG.replace(learn_embeddings=False), keys=keys, queries=queries[:, :, :-1]
        )


def test_fixed_embeddings_are_buffers_not_parameters():
    """They travel with the module and are saved, but no optimizer sees them."""
    keys, queries = fixed_embeddings()
    encoder = InteractionEncoder(
        CFG.replace(learn_embeddings=False), keys=keys, queries=queries
    )
    assert not any(p.requires_grad for p in encoder.parameters())
    assert "keys" in encoder.state_dict()


def test_fixed_embeddings_are_copied_not_aliased():
    """Mutating the caller's tensor afterwards must not change the encoder."""
    keys, queries = fixed_embeddings()
    encoder = InteractionEncoder(
        CFG.replace(learn_embeddings=False), keys=keys, queries=queries
    )
    before = encoder.keys.clone()
    keys.zero_()
    assert torch.equal(encoder.keys, before)


def test_encoder_rejects_wrong_var_id_shape():
    encoder = InteractionEncoder(CFG)
    with pytest.raises(ValueError, match="var_ids must have shape"):
        encoder(torch.zeros(N_EPISODES, CFG.n_contexts + 1, dtype=torch.long))


def test_learned_embeddings_start_on_the_unit_sphere():
    encoder = InteractionEncoder(CFG)
    assert torch.allclose(
        encoder.keys.norm(dim=-1), torch.ones(CFG.n_vars, CFG.n_observations), atol=1e-5
    )


def test_a_repeated_variable_is_allowed():
    """The same variable may be active twice, as it is in coggrid by default."""
    encoder = InteractionEncoder(CFG)
    strength = encoder(torch.tensor([[3, 3]])).strength
    assert strength.shape == (1, CFG.n_observations, CFG.n_interactions)
    assert torch.isfinite(strength).all()


# --------------------------------------------------------------------------- #
# stage 2
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("recurrent", [True, False])
def test_belief_is_normalized_per_variable(observations, interactions, recurrent):
    classifier = BeliefClassifier(CFG.replace(recurrent=recurrent))
    belief = classifier(observations, interactions)
    assert belief.shape == (N_EPISODES, N_STEPS, CFG.n_contexts, CFG.n_realizations)
    assert torch.allclose(belief.sum(-1), torch.ones_like(belief.sum(-1)), atol=1e-5)


def test_feedforward_belief_does_not_move_over_time(observations, interactions):
    """The ablation's whole point: no sequential integration."""
    classifier = BeliefClassifier(CFG.replace(recurrent=False))
    belief = classifier(observations, interactions)
    assert torch.allclose(belief, belief[:, :1].expand_as(belief), atol=1e-6)


def test_recurrent_belief_does_move_over_time(observations, interactions):
    classifier = BeliefClassifier(CFG)
    belief = classifier(observations, interactions)
    assert not torch.allclose(belief[:, 0], belief[:, -1], atol=1e-4)


def test_reservoir_freezes_only_the_recurrence():
    reservoir = BeliefClassifier(CFG.replace(reservoir=True))
    assert not any(p.requires_grad for p in reservoir.rnn.parameters())
    assert reservoir.initial_short_term.requires_grad
    assert all(p.requires_grad for p in reservoir.readin.parameters())
    assert all(p.requires_grad for p in reservoir.readout.parameters())


def test_reservoir_recurrence_survives_a_backward_pass(observations, interactions):
    classifier = BeliefClassifier(CFG.replace(reservoir=True))
    before = [p.clone() for p in classifier.rnn.parameters()]
    belief = classifier(observations, interactions)
    belief.sum().backward()
    assert all(p.grad is None for p in classifier.rnn.parameters())
    frozen = zip(classifier.rnn.parameters(), before, strict=True)
    assert all(torch.equal(p, b) for p, b in frozen)


def test_classifier_rejects_wrong_observation_width(interactions):
    classifier = BeliefClassifier(CFG)
    with pytest.raises(ValueError, match="observations must have shape"):
        classifier(torch.rand(N_EPISODES, N_STEPS, CFG.n_observations + 1), interactions)


def test_classifier_rejects_mismatched_interactions(observations):
    classifier = BeliefClassifier(CFG)
    with pytest.raises(ValueError, match="interactions must have shape"):
        classifier(observations, torch.randn(N_EPISODES, CFG.n_observations, 99))


def test_classifier_accepts_any_episode_length(interactions):
    """n_steps is read off the data, never stored, so it can vary run to run."""
    classifier = BeliefClassifier(CFG)
    for n_steps in (1, 3, 40):
        observations = torch.rand(N_EPISODES, n_steps, CFG.n_observations).round()
        assert classifier(observations, interactions).shape[1] == n_steps


def test_select_goal_picks_the_named_variable():
    belief = torch.rand(N_EPISODES, N_STEPS, CFG.n_contexts, CFG.n_realizations)
    goal_index = torch.randint(0, CFG.n_contexts, (N_EPISODES,))
    goal_belief = select_goal(belief, goal_index)
    assert goal_belief.shape == (N_EPISODES, N_STEPS, CFG.n_realizations)
    for episode in range(N_EPISODES):
        assert torch.equal(goal_belief[episode], belief[episode, :, goal_index[episode]])


def test_select_goal_rejects_a_mismatched_index():
    belief = torch.rand(N_EPISODES, N_STEPS, CFG.n_contexts, CFG.n_realizations)
    with pytest.raises(ValueError, match="goal_index must have shape"):
        select_goal(belief, torch.zeros(N_EPISODES + 1, dtype=torch.long))


def test_sample_goal_is_reproducible_given_a_generator():
    goal_belief = torch.softmax(torch.randn(N_EPISODES, N_STEPS, 4), -1)
    draws = [
        sample_goal(goal_belief, torch.Generator().manual_seed(0)) for _ in range(2)
    ]
    assert torch.equal(*draws)


def test_goal_accuracy_rejects_a_mismatched_truth():
    with pytest.raises(ValueError, match="goal_value must have shape"):
        goal_accuracy(
            torch.zeros(3, 2, dtype=torch.long), torch.zeros(4, dtype=torch.long)
        )


# --------------------------------------------------------------------------- #
# stage 3
# --------------------------------------------------------------------------- #
def test_generator_rates_are_probabilities(interactions):
    generator = ObservationGenerator(CFG)
    rates = generator(
        torch.randint(0, CFG.n_realizations, (N_EPISODES, CFG.n_contexts)),
        torch.rand(N_EPISODES, CFG.n_contexts),
        interactions,
    )
    assert rates.shape == (N_EPISODES, CFG.n_observations)
    assert ((rates > 0) & (rates < 1)).all()


def test_generator_rejects_a_mismatched_confidence(interactions):
    generator = ObservationGenerator(CFG)
    with pytest.raises(ValueError, match="confidence must have shape"):
        generator(
            torch.zeros(N_EPISODES, CFG.n_contexts, dtype=torch.long),
            torch.rand(N_EPISODES, CFG.n_contexts + 1),
            interactions,
        )


def test_query_needs_a_single_time_slice():
    belief = torch.rand(N_EPISODES, N_STEPS, CFG.n_contexts, CFG.n_realizations)
    with pytest.raises(ValueError, match="index a single time step"):
        query_from_belief(belief, torch.zeros(N_EPISODES, dtype=torch.long))


def test_teaching_signal_is_all_or_nothing():
    belief = torch.softmax(torch.randn(N_EPISODES, CFG.n_contexts, 4), -1)
    with pytest.raises(ValueError, match="go together"):
        query_from_belief(
            belief,
            torch.zeros(N_EPISODES, dtype=torch.long),
            goal_selection=torch.zeros(N_EPISODES, dtype=torch.long),
        )


def test_query_confidence_is_the_probability_of_what_was_sampled():
    belief = torch.softmax(torch.randn(N_EPISODES, CFG.n_contexts, 4), -1)
    realizations, confidence = query_from_belief(
        belief, torch.zeros(N_EPISODES, dtype=torch.long)
    )
    expected = belief.gather(-1, realizations.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(confidence, expected)


def test_teaching_touches_only_the_goal_variable():
    belief = torch.softmax(torch.randn(N_EPISODES, CFG.n_contexts, 4), -1)
    goal_index = torch.randint(0, CFG.n_contexts, (N_EPISODES,))
    generator = torch.Generator().manual_seed(3)
    untaught, _ = query_from_belief(belief, goal_index, generator=generator)
    taught, _ = query_from_belief(
        belief,
        goal_index,
        goal_selection=torch.full((N_EPISODES,), 3),
        goal_correct=torch.ones(N_EPISODES),
        generator=torch.Generator().manual_seed(3),
    )
    for episode in range(N_EPISODES):
        goal = int(goal_index[episode])
        others = [c for c in range(CFG.n_contexts) if c != goal]
        assert taught[episode, goal] == 3
        assert all(taught[episode, c] == untaught[episode, c] for c in others)


# --------------------------------------------------------------------------- #
# stage 4
# --------------------------------------------------------------------------- #
def test_policy_is_a_distribution_over_the_joint_grid(interactions):
    controller = Controller(CFG)
    policy, value = controller(interactions)
    assert policy.shape == (N_EPISODES, *CFG.realization_shape)
    assert torch.allclose(policy.flatten(1).sum(-1), torch.ones(N_EPISODES), atol=1e-5)
    assert value.shape == (N_EPISODES,)
    assert ((value > 0) & (value < 1)).all()


def test_act_returns_one_realization_per_variable(interactions):
    controller = Controller(CFG)
    policy, _ = controller(interactions)
    actions, log_prob, entropy = controller.act(
        policy, torch.Generator().manual_seed(0)
    )
    assert actions.shape == (N_EPISODES, CFG.n_contexts)
    assert (actions >= 0).all() and (actions < CFG.n_realizations).all()
    assert log_prob.shape == entropy.shape == (N_EPISODES,)
    assert (log_prob <= 0).all() and (entropy >= 0).all()


def test_act_log_prob_is_the_policy_at_the_action_taken(interactions):
    controller = Controller(CFG)
    policy, _ = controller(interactions)
    actions, log_prob, _ = controller.act(policy, torch.Generator().manual_seed(0))
    for episode in range(N_EPISODES):
        taken = policy[(episode, *actions[episode].tolist())]
        assert torch.allclose(log_prob[episode], taken.log(), atol=1e-5)


def test_best_finds_the_grid_maximum(interactions):
    controller = Controller(CFG)
    policy, _ = controller(interactions)
    best = controller.best(policy)
    for episode in range(N_EPISODES):
        assert torch.allclose(
            policy[(episode, *best[episode].tolist())], policy[episode].max()
        )


def test_controller_rejects_a_mismatched_interaction_width():
    controller = Controller(CFG)
    with pytest.raises(ValueError, match="interactions must have shape"):
        controller(torch.randn(N_EPISODES, CFG.n_observations + 2, CFG.n_interactions))


def test_intrinsic_value_is_maximized_by_matching_the_preferences():
    preferences = torch.tensor([1.0, 0.0, 1.0])
    matched = torch.tensor([[0.99, 0.01, 0.99]])
    opposed = torch.tensor([[0.01, 0.99, 0.01]])
    assert intrinsic_value(matched, preferences) > intrinsic_value(opposed, preferences)


def test_intrinsic_value_punishes_any_single_miss():
    """A geometric mean means one missed channel cannot be bought back."""
    preferences = torch.tensor([1.0, 1.0, 1.0])
    all_good = intrinsic_value(torch.tensor([[0.9, 0.9, 0.9]]), preferences)
    one_bad = intrinsic_value(torch.tensor([[0.9, 0.9, 0.01]]), preferences)
    two_good_one_perfect = intrinsic_value(
        torch.tensor([[0.999, 0.999, 0.01]]), preferences
    )
    assert one_bad < all_good
    assert two_good_one_perfect < all_good


def test_intrinsic_value_rejects_a_mismatched_preference_vector():
    with pytest.raises(ValueError, match="preferences must cover"):
        intrinsic_value(torch.rand(2, 3), torch.rand(4))
