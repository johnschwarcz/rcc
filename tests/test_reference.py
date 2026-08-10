"""The rewrite has to compute what the original computed.

`CognitiveGridworld <https://github.com/johnschwarcz/CognitiveGridworld>`_ ships
the architecture as five ``nn.Module`` mixins that pass state through instance
attributes. This package computes the same functions with explicit arguments,
which is a rewrite, and a rewrite of a research result is worth nothing if it
quietly changed the result.

So the original's arithmetic is transcribed below, verbatim in structure — the
loops, the reshapes, the argument order — and every stage is checked against it
with shared weights. When these tests pass, the two implementations are the same
function of the same inputs. Where behaviour *was* deliberately changed, there is
a test here saying so.
"""

import pytest
import torch
from rcc import (
    BeliefClassifier,
    Controller,
    InteractionEncoder,
    ObservationGenerator,
    RCCConfig,
    intrinsic_value,
)
from rcc.losses import (
    controller_loss,
    embedding_norm_penalty,
    prediction_loss,
    reward_loss,
    soft_clip,
    supervised_loss,
)

CFG = RCCConfig(
    n_vars=17,
    n_contexts=2,
    n_realizations=4,
    n_observations=3,
    embedding_dim=6,
    hidden_dim=12,
    learn_embeddings=False,
    seed=0,
)


@pytest.fixture
def batch():
    """Fixed inputs. Small, and pinned so a failure is always reproducible."""
    torch.manual_seed(0)
    n_episodes, n_steps = 5, 7
    return {
        "observations": torch.rand(n_episodes, n_steps, CFG.n_observations).round(),
        "ctx_inds": torch.randint(0, CFG.n_vars, (n_episodes, CFG.n_contexts)),
        "keys": torch.randn(CFG.n_vars, CFG.n_observations, CFG.embedding_dim),
        "queries": torch.randn(CFG.n_vars, CFG.n_observations, CFG.embedding_dim),
    }


# --------------------------------------------------------------------------- #
# stage 1 — Model_forward.default_interactions / KQ_to_Z
# --------------------------------------------------------------------------- #
def original_interactions(all_K, all_Q, ctx_inds, n_contexts, z_dims):
    """Transcribed from ``Model_forward.default_interactions``."""
    active_K = all_K[ctx_inds, :]
    active_Q = all_Q[ctx_inds, :]
    active_Z = torch.zeros(z_dims)
    count = 0

    def KQ_to_Z(ctx_1, ctx_2, count):
        K = active_K[:, ctx_1]
        Q = active_Q[:, ctx_2]
        active_Z[:, :, count] = (K * Q).sum(-1)
        return count + 1

    if n_contexts == 1:
        count = KQ_to_Z(0, 0, count)
    else:
        for k in range(n_contexts):
            for q in range(k + 1, n_contexts):
                count = KQ_to_Z(k, q, count)
                count = KQ_to_Z(q, k, count)
    return active_Z


@pytest.mark.parametrize("n_contexts", [1, 2, 3])
def test_interactions_match(batch, n_contexts):
    cfg = CFG.replace(n_contexts=n_contexts)
    keys, queries = batch["keys"], batch["queries"]
    ctx_inds = torch.randint(0, cfg.n_vars, (5, n_contexts))

    encoder = InteractionEncoder(cfg, keys=keys, queries=queries)
    mine = encoder(ctx_inds).score

    theirs = original_interactions(
        keys, queries, ctx_inds, n_contexts, (5, cfg.n_observations, cfg.n_interactions)
    )
    assert torch.allclose(mine, theirs, atol=1e-6)


# --------------------------------------------------------------------------- #
# stage 2 — Model_forward.RNN_classification / FF_classification
# --------------------------------------------------------------------------- #
def original_rnn_classification(m, obs_flat, active_Z, cfg):
    """Transcribed from ``RNN_readin`` + ``RNN_classification`` + ``RNN_readout``."""
    n_episodes, n_steps, _ = obs_flat.shape
    Z = active_Z.detach().reshape(n_episodes, 1, -1).expand(-1, n_steps, -1)
    inp = torch.relu(m.readin(torch.cat((obs_flat, Z), dim=-1)))

    stm = m.initial_short_term.expand(-1, n_episodes, -1).contiguous()
    ltm = m.initial_long_term.expand(-1, n_episodes, -1).contiguous()
    update, _ = m.rnn(inp, (stm, ltm))

    update = m.readout(update)
    update = update.reshape(n_episodes, n_steps, cfg.n_contexts, cfg.n_realizations)
    return torch.softmax(update.cumsum(1), -1)


def test_classifier_matches(batch):
    classifier = BeliefClassifier(CFG)
    encoder = InteractionEncoder(CFG, keys=batch["keys"], queries=batch["queries"])
    interactions = encoder(batch["ctx_inds"]).score

    mine = classifier(batch["observations"], interactions)
    theirs = original_rnn_classification(
        classifier, batch["observations"], interactions, CFG)
    assert torch.allclose(mine, theirs, atol=1e-6)


def test_belief_is_the_softmax_of_a_running_sum(batch):
    """The architectural claim, checked directly rather than through the layers.

    Undoing the softmax has to leave a sequence whose increments are what the
    readout emitted — that is what makes the accumulation multiplicative.
    """
    classifier = BeliefClassifier(CFG)
    encoder = InteractionEncoder(CFG, keys=batch["keys"], queries=batch["queries"])
    interactions = encoder(batch["ctx_inds"]).score
    belief = classifier(batch["observations"], interactions)

    # log belief recovers the cumulative logits up to a per-step constant, so
    # successive differences of the log belief are the raw increments, also up
    # to a constant. Differencing twice across realizations removes it.
    log_belief = belief.log()
    steps = log_belief[:, 1:] - log_belief[:, :-1]
    centred = steps - steps.mean(-1, keepdim=True)

    per_step = classifier.readout(
        classifier.rnn(
            torch.relu(
                classifier.readin(
                    torch.cat(
                        (
                            batch["observations"],
                            interactions.reshape(5, 1, -1).expand(-1, 7, -1),
                        ),
                        -1,
                    )
                )
            ),
            (
                classifier.initial_short_term.expand(-1, 5, -1).contiguous(),
                classifier.initial_long_term.expand(-1, 5, -1).contiguous(),
            ),
        )[0]
    ).reshape(5, 7, CFG.n_contexts, CFG.n_realizations)[:, 1:]
    assert torch.allclose(centred, per_step - per_step.mean(-1, keepdim=True), atol=1e-5)


# --------------------------------------------------------------------------- #
# stage 3 — Model_forward.get_prediction
# --------------------------------------------------------------------------- #
def original_get_prediction(m, sample, conf, active_Z):
    """Transcribed from ``Model_forward.get_prediction``."""
    n_episodes = sample.shape[0]
    sample_emb = m.realization_embedding(sample).reshape(n_episodes, -1)
    s = m.realization_proj(sample_emb).unsqueeze(1)
    c = m.confidence_proj(conf).unsqueeze(1)
    z = m.interaction_proj(active_Z)
    x = torch.relu(s + c + z)
    x = m.hidden(x)
    x = m.rate(torch.relu(x))
    return torch.sigmoid(x).squeeze()


def test_generator_matches(batch):
    generator = ObservationGenerator(CFG)
    encoder = InteractionEncoder(CFG, keys=batch["keys"], queries=batch["queries"])
    interactions = encoder(batch["ctx_inds"]).score
    ctx_vals = torch.randint(0, CFG.n_realizations, (5, CFG.n_contexts))
    confidence = torch.rand(5, CFG.n_contexts)

    mine = generator(ctx_vals, confidence, interactions)
    theirs = original_get_prediction(generator, ctx_vals, confidence, interactions)
    assert torch.allclose(mine, theirs, atol=1e-6)


def test_generator_keeps_a_single_channel_axis():
    """A deliberate change: ``.squeeze()`` became ``.squeeze(-1)``.

    The original collapsed every length-one axis, so one observation channel
    (or one episode) silently lost a dimension. Shape now depends only on the
    inputs' meaning, never on their size.
    """
    cfg = CFG.replace(n_observations=1)
    generator = ObservationGenerator(cfg)
    rates = generator(
        torch.zeros(4, cfg.n_contexts, dtype=torch.long),
        torch.rand(4, cfg.n_contexts),
        torch.randn(4, 1, cfg.n_interactions),
    )
    assert rates.shape == (4, 1)


# --------------------------------------------------------------------------- #
# stage 4 — Model_controller.controller_forward
# --------------------------------------------------------------------------- #
def original_controller_forward(m, active_Z, cfg):
    """Transcribed from ``Model_controller.controller_forward``."""
    n_episodes = active_Z.shape[0]
    z = active_Z.reshape(n_episodes, -1).detach()
    v = torch.relu(m.critic[2](torch.relu(m.critic[0](z))))
    predicted = torch.sigmoid(m.critic[4](v)).squeeze()
    a = m.actor[4](torch.relu(m.actor[2](torch.relu(m.actor[0](z)))))
    policy = torch.softmax(a, dim=-1).reshape(n_episodes, *cfg.realization_shape)
    return policy, predicted


def test_controller_matches(batch):
    controller = Controller(CFG)
    encoder = InteractionEncoder(CFG, keys=batch["keys"], queries=batch["queries"])
    interactions = encoder(batch["ctx_inds"]).score

    policy, value = controller(interactions)
    their_policy, their_value = original_controller_forward(
        controller, interactions, CFG
    )
    assert torch.allclose(policy, their_policy, atol=1e-6)
    assert torch.allclose(value, their_value, atol=1e-6)


def test_intrinsic_value_matches():
    """Transcribed from ``Model_controller.evaluate_control``."""
    rates = torch.rand(6, CFG.n_observations)
    preferences = (torch.rand(CFG.n_observations) > 0.5).float()

    O = rates.clip(1e-6, 1 - 1e-6)  # noqa: E741 — the original's name
    theirs = (
        (O.log() * preferences + (1 - O).log() * (1 - preferences)).sum(1)
        / CFG.n_observations
    ).exp()
    assert torch.allclose(intrinsic_value(rates, preferences), theirs, atol=1e-6)


# --------------------------------------------------------------------------- #
# objectives — Model_backward
# --------------------------------------------------------------------------- #
def original_soft_clip(x, eps=1e-3):
    """Transcribed from ``Model_backward.soft_clip``."""
    ciel = float(1) - eps
    x = ciel - (1 / (1 + torch.exp(-(ciel - x) / eps))) * (ciel - x)
    return eps + (1 / (1 + torch.exp(-(x - eps) / eps))) * (x - eps)


def original_DKL(x, y):
    return original_soft_clip(x) * torch.log(
        original_soft_clip(x) / original_soft_clip(y)
    )


def original_DKL_sym(x, y, PM=True):
    """Transcribed from ``Model_backward.DKL_sym``."""
    forward = original_DKL(x, y) + PM * original_DKL(1 - x, 1 - y)
    backward = original_DKL(y, x) + PM * original_DKL(1 - y, 1 - x)
    return (forward + backward) / 2


def test_soft_clip_matches():
    x = torch.linspace(-2, 3, 200)
    assert torch.allclose(soft_clip(x), original_soft_clip(x), atol=1e-6)


def test_supervised_loss_matches():
    """Transcribed from ``Model_backward.SANITY_loss``."""
    P = torch.softmax(torch.randn(5, 7, CFG.n_realizations), -1)
    Q = torch.softmax(torch.randn(5, 7, CFG.n_realizations), -1)

    DKL = original_DKL_sym(Q, P, PM=False)
    theirs = ((DKL - DKL.detach().min() + 1e-8) ** 0.5).mean()
    assert torch.allclose(supervised_loss(Q, P), theirs, atol=1e-6)


def test_reward_loss_matches():
    """Transcribed from ``Model_backward.RL_loss``."""
    n_episodes, n_steps = 5, 7
    goal_belief = torch.softmax(torch.randn(n_episodes, n_steps, 4), -1)
    selection = torch.randint(0, 4, (n_episodes,))
    correct = (torch.rand(n_episodes) > 0.5).float()
    bonus = 0.1

    CGS = selection[:, None].repeat(1, n_steps)
    BR = torch.arange(n_episodes)[:, None]
    SR = torch.arange(n_steps)[None, :]
    belief = original_soft_clip(goal_belief[BR, SR, CGS])
    acc = correct[:, None]
    rew = acc * -belief.log()
    pun = (1 - acc) * -(1 - belief).log()
    ent = -belief * belief.log() * bonus
    theirs = (rew + pun - ent).mean()

    mine = reward_loss(goal_belief, selection, correct, entropy_bonus=bonus)
    assert torch.allclose(mine, theirs, atol=1e-6)


def test_prediction_loss_matches():
    """Transcribed from ``Model_backward.SSL_loss``."""
    predicted = torch.rand(5, CFG.n_observations)
    observed = torch.rand(5, CFG.n_observations)
    # At least one correct episode, or the original divides by zero.
    correct = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0])
    chance = 1 / CFG.n_realizations

    OPE = original_DKL_sym(predicted, observed)
    OPE__ACC = (correct[:, None] * OPE).sum() / correct[:, None].sum()
    theirs = OPE.mean() * chance + (1 - chance) * OPE__ACC

    mine = prediction_loss(predicted, observed, correct=correct, chance=chance)
    assert torch.allclose(mine, theirs, atol=1e-6)


def test_prediction_loss_survives_a_batch_with_nothing_correct():
    """A deliberate change: the original divided by zero here.

    Early in training every episode is wrong, which made the embedding gradient
    NaN on exactly the batches where it mattered most.
    """
    predicted, observed = torch.rand(5, 3), torch.rand(5, 3)
    loss = prediction_loss(predicted, observed, correct=torch.zeros(5), chance=0.25)
    assert torch.isfinite(loss)


def test_embedding_norm_penalty_matches(batch):
    cfg = CFG.replace(learn_embeddings=True)
    encoder = InteractionEncoder(cfg)
    interaction = encoder(batch["ctx_inds"])

    K_norm = (torch.norm(interaction.keys, dim=-1) - 1) ** 2
    Q_norm = (torch.norm(interaction.queries, dim=-1) - 1) ** 2
    theirs = (K_norm + Q_norm).mean()

    mine = embedding_norm_penalty(interaction.keys, interaction.queries)
    assert torch.allclose(mine, theirs, atol=1e-6)


def test_controller_loss_matches():
    """Transcribed from ``Model_controller.update_controller``."""
    value = torch.rand(6)
    predicted = torch.rand(6, requires_grad=True)
    log_prob = -torch.rand(6)
    entropy = torch.rand(6)
    bonus = 0.05

    CPE = value - predicted
    theirs = (
        (-log_prob * CPE.detach()).mean() + (CPE**2).mean() - bonus * entropy.mean()
    )
    mine = controller_loss(value, predicted, log_prob, entropy, entropy_bonus=bonus)
    assert torch.allclose(mine, theirs, atol=1e-6)
