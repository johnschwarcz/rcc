"""The Trainer, and the one thing it exists to spare every caller.

A run is stated once, in the config. If the Trainer did not read it there, every
script would restate it — and a script that restates a setting is a script that can
disagree with the config about it, silently, which is exactly what happened to
``examples/quickstart.py`` before these tests existed.
"""

import pytest
import torch
from rcc import RCC, RCCConfig, Trainer

CFG = RCCConfig(
    n_vars=13, n_contexts=2, n_realizations=4, n_observations=3,
    embedding_dim=5, hidden_dim=16, seed=0,
)
N_EPISODES, N_STEPS = 6, 8


def batch(device=None, n_episodes=N_EPISODES):
    """One batch of the right shapes; the values do not matter to these tests."""
    generator = torch.Generator().manual_seed(0)
    return dict(
        observations=torch.rand(n_episodes, N_STEPS, CFG.n_observations,
                                generator=generator).round().to(device),
        ctx_inds=torch.randint(0, CFG.n_vars, (n_episodes, CFG.n_contexts),
                               generator=generator).to(device),
        goal_ind=torch.zeros(n_episodes, dtype=torch.long, device=device),
        goal_value=torch.zeros(n_episodes, dtype=torch.long, device=device),
    )


def rates(trainer):
    return {name: getattr(trainer, name).param_groups[0]["lr"]
            for name in ("inference", "estimation", "control")}


# --------------------------------------------------------------------------- #
# everything comes off the config
# --------------------------------------------------------------------------- #
def test_every_rate_is_read_from_the_config():
    cfg = CFG.replace(classifier_lr=1e-4, generator_lr=5e-4, control_lr=2e-2)
    assert rates(Trainer(RCC(cfg))) == {
        "inference": 1e-4, "estimation": 5e-4, "control": 2e-2}


def test_the_entropy_bonuses_and_micro_batch_come_off_the_config():
    cfg = CFG.replace(classifier_entropy_bonus=0.4, controller_entropy_bonus=0.0,
                      micro_batch=3)
    trainer = Trainer(RCC(cfg))
    assert trainer.classifier_entropy_bonus == 0.4
    assert trainer.controller_entropy_bonus == 0.0
    assert trainer.micro_batch == 3


def test_the_estimation_rate_follows_the_classifier_when_nothing_pins_it():
    assert rates(Trainer(RCC(CFG.replace(classifier_lr=7e-4))))["estimation"] == 7e-4


def test_an_argument_overrides_the_config():
    """The one case a config cannot cover: sweeping a value over equal chains."""
    trainer = Trainer(RCC(CFG.replace(classifier_lr=1e-4)), classifier_lr=1e-2)
    assert rates(trainer)["inference"] == 1e-2


def test_overriding_the_classifier_rate_carries_the_estimation_rate_with_it():
    trainer = Trainer(RCC(CFG.replace(classifier_lr=1e-4)), classifier_lr=1e-2)
    assert rates(trainer)["estimation"] == 1e-2


def test_a_config_that_pinned_the_generator_rate_is_not_overruled_by_accident():
    """Overriding one field must not silently move a field it was never given."""
    cfg = CFG.replace(classifier_lr=1e-4, generator_lr=5e-4)
    trainer = Trainer(RCC(cfg), classifier_lr=1e-2)
    assert rates(trainer) == {"inference": 1e-2, "estimation": 5e-4,
                              "control": cfg.control_lr}


def test_a_seeded_config_builds_the_generator_for_you():
    """Sampling has to be pinned by the same seed as initialization, on the same
    device, and only the Trainer knows both."""
    trainer = Trainer(RCC(CFG))
    assert trainer.generator is not None
    assert trainer.generator.device == RCC(CFG).device


def test_an_unseeded_config_leaves_the_sampling_free():
    assert Trainer(RCC(CFG.replace(seed=None))).generator is None


# --------------------------------------------------------------------------- #
# the steps
# --------------------------------------------------------------------------- #
def test_micro_batching_does_not_change_what_a_step_reports():
    """A memory knob, not a statistical one: accuracy is a plain mean over episodes,
    so slicing the batch has to leave it alone."""
    whole = Trainer(RCC(CFG)).step(**batch(n_episodes=8))
    sliced = Trainer(RCC(CFG.replace(micro_batch=2))).step(**batch(n_episodes=8))
    assert sliced.accuracy == pytest.approx(whole.accuracy)
    assert sliced.belief.shape == whole.belief.shape


def test_a_step_accepts_a_batch_from_anywhere():
    """Forgetting to draw onto the chain's device is a slower run, not a crash."""
    trainer = Trainer(RCC(CFG))
    assert 0.0 <= trainer.step(**batch(device="cpu")).accuracy <= 1.0


def test_control_step_accepts_a_landscape_from_anywhere():
    trainer = Trainer(RCC(CFG))
    ctx_inds = torch.randint(0, CFG.n_vars, (N_EPISODES, CFG.n_contexts))
    landscape = torch.rand(N_EPISODES, *CFG.realization_shape)
    control = trainer.control_step(ctx_inds, landscape)
    # The value taken is one the landscape actually holds, so it has to be inside it.
    assert landscape.min() <= control.value <= landscape.max()
    assert torch.isfinite(torch.tensor(control.loss))


# --------------------------------------------------------------------------- #
# where the chain ends up
# --------------------------------------------------------------------------- #
def test_a_chain_stays_on_the_cpu_unless_the_config_says_otherwise():
    """A library that reaches for a gpu nobody asked it to take is a surprise."""
    assert RCC(CFG).device == torch.device("cpu")
    assert RCC(CFG.replace(device="cpu")).device == torch.device("cpu")


def test_an_impossible_device_request_warns_and_falls_back():
    """A config may pin the machine its figures were made on and still run anywhere."""
    if torch.cuda.is_available():
        pytest.skip("the fallback under test is the one taken without cuda")
    with pytest.warns(RuntimeWarning, match="falling back to the cpu"):
        chain = RCC(CFG.replace(device="cuda:0"))
    assert chain.device == torch.device("cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a cuda device")
def test_a_chain_places_itself_on_the_configured_device():
    chain = RCC(CFG.replace(device="cuda:0"))
    assert chain.device.type == "cuda"
    assert Trainer(chain).generator.device.type == "cuda"
