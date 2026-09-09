"""The config is the only place a chain's shape is decided, so it validates."""

import dataclasses
import pytest
from rcc import RCCConfig


def test_is_immutable():
    cfg = RCCConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.n_contexts = 3


@pytest.mark.parametrize(
    "field", ["n_vars", "n_contexts", "n_realizations", "n_observations",
              "embedding_dim", "hidden_dim"]
)
@pytest.mark.parametrize("bad", [0, -1, 2.5, "4"])
def test_rejects_non_positive_ints(field, bad):
    with pytest.raises(ValueError, match=field):
        RCCConfig(**{field: bad})


def test_rejects_bool_as_a_size():
    """``True`` is an int in Python, and a hidden_dim of 1 is not what was meant."""
    with pytest.raises(ValueError, match="hidden_dim"):
        RCCConfig(hidden_dim=True)


@pytest.mark.parametrize(
    "n_contexts,expected", [(1, 1), (2, 2), (3, 6), (4, 12)]
)
def test_n_interactions(n_contexts, expected):
    assert RCCConfig(n_contexts=n_contexts).n_interactions == expected


def test_classifier_input_dim_covers_observations_and_every_interaction():
    cfg = RCCConfig(n_observations=5, n_contexts=3)
    assert cfg.classifier_input_dim == 5 + 5 * cfg.n_interactions


def test_replace_revalidates():
    with pytest.raises(ValueError, match="n_vars"):
        RCCConfig().replace(n_vars=0)


def test_replace_leaves_the_original_alone():
    cfg = RCCConfig(n_contexts=2)
    assert cfg.replace(n_contexts=3).n_contexts == 3
    assert cfg.n_contexts == 2


# --------------------------------------------------------------------------- #
# the training fields
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("field", ["classifier_lr", "control_lr", "generator_lr"])
@pytest.mark.parametrize("bad", [0, -1e-3, "3e-4", True])
def test_rejects_rates_that_are_not_positive_floats(field, bad):
    with pytest.raises(ValueError, match=field):
        RCCConfig(**{field: bad})


@pytest.mark.parametrize(
    "field", ["classifier_entropy_bonus", "controller_entropy_bonus"]
)
def test_an_entropy_bonus_of_zero_is_allowed(field):
    """Zero asks the objective for nothing but its own return, which is a choice."""
    assert getattr(RCCConfig(**{field: 0.0}), field) == 0.0


@pytest.mark.parametrize(
    "field", ["classifier_entropy_bonus", "controller_entropy_bonus"]
)
def test_rejects_a_negative_entropy_bonus(field):
    with pytest.raises(ValueError, match=field):
        RCCConfig(**{field: -0.1})


@pytest.mark.parametrize("bad", [0, -5, 2.5, True])
def test_rejects_a_micro_batch_that_is_not_a_positive_int(bad):
    with pytest.raises(ValueError, match="micro_batch"):
        RCCConfig(micro_batch=bad)


def test_micro_batch_of_none_means_one_pass():
    assert RCCConfig().micro_batch is None


def test_rejects_a_device_that_is_not_a_string():
    with pytest.raises(ValueError, match="device"):
        RCCConfig(device=0)


def test_estimation_lr_follows_the_classifier_until_it_is_pinned():
    assert RCCConfig(classifier_lr=1e-4).estimation_lr == 1e-4
    assert RCCConfig(classifier_lr=1e-4, generator_lr=5e-4).estimation_lr == 5e-4

