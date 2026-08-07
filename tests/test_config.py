"""The config is the only place a chain's shape is decided, so it validates."""

from __future__ import annotations

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


def test_reservoir_needs_a_recurrence_to_freeze():
    with pytest.raises(ValueError, match="reservoir"):
        RCCConfig(recurrent=False, reservoir=True)


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
