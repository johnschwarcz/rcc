"""Randomness helpers, so that seeding a chain seeds all of it.

PyTorch has no per-module generator for ``nn.Linear`` and friends — they draw
from the global RNG. So the only way to make initialization reproducible is to
seed that RNG, and the only way to do it politely is to put it back afterwards.

Sampling is the mirror problem: ``torch.distributions`` will not accept a
generator, so anything that needs reproducible draws has to reach for
``torch.multinomial`` instead.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch
from torch import Tensor

__all__ = ["sample_categorical", "seeded"]


@contextmanager
def seeded(seed: int | None) -> Iterator[None]:
    """Seed the global CPU RNG for the duration of the block, then restore it.

    ``None`` is a no-op, so a config with no seed leaves global randomness
    exactly as it found it.

    Examples
    --------
    Inside the block, the stream is determined by the seed:

    >>> with seeded(0):
    ...     first = torch.randn(3)
    >>> with seeded(0):
    ...     second = torch.randn(3)
    >>> bool(torch.equal(first, second))
    True

    Outside it, the caller's own stream is untouched:

    >>> torch.manual_seed(123)  # doctest: +ELLIPSIS
    <torch...>
    >>> before = torch.randn(3)
    >>> torch.manual_seed(123)  # doctest: +ELLIPSIS
    <torch...>
    >>> with seeded(7):
    ...     _ = torch.randn(100)
    >>> bool(torch.equal(before, torch.randn(3)))
    True
    """
    if seed is None:
        yield
        return
    state = torch.random.get_rng_state()
    try:
        torch.manual_seed(seed)
        yield
    finally:
        torch.random.set_rng_state(state)


def sample_categorical(
    probs: Tensor, generator: torch.Generator | None = None
) -> Tensor:
    """Draw one index per distribution along the last axis.

    Equivalent to ``Categorical(probs=probs).sample()``, except that it takes a
    generator, which is what makes a sampled chain testable.

    Parameters
    ----------
    probs:
        ``(..., n_categories)``, non-negative, normalized over the last axis.
    generator:
        Draws from the global RNG when ``None``.

    Returns
    -------
    Tensor
        Long tensor shaped like ``probs`` without its last axis.

    Examples
    --------
    >>> probs = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    >>> sample_categorical(probs).tolist()
    [0, 2]

    The batch shape is preserved:

    >>> sample_categorical(torch.ones(4, 5, 3) / 3).shape
    torch.Size([4, 5])
    """
    flat = probs.reshape(-1, probs.shape[-1])
    drawn = torch.multinomial(flat, 1, generator=generator)
    return drawn.reshape(probs.shape[:-1])
