"""Internal helpers: seeding, sampling, shape validation.
nn.Linear and friends draw from the global RNG, so reproducible initialization
means seeding it and putting it back. torch.distributions will not take a
generator, so reproducible sampling goes through torch.multinomial.
"""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
import torch
from torch import Tensor

__all__ = ["require_shape", "sample_categorical", "seeded"]


@contextmanager
def seeded(seed: int | None) -> Iterator[None]:
    """Seed the CPU RNG for the block, then put it back. None is a no-op.
    Parameters are initialized on CPU, so the CPU generator is the only one to touch.

    >>> with seeded(0):
    ...     first = torch.randn(3)
    >>> _ = torch.manual_seed(123)
    >>> before = torch.randn(3)
    >>> _ = torch.manual_seed(123)
    >>> with seeded(0):
    ...     second = torch.randn(3)
    >>> bool(torch.equal(first, second)), bool(torch.equal(before, torch.randn(3)))
    (True, True)
    """
    if seed is None:
        yield
        return
    state = torch.random.get_rng_state()
    try:
        # Seed exactly what gets restored. ``torch.manual_seed`` would also reseed
        # every CUDA device, which ``set_rng_state`` does not put back — so merely
        # building a seeded chain would reset a caller's GPU stream for good.
        torch.default_generator.manual_seed(seed)
        yield
    finally:
        torch.random.set_rng_state(state)


def sample_categorical(
    probs: Tensor, generator: torch.Generator | None = None
) -> Tensor:
    """Draw one index per distribution along the last axis.
    Categorical(probs=probs).sample(), except that it takes a generator.
    probs: (..., n_categories), non-negative, normalized over the last axis
    returns: long tensor shaped like probs without its last axis

    >>> probs = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    >>> sample_categorical(probs).tolist()
    [0, 2]
    >>> sample_categorical(torch.ones(4, 5, 3) / 3).shape
    torch.Size([4, 5])
    """
    flat = probs.reshape(-1, probs.shape[-1])
    drawn = torch.multinomial(flat, 1, generator=generator)
    return drawn.reshape(probs.shape[:-1])


def require_shape(name: str, tensor: Tensor, expected: Sequence[int]) -> None:
    """Raise ValueError unless tensor has exactly the expected shape.

    >>> require_shape("belief", torch.zeros(2, 3), (2, 3))
    >>> require_shape("belief", torch.zeros(2, 3), (2, 4))
    Traceback (most recent call last):
        ...
    ValueError: belief must have shape (2, 4), got (2, 3)
    """
    actual, wanted = tuple(tensor.shape), tuple(expected)
    if actual != wanted:
        raise ValueError(f"{name} must have shape {wanted}, got {actual}")
