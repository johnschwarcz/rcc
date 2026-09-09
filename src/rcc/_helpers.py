# Internal helpers: seeding, sampling, shape validation.
import warnings
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
import torch
from torch import Tensor

__all__ = ["require_shape", "resolve_device", "sample_categorical", "seeded"]

@contextmanager
def seeded(seed: int | None) -> Iterator[None]:
    if seed is None:
        yield
        return
    state = torch.random.get_rng_state()
    try:
        torch.default_generator.manual_seed(seed)
        yield
    finally:
        torch.random.set_rng_state(state) # Does not restore CUDA device seeds

def sample_categorical(probs: Tensor, generator: torch.Generator | None = None) -> Tensor:
    """Draw one index per distribution along the last axis.
    probs: (..., n_categories)
    returns: long tensor shaped like probs without its last axis
    """
    flat = probs.reshape(-1, probs.shape[-1])
    drawn = torch.multinomial(flat, 1, generator=generator)
    return drawn.reshape(probs.shape[:-1])

def require_shape(name: str, tensor: Tensor, expected: Sequence[int | None]) -> None:
    """Raise ValueError unless tensor matches expected. ``None`` is any size."""
    actual, wanted = tuple(tensor.shape), tuple(expected)
    if len(actual) != len(wanted) or any(
        w is not None and a != w for a, w in zip(actual, wanted, strict=True)
    ):
        shown = ", ".join("n" if w is None else str(w) for w in wanted)
        shown += "," if len(wanted) == 1 else ""   # match tuple repr: (4,) not (4)
        raise ValueError(f"{name} must have shape ({shown}), got {actual}")


def resolve_device(name: str | None = None) -> torch.device:
    """The device to run on, ``None`` -> cpu, ``"auto"`` -> cuda when available."""
    if name is None:
        return torch.device("cpu")
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type != "cuda":
        return device
    if not torch.cuda.is_available():
        warnings.warn(
            f"{name} was asked for, but no cuda device is available here; "
            f"falling back to the cpu.", RuntimeWarning, stacklevel=2,
        )
        return torch.device("cpu")
    visible = torch.cuda.device_count()
    if device.index is not None and device.index >= visible:
        warnings.warn(
            f"{name} was asked for, but only {visible} cuda device(s) are "
            f"visible; falling back to cuda:0.", RuntimeWarning, stacklevel=2,
        )
        return torch.device("cuda", 0)
    return device
