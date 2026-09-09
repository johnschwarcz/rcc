"""A finished run, on disk: the chain, its optimizers, and what it measured.
Everything saved goes through ``torch.load(..., weights_only=True)`` on the way back,
Everything must be tensors, numbers, strings, or lists and dicts of those. 
"""
from dataclasses import asdict
from pathlib import Path
from typing import Any, NamedTuple
import torch
from .chain import RCC
from .training import Trainer

__all__ = ["Run", "load_run", "save_run"]


class Run(NamedTuple):
    """What :func:`load_run` gives back. Unpacks as chain, trainer_state, artifacts."""

    chain: RCC
    trainer_state: dict[str, Any] | None
    artifacts: dict[str, Any]

    def trainer(self, **overrides: Any) -> Trainer:
        """A Trainer on this run's chain, with its optimizer moments restored."""
        trainer = Trainer(self.chain, **overrides)
        if self.trainer_state is not None:
            trainer.load_state_dict(self.trainer_state)
        return trainer


def save_run(path: str | Path, chain: RCC, *, trainer: Trainer | None = None, **artifacts: Any) -> None:
    """Write a chain, optionally its optimizers, and whatever the run measured."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cfg": asdict(chain.cfg),
        "state": chain.state_dict(),
        "trainer": None if trainer is None else trainer.state_dict(),
        "artifacts": artifacts,
    }
    temporary = path.with_name(path.name + ".partial")
    try:
        torch.save(payload, temporary)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    temporary.replace(path)


def load_run(path: str | Path, map_location: str | None = None) -> Run:
    """Read back what :func:`save_run` wrote."""
    saved = torch.load(path, map_location="cpu", weights_only=True)
    chain = RCC.from_saved(saved, map_location)
    return Run(chain, saved.get("trainer"), saved.get("artifacts", {}))
