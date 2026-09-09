"""Shared plumbing for the scripts — arguments, the world, numpy to torch, scoring.

None of this is the architecture. It lives here so that each script reads as the
run it exists to show, and nothing else.

`coggrid <https://github.com/johnschwarcz/coggrid>`_ is a development dependency,
never a runtime one: nothing in ``src/rcc`` imports it, and nothing should. The
examples need a real environment to run against, which is what it is for.
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal, NamedTuple
import numpy as np
import torch
from torch import Tensor
from tqdm.auto import tqdm
# Nothing here resolves a device any more: that moved into rcc, where a chain
# places itself on cfg.device. Scripts pass chain.device to draw() and no more.
from rcc import RCC, RCCConfig, intrinsic_value, select_goal

try:
    from coggrid import CogGridConfig, World, run_observers
except ImportError:  # pragma: no cover - exercised only without the extra
    raise SystemExit(
        "The examples need coggrid:\n"
        "    pip install git+https://github.com/johnschwarcz/coggrid"
    ) from None

Split = Literal["train", "held_out"]

#: coggrid's two variable pools, named here so that no script has to spell them.
#: ``TRAIN`` allows at most one held-out variable per episode and never a held-out
#: goal; ``HELD_OUT`` draws every active variable from the held-out pool.
TRAIN: Split = "train"
HELD_OUT: Split = "held_out"

#: Default shape of the task, read by *both* packages — the field names are shared
#: deliberately, so the environment and the chain always describe the same task.
#: These are only the fallbacks: every one is a flag (``--n-realizations 8``) and
#: can be pinned per script (``arguments(n_realizations=8)``).
SHAPE = dict(
    n_vars=12, n_contexts=2, n_realizations=5, n_observations=5, embedding_dim=8
)


class Batch(NamedTuple):
    """One batch of episodes as tensors, with what the ideal observers made of it.

    ``posterior`` is the joint observer's belief: the target worth distilling and
    the ceiling a chain is reaching for. ``naive`` is the same inference done
    assuming the active variables are independent, so the gap between the two
    accuracies is the factorization regret the chain has to capture.
    """

    observations: Tensor  # (episodes, steps, channels)
    ctx_inds: Tensor  # (episodes, contexts)
    goal_ind: Tensor  # (episodes,)
    goal_value: Tensor  # (episodes,)
    posterior: Tensor  # (episodes, steps, contexts, realizations)
    rates: Tensor  # (episodes, channels, *realization_shape)
    ideal: float  # joint-observer accuracy at the final step
    naive: float  # factorized-observer accuracy at the final step


#: Defaults for the two shape fields coggrid and rcc do not share.
N_STEPS, HIDDEN_DIM = 20, 256

#: Task fields that reach the world alone — coggrid has them, RCCConfig does not.
_WORLD_EXTRA = ("n_steps", "n_held_out_vars", "likelihood_temp", "likelihood_freq")

#: What :func:`arguments` collects that ends up inside :class:`RCCConfig` — the
#: task, the architecture, and how the chain is trained. All of it reaches the
#: config, so ``RCC(cfg)`` and ``Trainer(chain)`` need nothing else spelled out.
_SETTINGS = (*SHAPE, *_WORLD_EXTRA, "hidden_dim", "learn_embeddings", "seed",
             "device", "micro_batch", "classifier_lr", "generator_lr", "control_lr",
             "classifier_entropy_bonus", "controller_entropy_bonus")
#: Everything else it collects — how many times to go round each loop, how often to
#: measure, and where to put the output. These belong to the script's loops rather
#: than to the chain, which is why they are the only things a script still reads.
_RUN = ("out", "checkpoint", "replot", "progress", "iterations", "batch_size",
        "control_iterations", "evaluate_every", "eval_episodes")


def arguments(**defaults: Any) -> argparse.Namespace:
    """Every run, task and architecture parameter, as a command-line flag.

    Nothing worth changing between tasks lives only in this file. A script pins
    whatever it cares about, and the command line still wins::

        args = arguments(iterations=2000, n_realizations=8)    # in the script
        python examples/quickstart.py --n-realizations 6   # beats it

    Task flags reach both the world and the chain, because the two have to be
    describing the same task; architecture flags are the chain's alone. Unknown
    *command-line* flags are ignored so a script can add its own, but an unknown
    default here is a typo and raises.
    """
    unknown = sorted(set(defaults) - set(_SETTINGS) - set(_RUN))
    if unknown:
        raise TypeError(f"arguments() got unexpected defaults: {unknown}. "
                        f"Settable here: {sorted((*_SETTINGS, *_RUN))}")

    parser = argparse.ArgumentParser(allow_abbrev=False)

    run = parser.add_argument_group("run")
    run.add_argument("--out", type=Path, default=None, help="save figures here")
    run.add_argument(
        "--checkpoint", type=Path, default=None,
        help="write the finished run here — the chain, its optimizers, and what the "
             "loop measured — so that re-plotting it never means retraining",
    )
    run.add_argument(
        "--replot", action="store_true",
        help="load --checkpoint and go straight to the figures, training nothing",
    )
    run.add_argument(
        "--progress", action=argparse.BooleanOptionalAction, default=None,
        help="show a progress bar; the default shows one on a terminal and logs "
             "lines instead when the output is redirected",
    )
    # Not --n-steps: that counts observations inside one episode, and "step" means
    # the latter everywhere in this repo, as "update" means a belief update.
    run.add_argument(
        "--iterations", type=int, default=5000, help="training iterations"
    )
    run.add_argument(
        "--batch-size", type=int, default=128, help="episodes drawn per iteration"
    )
    run.add_argument(
        "--device", default=None,
        help="torch device: 'auto' takes cuda when it is available, 'cuda:0' asks "
             "for one and falls back with a warning, and the default stays on the cpu",
    )
    run.add_argument(
        "--micro-batch", type=int, default=None,
        help="split each batch into slices of this many episodes, accumulating "
             "their gradients; a memory knob, leaving the batch what a step means",
    )
    run.add_argument("--classifier-lr", type=float, default=1e-3,
                     help="Adam learning rate for the inference objective (stage 2)")
    run.add_argument(
        "--generator-lr", type=float, default=None,
        help="Adam learning rate for the estimation objective, which is stages 1 "
             "and 3 together; defaults to --classifier-lr",
    )
    run.add_argument(
        "--evaluate-every", type=int, default=25,
        help="iterations between held-out measurements; each is a forward pass",
    )
    run.add_argument(
        "--eval-episodes", type=int, default=None,
        help="episodes per evaluation batch; defaults to --batch-size",
    )
    run.add_argument(
        "--seed", type=int, default=None,
        help="pin the run; the default explores fresh randomness",
    )

    # The weights that decide what each objective is actually asking for. They are
    # easy to leave at their defaults without noticing they were ever choices.
    objective = parser.add_argument_group("objectives (what each loss asks for)")
    objective.add_argument(
        "--classifier-entropy-bonus", type=float, default=0.1,
        help="reward_loss's entropy bonus",
    )
    objective.add_argument(
        "--controller-entropy-bonus", type=float, default=0.05,
        help="controller_loss's entropy bonus; constant, never decayed",
    )

    # ASCII in the group titles: a Windows console renders them in cp1252, where an
    # em dash becomes a replacement character.
    task = parser.add_argument_group("task (shared by the world and the chain)")
    for name, value in SHAPE.items():
        task.add_argument(f"--{name.replace('_', '-')}", type=int, default=value)
    task.add_argument(
        "--n-steps", type=int, default=N_STEPS,
        help="observations per episode (the trajectory the chain integrates)",
    )
    task.add_argument(
        "--n-held-out-vars", type=int, default=None,
        help="size of the held-out pool; defaults to a third of n_vars",
    )
    task.add_argument(
        "--likelihood-temp", type=float, default=2.0,
        help="scales the interaction potentials before the sigmoid; the paper's 2",
    )
    task.add_argument(
        "--likelihood-freq", type=float, default=1.0,
        help="periods in the sinusoidal value profile; >1 makes the map multimodal",
    )

    chain = parser.add_argument_group("architecture (the chain only)")
    chain.add_argument("--hidden-dim", type=int, default=HIDDEN_DIM)
    chain.add_argument(
        "--learn-embeddings", action=argparse.BooleanOptionalAction, default=False,
        help="stage 1 finds the representation; the default hands it the world's",
    )

    stage4 = parser.add_argument_group("control (stage 4, where a script trains one)")
    stage4.add_argument("--control-iterations", type=int, default=400)
    stage4.add_argument("--control-lr", type=float, default=3e-3)

    parser.set_defaults(**defaults)
    return parser.parse_known_args()[0]


def describe(args: argparse.Namespace, cfg: RCCConfig) -> str:
    """What this run actually resolved to, so a figure is never ambiguous.

    The second line spells out the three counts that are easy to conflate: how
    many times we train, how many episodes go into each of those, and how long an
    episode is.
    """
    return (
        f"{cfg}\n"
        f"{args.iterations} iterations x {args.batch_size} episodes per batch "
        f"x {args.n_steps} observations per episode"
    )


def world_and_config(
    args: argparse.Namespace | None = None, /, **overrides: Any
) -> tuple[World, RCCConfig, dict[str, Tensor]]:
    """A coggrid world, a chain config agreeing with it, and ``RCC(cfg, **given)``.

    Pass the parsed ``args`` to take every setting from the command line, plus
    keyword ``overrides`` for what a caller fixes about itself — a test with no
    argument parsing supplies all of them that way instead.

    Names are routed by destination: anything in :data:`SHAPE` reaches *both*
    configs, because the environment and the chain have to be describing the same
    task; ``learn_embeddings`` is rcc's alone.

    The third return is what the chain has to be *given* rather than told: the
    world's embeddings when stage 1 is not learning its own, and ``{}`` when it is.
    They stay out of the config because a config is the task's shape, hashable and
    comparable and small enough to print, while these are one world's data.
    """
    if args is not None:
        overrides = {**{name: getattr(args, name) for name in _SETTINGS}, **overrides}
    shape = {**SHAPE, **{k: v for k, v in overrides.items() if k in SHAPE}}
    rcc_only = {k: v for k, v in overrides.items() if k not in SHAPE}
    seed = rcc_only.pop("seed", None)
    hidden_dim = rcc_only.pop("hidden_dim", HIDDEN_DIM)
    world_extra = {k: rcc_only.pop(k) for k in _WORLD_EXTRA if k in rcc_only}
    world_extra.setdefault("n_steps", N_STEPS)
    # A third of the pool held out, so a "novel pair" comes from more than one
    # variable; coggrid's default of n_vars // 10 would leave a single one, and a
    # pair drawn from a pool of one is the same variable twice.
    if world_extra.get("n_held_out_vars") is None:
        world_extra["n_held_out_vars"] = max(2, shape["n_vars"] // 3)
    world = World(CogGridConfig(**shape, **world_extra, seed=seed))
    cfg = RCCConfig(**shape, hidden_dim=hidden_dim, seed=seed, **rcc_only)
    return world, cfg, embeddings(world, cfg)


def embeddings(world: World, cfg: RCCConfig | None = None) -> dict[str, Tensor]:
    """The world's true embeddings, as ``RCC(cfg, **embeddings(world, cfg))``.

    Handing these over is what isolates stage 2: the representation stage 1 would
    otherwise have to find is simply given. Passing ``cfg`` makes the call safe
    either way — a chain that learns its own embeddings must not be given any, so
    it gets ``{}`` and the caller needs no conditional.
    """
    if cfg is not None and cfg.learn_embeddings:
        return {}
    return {
        "keys": torch.as_tensor(world.keys, dtype=torch.float32),
        "queries": torch.as_tensor(world.queries, dtype=torch.float32),
    }


def save_world(world: World, path: str | Path) -> None:
    """Write what makes this world the world it is: its config and its embeddings.

    Everything else a world does is derived from those two, so this plus
    :meth:`rcc.RCC.load` reproduces a run without either side having been seeded.
    """
    np.savez(str(path), keys=world.keys, queries=world.queries,
             cfg=json.dumps(asdict(world.cfg)))


def load_world(path: str | Path) -> World:
    """Rebuild a world that :func:`save_world` wrote."""
    saved = np.load(str(path))
    keys, queries = saved["keys"], saved["queries"]
    return World(CogGridConfig(**json.loads(str(saved["cfg"].item()))),
                 embeddings=lambda cfg, rng: (keys, queries))


def draw(
    world: World, n_episodes: int, *, split: Split = "train", rng: Any = None,
    device: torch.device | str | None = None,
) -> Batch:
    """Sample episodes and run the ideal observers over them.

    Names carry across unchanged — ``ctx_inds``, ``goal_ind``, ``goal_value``,
    ``observations``, ``rates`` are coggrid's own — so this only converts numpy to
    torch and attaches what the ideal observers concluded. ``ctx_vals`` is left
    behind on purpose: it is the truth, and the chain never sees it.
    """
    batch = world.sample_episodes(n_episodes, split=split, rng=rng)
    traces = run_observers(batch)

    def as_float(array: Any) -> Tensor:
        return torch.as_tensor(array, dtype=torch.float32, device=device)

    def as_long(array: Any) -> Tensor:
        return torch.as_tensor(array, dtype=torch.long, device=device)

    return Batch(
        observations=as_float(batch.observations),
        ctx_inds=as_long(batch.ctx_inds),
        goal_ind=as_long(batch.goal_ind),
        goal_value=as_long(batch.goal_value),
        posterior=as_float(traces["joint"].belief),
        rates=as_float(batch.rates),
        ideal=float(traces["joint"].accuracy[:, -1].mean()),
        naive=float(traces["naive"].accuracy[:, -1].mean()),
    )


def _someone_is_watching() -> bool:
    """Whether a bar would be seen. A terminal, or an interactive console.

    An interactive console is the case tqdm's own guess gets wrong: it replaces
    stderr with a stream that is not a tty, so asking the stream alone turns the
    bar off in exactly the place a bar is most wanted.
    """
    if sys.stderr.isatty():
        return True
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    return get_ipython() is not None


def progress(total: int, desc: str, enabled: bool | None = None) -> tqdm:
    """A bar over one loop, whose postfix is where a script reports how it is doing.
    ``enabled`` None asks :func:`_someone_is_watching`; redirected output gets no
    bar, so a long run logs its own lines rather than a million redraws.
    ``--progress`` and ``--no-progress`` say so outright when the guess is wrong.
    """
    if enabled is None:
        enabled = _someone_is_watching()
    return tqdm(range(total), desc=desc, disable=not enabled, dynamic_ncols=True,
        leave=True,
        bar_format="{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                   "[{elapsed}<{remaining}]{postfix}")


def accuracy(goal_belief: Tensor, goal_value: Tensor) -> float:
    """Share of episodes whose final-step argmax matched the truth."""
    return (goal_belief[:, -1].argmax(-1) == goal_value).float().mean().item()


def evaluate(chain: RCC, batch: Batch) -> dict[str, float]:
    """The chain's accuracy on one batch, beside both ideal observers'.

    Takes a batch rather than drawing one so that a caller measuring repeatedly
    through training watches the same episodes every time.

    The joint observer knows the interactions and the naive one assumes the active
    variables are independent, so the gap between those two is the part of the task
    that can only be got right by having represented the interaction structure.
    """
    with torch.no_grad():
        belief, _ = chain(batch.observations, batch.ctx_inds)
    return {
        "chain": accuracy(select_goal(belief, batch.goal_ind), batch.goal_value),
        "ideal": batch.ideal,
        "naive": batch.naive,
    }


def random_preferences(cfg: RCCConfig, seed: int | None = None, *,
    device: torch.device | str | None = None) -> Tensor:
    """Which observation channels the chain wants to see on, as a 0/1 vector.

    Unseeded like everything else, and drawn from a generator of its own so that
    passing a ``seed`` pins the preferences without pinning anything upstream. They
    are drawn once per run either way, which is what stage 4's figure needs.
    """
    generator = None if seed is None else torch.Generator().manual_seed(seed)
    drawn = (torch.rand(cfg.n_observations, generator=generator) > 0.5).float()
    return drawn.to(device)


def value_landscape(rates: Tensor, preferences: Tensor) -> Tensor:
    """What every joint realization is worth, given the preferences.

    ``rates`` is coggrid's likelihood table, one Bernoulli rate per channel per
    hypothetical joint realization.

    rates: (n_episodes, n_observations, *realization_shape)
    preferences: (n_observations,)
    returns: (n_episodes, *realization_shape)
    """
    n_episodes, n_observations = rates.shape[:2]
    per_action = rates.reshape(n_episodes, n_observations, -1).transpose(1, 2)
    value = intrinsic_value(per_action.reshape(-1, n_observations), preferences)
    return value.reshape(n_episodes, *rates.shape[2:])
