"""Shared plumbing for the scripts — arguments, the world, numpy to torch, scoring.

None of this is the architecture. It lives here so that each script reads as the
run it exists to show, and nothing else.

`coggrid <https://github.com/johnschwarcz/coggrid>`_ is a development dependency,
never a runtime one: nothing in ``src/rcc`` imports it, and nothing should. The
examples need a real environment to run against, which is what it is for.
"""

import argparse
from pathlib import Path
from typing import Any, Literal, NamedTuple
import torch
from torch import Tensor
from rcc import RCC, RCCConfig, intrinsic_value, select_goal

try:
    from coggrid import CogGridConfig, World, run_observers
except ImportError:  # pragma: no cover - exercised only without the extra
    raise SystemExit(
        "The examples need coggrid:\n"
        "    pip install git+https://github.com/johnschwarcz/coggrid"
    ) from None

Split = Literal["train", "held_out"]

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

#: What :func:`arguments` collects that describes the task or the architecture.
_SETTINGS = (*SHAPE, "n_steps", "hidden_dim", "learn_embeddings", "seed")
#: Everything else it collects — how long to run and where to put the output.
_RUN = ("out", "iterations", "batch_size", "lr", "control_iterations", "control_lr")


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
    # Not --n-steps: that counts observations inside one episode, and "step" means
    # the latter everywhere in this repo, as "update" means a belief update.
    run.add_argument(
        "--iterations", type=int, default=5000, help="training iterations"
    )
    run.add_argument(
        "--batch-size", type=int, default=128, help="episodes drawn per iteration"
    )
    run.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    run.add_argument(
        "--seed", type=int, default=None,
        help="pin the run; the default explores fresh randomness",
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
) -> tuple[World, RCCConfig]:
    """A coggrid world and a chain config that agree on every shared field.

    Pass the parsed ``args`` to take every setting from the command line, plus
    keyword ``overrides`` for what a caller fixes about itself — a test with no
    argument parsing supplies all of them that way instead.

    Names are routed by destination: anything in :data:`SHAPE` reaches *both*
    configs, because the environment and the chain have to be describing the same
    task; ``learn_embeddings`` is rcc's alone.
    """
    if args is not None:
        overrides = {**{name: getattr(args, name) for name in _SETTINGS}, **overrides}
    shape = {**SHAPE, **{k: v for k, v in overrides.items() if k in SHAPE}}
    rcc_only = {k: v for k, v in overrides.items() if k not in SHAPE}
    seed = rcc_only.pop("seed", None)
    n_steps = rcc_only.pop("n_steps", N_STEPS)
    hidden_dim = rcc_only.pop("hidden_dim", HIDDEN_DIM)
    # A third of the pool held out, so a "novel pair" comes from more than one
    # variable; coggrid's default of n_vars // 10 would leave a single one, and a
    # pair drawn from a pool of one is the same variable twice.
    world = World(
        CogGridConfig(
            **shape,
            n_steps=n_steps,
            n_held_out_vars=max(2, shape["n_vars"] // 3),
            seed=seed,
        )
    )
    return world, RCCConfig(**shape, hidden_dim=hidden_dim, seed=seed, **rcc_only)


def embeddings(world: World) -> dict[str, Tensor]:
    """The world's true embeddings, as ``RCC(cfg, **embeddings(world))``.

    Handing these over is what isolates stage 2: the representation stage 1 would
    otherwise have to find is simply given.
    """
    return {
        "keys": torch.as_tensor(world.keys, dtype=torch.float32),
        "queries": torch.as_tensor(world.queries, dtype=torch.float32),
    }


def draw(
    world: World, n_episodes: int, *, split: Split = "train", rng: Any = None
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
        return torch.as_tensor(array, dtype=torch.float32)

    def as_long(array: Any) -> Tensor:
        return torch.as_tensor(array, dtype=torch.long)

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


def random_preferences(cfg: RCCConfig, seed: int = 0) -> Tensor:
    """Which observation channels the chain wants to see on, as a 0/1 vector.

    Pinned by default, and by a generator of its own: stage 4's figure is about a
    policy finding the best joint realization, which only means something against a
    fixed set of preferences.
    """
    generator = torch.Generator().manual_seed(seed)
    return (torch.rand(cfg.n_observations, generator=generator) > 0.5).float()


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
