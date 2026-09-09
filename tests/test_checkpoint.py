"""A finished run, written down and read back.

The figures a run produced were drawn from numbers that existed only inside the
loop. If a checkpoint holds the weights alone, re-plotting means retraining — so
what is tested here is that a run comes back whole, and that redrawing from it
gives the same answer as the run that wrote it.
"""

import pytest
import torch
from rcc import RCC, RCCConfig, Trainer, load_run, save_run

CFG = RCCConfig(
    n_vars=13, n_contexts=2, n_realizations=4, n_observations=3,
    embedding_dim=5, hidden_dim=16, seed=0, classifier_lr=1e-4, micro_batch=2,
)
N_EPISODES, N_STEPS = 6, 8


@pytest.fixture
def trained(tmp_path):
    """A chain that has taken a step, so its optimizers hold real moments."""
    chain = RCC(CFG)
    trainer = Trainer(chain)
    generator = torch.Generator().manual_seed(0)
    step = trainer.step(
        observations=torch.rand(N_EPISODES, N_STEPS, CFG.n_observations,
                                generator=generator).round(),
        ctx_inds=torch.randint(0, CFG.n_vars, (N_EPISODES, CFG.n_contexts),
                               generator=generator),
        goal_ind=torch.zeros(N_EPISODES, dtype=torch.long),
        goal_value=torch.zeros(N_EPISODES, dtype=torch.long),
    )
    return chain, trainer, step, tmp_path / "run.pt"


def test_the_chain_comes_back_parameter_for_parameter(trained):
    chain, trainer, _, path = trained
    save_run(path, chain, trainer=trainer)
    restored = load_run(path).chain
    for name, tensor in chain.state_dict().items():
        assert torch.equal(restored.state_dict()[name], tensor), name


def test_the_config_comes_back_whole(trained):
    """Including the training fields: what the run was is part of what it produced."""
    chain, trainer, _, path = trained
    save_run(path, chain, trainer=trainer)
    assert load_run(path).chain.cfg == CFG


def test_what_the_loop_measured_comes_back(trained):
    """The point of the exercise — these numbers exist nowhere in the parameters."""
    chain, trainer, step, path = trained
    save_run(path, chain, trainer=trainer,
             perfs={"train": [0.1, 0.5], "ideal": [0.9, 0.9]},
             belief=step.belief, iterations=2)
    artifacts = load_run(path).artifacts
    assert artifacts["perfs"] == {"train": [0.1, 0.5], "ideal": [0.9, 0.9]}
    assert artifacts["iterations"] == 2
    assert torch.equal(artifacts["belief"], step.belief)


def test_a_reloaded_chain_predicts_exactly_what_it_predicted_before(trained):
    """Byte-identical figures on re-plot depend on this and nothing else."""
    chain, trainer, _, path = trained
    save_run(path, chain, trainer=trainer)
    inputs = (torch.rand(N_EPISODES, N_STEPS, CFG.n_observations).round(),
              torch.randint(0, CFG.n_vars, (N_EPISODES, CFG.n_contexts)))
    with torch.no_grad():
        before, _ = chain(*inputs)
        after, _ = load_run(path).chain(*inputs)
    assert torch.equal(before, after)


def test_resuming_restores_the_optimizer_moments(trained):
    """Adam without its moments is not a resumed run, it is a cold restart."""
    chain, trainer, _, path = trained
    save_run(path, chain, trainer=trainer)
    resumed = load_run(path).trainer()
    assert resumed.inference.state_dict()["state"]
    assert (resumed.estimation.state_dict()["param_groups"][0]["lr"]
            == trainer.estimation.param_groups[0]["lr"])


def test_a_run_saved_without_a_trainer_still_loads(trained):
    """The chain is worth keeping even when the run that made it is over."""
    chain, _, _, path = trained
    save_run(path, chain, note="weights only")
    run = load_run(path)
    assert run.trainer_state is None
    assert run.trainer().inference.param_groups[0]["lr"] == CFG.classifier_lr


def test_the_trainer_it_hands_back_reads_the_saved_config(trained):
    chain, trainer, _, path = trained
    save_run(path, chain, trainer=trainer)
    resumed = load_run(path).trainer()
    assert resumed.micro_batch == CFG.micro_batch
    assert resumed.inference.param_groups[0]["lr"] == CFG.classifier_lr


def test_a_run_pinned_to_a_gpu_loads_on_a_machine_without_one(trained):
    """The case re-plotting exists for: figures trained on a gpu, redrawn on a
    laptop. The file is read onto the cpu whatever it was written from, so this has
    to need no argument from the caller."""
    chain, trainer, _, path = trained
    save_run(path, chain, trainer=trainer, marker="gpu run")
    # Rewrite the config as a gpu run would have left it, without needing a gpu.
    saved = torch.load(path, map_location="cpu", weights_only=True)
    saved["cfg"]["device"] = "cuda:0"
    torch.save(saved, path)

    if torch.cuda.is_available():
        assert load_run(path).chain.device.type == "cuda"
    else:
        with pytest.warns(RuntimeWarning, match="falling back to the cpu"):
            run = load_run(path)
        assert run.chain.device == torch.device("cpu")
        assert run.artifacts["marker"] == "gpu run"


def test_map_location_overrides_the_device_the_config_asked_for(trained):
    """A checkpoint made on a gpu is worth reading on a laptop."""
    chain, trainer, _, path = trained
    save_run(path, chain.to("cpu"), trainer=trainer)
    assert load_run(path, map_location="cpu").chain.device == torch.device("cpu")


def test_a_half_written_checkpoint_never_replaces_a_good_one(trained, tmp_path):
    """The moment a long run is most likely to be interrupted is while it saves."""
    chain, trainer, _, path = trained
    save_run(path, chain, trainer=trainer, marker="first")
    original = path.read_bytes()

    class Unsaveable:
        def __reduce__(self):
            raise RuntimeError("this artifact cannot be written")

    with pytest.raises(RuntimeError, match="cannot be written"):
        save_run(path, chain, trainer=trainer, marker=Unsaveable())
    assert path.read_bytes() == original
    assert not list(tmp_path.glob("*.partial"))


def test_loading_creates_the_directory_it_is_asked_to_write_into(trained, tmp_path):
    chain, trainer, _, _ = trained
    nested = tmp_path / "runs" / "nested" / "run.pt"
    save_run(nested, chain, trainer=trainer)
    assert load_run(nested).chain.cfg == CFG
