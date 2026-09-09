from typing import Any, NamedTuple
import torch
from torch import Tensor
from ._helpers import require_shape, sample_categorical
from .chain import RCC
from .classifier import goal_accuracy, select_goal
from .losses import (
    controller_loss,
    embedding_norm_penalty,
    prediction_loss,
    reward_loss,
)

__all__ = ["ControlStep", "Trainer", "TrainingStep"]

def _or_config(override: Any, configured: Any) -> Any:
    """The explicit argument when there is one, the config's field otherwise.
    ``None`` is not a legal value for any of these, so it is free to mean "unset".
    """
    return configured if override is None else override
class TrainingStep(NamedTuple):
    """One iteration of stages 1-3.
    Unpacks as classifier_loss, generator_loss, accuracy, belief.
    accuracy: share of episodes whose final-step argmax matched the truth
    belief: (n_episodes, n_steps, n_contexts, n_realizations)
    """
    classifier_loss: float
    generator_loss: float
    accuracy: float
    belief: Tensor
class ControlStep(NamedTuple):
    """One iteration of stage 4. Unpacks as loss, value.
    value: the mean worth of the actions actually taken
    """
    loss: float
    value: float
class Trainer:
    """Each objective with its own optimizer over its own parameter group.
    ``step`` trains stages 1-3 from a single forward pass; ``control_step`` trains
    stage 4, which shares no parameter with either.
    """

    def __init__(self, chain: RCC, *,
            classifier_lr: float | None = None, generator_lr: float | None = None, control_lr: float | None = None,
            classifier_entropy_bonus: float | None = None, controller_entropy_bonus: float | None = None,
            micro_batch: int | None = None, generator: torch.Generator | None = None) -> None:
        cfg = chain.cfg
        self.chain = chain
        self.chance = 1 / cfg.n_realizations
        self.classifier_entropy_bonus = _or_config(
            classifier_entropy_bonus, cfg.classifier_entropy_bonus)
        self.controller_entropy_bonus = _or_config(
            controller_entropy_bonus, cfg.controller_entropy_bonus)
        self.micro_batch = _or_config(micro_batch, cfg.micro_batch)

        self.generator = generator
        if generator is None and cfg.seed is not None:
            self.generator = torch.Generator(device=chain.device).manual_seed(cfg.seed)

        inference_lr = _or_config(classifier_lr, cfg.classifier_lr)
        estimation_lr = _or_config(generator_lr, _or_config(cfg.generator_lr, inference_lr))
        self.inference = torch.optim.Adam(chain.inference_parameters(), lr=inference_lr)
        self.estimation = torch.optim.Adam(chain.estimation_parameters(), lr=estimation_lr)
        self.control = torch.optim.Adam(chain.control_parameters(),
            lr=_or_config(control_lr, cfg.control_lr))

    # ------------------------------------------------------- saving and loading
    def state_dict(self) -> dict[str, Any]:
        """The three optimizers' state, so a run can be picked up where it stopped."""
        return {"inference": self.inference.state_dict(),
                "estimation": self.estimation.state_dict(),
                "control": self.control.state_dict()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore what :meth:`state_dict` wrote, onto the chain this Trainer holds."""
        self.inference.load_state_dict(state["inference"])
        self.estimation.load_state_dict(state["estimation"])
        self.control.load_state_dict(state["control"])

    # --------------------------------------------------------------- the steps
    def _place(self, *tensors: Tensor) -> tuple[Tensor, ...]:
        """The batch, on whatever device the chain is on."""
        device = self.chain.device
        return tuple(tensor.to(device) for tensor in tensors)

    def step(self, observations: Tensor, ctx_inds: Tensor, goal_ind: Tensor, goal_value: Tensor) -> TrainingStep:
        """Step the inference and estimation objectives over one batch.
        observations: (n_episodes, n_steps, n_observations)
        ctx_inds: (n_episodes, n_contexts)
        goal_ind, goal_value: (n_episodes,)
        returns: TrainingStep, whose belief is detached
        With ``micro_batch`` set this accumulates over slices. 
        """
        observations, ctx_inds, goal_ind, goal_value = self._place(
            observations, ctx_inds, goal_ind, goal_value)
        n_episodes = observations.shape[0]
        size = min(self.micro_batch or n_episodes, n_episodes)

        self.inference.zero_grad()
        self.estimation.zero_grad()
        classifier_total = generator_total = accuracy_total = 0.0
        beliefs = []
        for start in range(0, n_episodes, size):
            cut = slice(start, min(start + size, n_episodes))
            share = (cut.stop - cut.start) / n_episodes
            classifier_loss, generator_loss, accuracy, belief = self._accumulate(
                observations[cut], ctx_inds[cut], goal_ind[cut], goal_value[cut], share)
            classifier_total += classifier_loss * share
            generator_total += generator_loss * share
            accuracy_total += accuracy * share
            beliefs.append(belief)

        self.inference.step()
        self.estimation.step()
        return TrainingStep(classifier_total, generator_total, accuracy_total,
            torch.cat(beliefs))

    def _accumulate(self, observations: Tensor, ctx_inds: Tensor, goal_ind: Tensor, goal_value: Tensor,
            share: float) -> tuple[float, float, float, Tensor]:
        """One micro-batch, leaving its gradients accumulated on the parameters.
        share: this slice's fraction of the batch, which scales both losses so that
        what accumulates is the whole batch's gradient
        returns: classifier_loss, generator_loss, accuracy, detached belief
        """
        belief, interaction = self.chain(observations, ctx_inds)

        # The verdict on the goal variable: one committed answer, right or wrong. It
        # teaches stages 1 and 3, and under objective='reward' stage 2 as well.
        goal = select_goal(belief, goal_ind)
        with torch.no_grad():
            selection = sample_categorical(goal, self.generator)[:, -1]
            correct = goal_accuracy(selection[:, None], goal_value)[:, 0]

        # Stage 2. No retain_graph: stage 2 reads the interactions detached, so this
        # backward never enters the graph the estimation objective walks below.
        classifier_loss = reward_loss(goal, selection, correct,
            entropy_bonus=self.classifier_entropy_bonus)
        (classifier_loss * share).backward()

        rates = self.chain.reconstruct(belief, interaction, goal_ind,
            goal_selection=selection, goal_correct=correct, generator=self.generator)
        generator_loss = prediction_loss(rates, observations.mean(1),
            correct=correct, chance=self.chance)
        # The penalty is stage 1's own, holding the embeddings on the sphere they
        # started on, so it is stepped with the objective but not reported as it.
        ((generator_loss
            + embedding_norm_penalty(interaction.keys, interaction.queries))
            * share).backward()

        accuracy = goal_accuracy(goal.detach().argmax(-1), goal_value)[:, -1].mean()
        return (classifier_loss.item(), generator_loss.item(), accuracy.item(),
            belief.detach())

    def control_step(self, ctx_inds: Tensor, landscape: Tensor) -> ControlStep:
        """Step the control objective over one batch, on the interactions alone.
        ctx_inds: (n_episodes, n_contexts)
        landscape: (n_episodes, *realization_shape), what each joint action is worth
        returns: ControlStep
        """
        ctx_inds, landscape = self._place(ctx_inds, landscape)
        n_episodes = ctx_inds.shape[0]
        require_shape("landscape", landscape,
            (n_episodes, *self.chain.cfg.realization_shape))
        policy, predicted = self.chain.controller(self.chain.encoder(ctx_inds).score)
        actions, log_prob, entropy = self.chain.controller.act(policy, self.generator)

        episodes = torch.arange(n_episodes, device=landscape.device)
        taken = landscape[(episodes, *actions.unbind(1))]  # One index per active variable.
        loss = controller_loss(taken, predicted, log_prob, entropy,
            entropy_bonus=self.controller_entropy_bonus)
        self.control.zero_grad()
        loss.backward()
        self.control.step()
        return ControlStep(loss.item(), taken.mean().item())
