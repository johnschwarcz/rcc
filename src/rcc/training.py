"""Training — one optimizer per objective, and the order they are stepped in.
Nothing here is task-specific: a batch of episodes arrives as tensors, and what an
action turned out to be worth arrives as a table the caller has already scored.
"""
from typing import NamedTuple
import torch
from torch import Tensor
from ._helpers import require_shape
from .chain import RCC
from .classifier import goal_accuracy, sample_goal, select_goal
from .losses import (
    controller_loss,
    embedding_norm_penalty,
    prediction_loss,
    reward_loss,
)

__all__ = ["ControlStep", "Trainer", "TrainingStep"]

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
    stage 4, which shares no parameter with either. ``generator_lr`` covers stages
    1 and 3 together and defaults to ``classifier_lr``.

    Stage 2 is taught the paper's way: the chain commits to one answer for the goal
    variable and is told only whether it was right.

    ``micro_batch`` splits each batch into slices of that many episodes, accumulating
    their gradients before one optimizer step, so peak memory follows the slice
    rather than the batch. It is a memory knob and not a statistical one: the batch
    is still what a step is computed over. See ``step`` for where it is exact.
    >>> import torch
    >>> from rcc import RCC, RCCConfig
    >>> cfg = RCCConfig(n_vars=8, n_realizations=3, n_observations=2, hidden_dim=16, seed=0)
    >>> trainer = Trainer(RCC(cfg))
    >>> step = trainer.step(
    ...     observations=torch.rand(4, 5, 2).round(),
    ...     ctx_inds=torch.randint(0, cfg.n_vars, (4, cfg.n_contexts)),
    ...     goal_ind=torch.zeros(4, dtype=torch.long),
    ...     goal_value=torch.zeros(4, dtype=torch.long))
    >>> 0.0 <= step.accuracy <= 1.0
    True
    """

    def __init__(self, chain: RCC, *,
        classifier_lr: float = 1e-3, generator_lr: float | None = None,
        control_lr: float = 3e-3,
        classifier_entropy_bonus: float = 0.1, micro_batch: int | None = None,
        controller_entropy_bonus: float = 0.05,
        generator: torch.Generator | None = None,) -> None:
        self.chain = chain
        self.generator = generator
        self.classifier_entropy_bonus = classifier_entropy_bonus
        self.micro_batch = micro_batch
        self.controller_entropy_bonus = controller_entropy_bonus
        # What a guess is worth when the belief is uninformative, which is the
        # weight prediction_loss puts on its unconditional term. Not a free
        # parameter: the task's chance rate defines it.
        self.chance = 1 / chain.cfg.n_realizations
        self.inference = torch.optim.Adam(chain.inference_parameters(), lr=classifier_lr)
        self.estimation = torch.optim.Adam(chain.estimation_parameters(),
            lr=classifier_lr if generator_lr is None else generator_lr)
        self.control = torch.optim.Adam(chain.control_parameters(), lr=control_lr)

    def step(self, observations: Tensor, ctx_inds: Tensor, goal_ind: Tensor,
        goal_value: Tensor,) -> TrainingStep:
        """Step the inference and estimation objectives over one batch.
        observations: (n_episodes, n_steps, n_observations)
        ctx_inds: (n_episodes, n_contexts)
        goal_ind, goal_value: (n_episodes,)
        returns: TrainingStep, whose belief is detached

        With ``micro_batch`` set this accumulates over slices. That is exact for every
        objective that is a plain mean over episodes, which is ``reward_loss`` and the
        norm penalty. ``prediction_loss`` is not: it normalizes by how many episodes in
        the batch were correct, which then applies per slice, so a larger slice is
        closer.
        """
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

    def _accumulate(self, observations: Tensor, ctx_inds: Tensor, goal_ind: Tensor,
        goal_value: Tensor, share: float) -> tuple[float, float, float, Tensor]:
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
            selection = sample_goal(goal, self.generator)[:, -1]
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
