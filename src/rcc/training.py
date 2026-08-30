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
    supervised_loss,
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
    stage 4, which shares no parameter with either.
    >>> import torch
    >>> from rcc import RCC, RCCConfig
    >>> cfg = RCCConfig(n_vars=8, n_realizations=3, n_observations=2, hidden_dim=16, seed=0)
    >>> trainer = Trainer(RCC(cfg))
    >>> step = trainer.step(
    ...     observations=torch.rand(4, 5, 2).round(),
    ...     ctx_inds=torch.randint(0, cfg.n_vars, (4, cfg.n_contexts)),
    ...     posterior=torch.full((4, 5, cfg.n_contexts, cfg.n_realizations), 1 / 3),
    ...     goal_ind=torch.zeros(4, dtype=torch.long),
    ...     goal_value=torch.zeros(4, dtype=torch.long))
    >>> 0.0 <= step.accuracy <= 1.0
    True
    """

    def __init__(self, chain: RCC, *, lr: float = 1e-3, control_lr: float = 3e-3,
        generator: torch.Generator | None = None,) -> None:
        self.chain = chain
        self.generator = generator
        self.inference = torch.optim.Adam(chain.inference_parameters(), lr=lr)
        self.estimation = torch.optim.Adam(chain.estimation_parameters(), lr=lr)
        self.control = torch.optim.Adam(chain.control_parameters(), lr=control_lr)
        # What a guess is worth when the belief is uninformative, which is the
        # weight prediction_loss puts on its unconditional term.
        self.chance = 1 / chain.cfg.n_realizations

    def step(self, observations: Tensor, ctx_inds: Tensor, posterior: Tensor,
        goal_ind: Tensor, goal_value: Tensor,) -> TrainingStep:
        """Step the inference and estimation objectives over one batch.
        observations: (n_episodes, n_steps, n_observations)
        ctx_inds: (n_episodes, n_contexts)
        posterior: (n_episodes, n_steps, n_contexts, n_realizations), the target
        goal_ind, goal_value: (n_episodes,)
        returns: TrainingStep
        """
        belief, interaction = self.chain(observations, ctx_inds)

        # Stage 2. No retain_graph: stage 2 reads the interactions detached, so this
        # backward never enters the graph the estimation objective walks below.
        classifier_loss = supervised_loss(belief, posterior)
        self.inference.zero_grad()
        classifier_loss.backward()

        # Stages 1 and 3, taught by the verdict on the goal variable.
        with torch.no_grad():
            goal = select_goal(belief, goal_ind)
            selection = sample_goal(goal, self.generator)[:, -1]
            correct = goal_accuracy(selection[:, None], goal_value)[:, 0]
        rates = self.chain.reconstruct(belief, interaction, goal_ind,
            goal_selection=selection, goal_correct=correct, generator=self.generator)
        generator_loss = prediction_loss(rates, observations.mean(1),
            correct=correct, chance=self.chance)
        self.estimation.zero_grad()
        # The penalty is stage 1's own, holding the embeddings on the sphere they
        # started on, so it is stepped with the objective but not reported as it.
        (generator_loss
            + embedding_norm_penalty(interaction.keys, interaction.queries)).backward()

        self.inference.step()
        self.estimation.step()
        accuracy = goal_accuracy(goal.argmax(-1), goal_value)[:, -1].mean()
        return TrainingStep(
            classifier_loss.item(), generator_loss.item(), accuracy.item(), belief)

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
        loss = controller_loss(taken, predicted, log_prob, entropy)
        self.control.zero_grad()
        loss.backward()
        self.control.step()
        return ControlStep(loss.item(), taken.mean().item())
