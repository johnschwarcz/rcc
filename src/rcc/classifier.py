"""Stage 2 — observations and interactions to a belief.
The readout emits increments; belief is their running sum after a softmax.
"""

import torch
from torch import Tensor, nn
from ._helpers import require_shape, sample_categorical, seeded
from .config import RCCConfig

__all__ = ["BeliefClassifier", "goal_accuracy", "sample_goal", "select_goal"]

class BeliefClassifier(nn.Module):
    """Read the observation stream and emit marginal beliefs over active variable."""

    def __init__(self, cfg: RCCConfig) -> None:
        super().__init__()
        self.cfg = cfg
        with seeded(cfg.seed):
            self.readin = nn.Linear(cfg.classifier_input_dim, cfg.hidden_dim)
            self.readout = nn.Linear(cfg.hidden_dim, cfg.n_contexts * cfg.n_realizations)
            self.rnn = nn.LSTM(cfg.hidden_dim, cfg.hidden_dim, batch_first=True)
            self.initial_short_term = nn.Parameter(torch.randn(1, 1, cfg.hidden_dim))
            self.initial_long_term = nn.Parameter(torch.randn(1, 1, cfg.hidden_dim))

    def forward(self, observations: Tensor, interactions: Tensor) -> Tensor:
        """Integrate observations given the interactions.
        observations: (n_episodes, n_steps, n_observations),
        interactions: (n_episodes, n_observations, n_interactions)
        Returns: (n_episodes, n_steps, n_contexts, n_realizations)
        """
        self._check(observations, interactions)
        n_episodes, n_steps, _ = observations.shape
        Z = interactions.detach().reshape(n_episodes, 1, -1)

        per_step = torch.cat((observations, Z.expand(-1, n_steps, -1)), -1)
        hidden = torch.relu(self.readin(per_step))
        short_term = self.initial_short_term.expand(-1, n_episodes, -1).contiguous()
        long_term = self.initial_long_term.expand(-1, n_episodes, -1).contiguous()
        recurrent, _ = self.rnn(hidden, (short_term, long_term))
        logits = self.readout(recurrent).cumsum(1)
        logits = logits.reshape(
            n_episodes, n_steps, self.cfg.n_contexts, self.cfg.n_realizations)
        return torch.softmax(logits, -1)

    def _check(self, observations: Tensor, interactions: Tensor) -> None:
        cfg = self.cfg
        if observations.ndim != 3 or observations.shape[2] != cfg.n_observations:
            raise ValueError("observations must have shape (n_episodes, n_steps, "
                f"{cfg.n_observations}), got {tuple(observations.shape)}")
        require_shape("interactions", interactions,
            (observations.shape[0], cfg.n_observations, cfg.n_interactions),)


def select_goal(belief: Tensor, goal_ind: Tensor) -> Tensor:
    """Pick out the belief over the one variable that is being asked about.
    belief: (n_episodes, n_steps, n_contexts, n_realizations)
    returns: (n_episodes, n_steps, n_realizations)
    """
    require_shape("goal_ind", goal_ind, (belief.shape[0],))
    episodes = torch.arange(belief.shape[0], device=belief.device)
    return belief[episodes, :, goal_ind]


def sample_goal(
    goal_belief: Tensor, generator: torch.Generator | None = None
) -> Tensor:
    """Draw a realization for the goal variable from the belief about it.
    goal_belief: (n_episodes, n_steps, n_realizations)
    returns: (n_episodes, n_steps)
    """
    return sample_categorical(goal_belief, generator)


def goal_accuracy(selection: Tensor, goal_value: Tensor) -> Tensor:
    """Whether each sampled realization matched the truth, as a float mask.
    selection: (n_episodes, n_steps)
    goal_value: (n_episodes,)
    returns: (n_episodes, n_steps)
    """
    require_shape("goal_value", goal_value, (selection.shape[0],))
    return (selection == goal_value[:, None]).float()
