# rcc

PyTorch implemention of **Representation Classification Chains** from *Factorization Regret mediates compositional generalization in latent space* ([arXiv:2603.27134](https://arxiv.org/abs/2603.27134)).

<img src="docs/images/architecture.png" width="100%">

***Left:** variables' key and query embeddings, contracted into interactions
`Ẑ`. **Centre:** Classifier infers realizations from observations and `Ẑ`, Generator predicts observations back from them. Reward trains the classifier,
self-supervision trains the generator & embeddings. Dashed arrows carry no gradient. **Right:** with
embeddings learned, a controller can be trained to maximize preferred observations.*

## Install

```bash
pip install git+https://github.com/johnschwarcz/rcc
```

Requires Python 3.10+ and PyTorch 2.2+.

## Quick start

```python
import torch
from rcc import RCC, RCCConfig

n_vars = 50
n_contexts = 2
n_realizations = 10
n_observations = 5
n_episodes = 100
n_steps = 30

observations = torch.rand(n_episodes, n_steps, n_observations).round()   # random observations
ctx_inds = torch.randint(0, n_vars, (n_episodes, n_contexts))            # which variables are active

cfg = RCCConfig(n_vars, n_contexts, n_realizations, n_observations, hidden_dim=32)
chain = RCC(cfg)

belief, interaction = chain(observations, ctx_inds)
belief.shape        # (n_episodes, n_steps, n_contexts, n_realizations) — episodes, steps, variables, realizations
```

`belief[e, t, c, r]` is how strongly the chain believes, in episode `e`, after `t` observations, that active variable `c` has realization `r`.

## The four stages

| Stage | Module | Reads | Produces |
| --- | --- | --- | --- |
| 1. Representation | `InteractionEncoder` | `ctx_inds` | interactions per channel |
| 2. Classification | `BeliefClassifier` | observations, interactions | beliefs over each active variable |
| 3. Generation | `ObservationGenerator` | realizations, interactions | predicted observation rates |
| 4. Control | `Controller` | interactions | a distribution over joint realizations |
Parameter groups are split by:

```python
estimation = torch.optim.Adam(chain.estimation_parameters(), lr=1e-3)  # InteractionEncoder and ObservationGenerator 
inference  = torch.optim.Adam(chain.inference_parameters(),  lr=1e-3)  # BeliefClassifier
control    = torch.optim.Adam(chain.control_parameters(),    lr=1e-3)  # Controller
```

InteractionEncoder holds *one key and query embedding per variable per channel*. Each pair of active
variables interact through the dot product of their keys with the other's query,
contributing a total of two interactions per channel.
Embeddings are trained to represent the world — not to make classification easier. 
The goal of `BeliefClassifier` is to provide useful classifications for learning to represent the world.

## Configuration

| Field | Default | Meaning |
| --- | --- | --- |
| `n_vars` | `500` | Size of the latent variable pool. |
| `n_contexts` | `2` | Number of simultaneously active latent variables. |
| `n_realizations` | `10` | Number of discrete values each active variable can take. |
| `n_observations` | `5` | Number of binary observation channels. |
| `embedding_dim` | `30` | Dimensionality of the key/query embeddings that are contracted into interactions. |
| `hidden_dim` | `1000` | Width of every hidden layer. |
| `learn_embeddings` | `True` | Whether embeddings are learned. `False` requires you to supply them. |
| `seed` | `None` | Seed for initialization and sampling. |
| `device` | `None` | `None` stays on the cpu, `'auto'` takes cuda when it is available, or provide explicitly, e.g. `'cuda:0'`. |
| `classifier_lr` | `0.001` | Adam learning rate for the inference objective. |
| `generator_lr` | `None` | Adam learning rate for the estimation objective. `None` follows `classifier_lr`. |
| `control_lr` | `0.003` | Adam learning rate for the control objective. |
| `classifier_entropy_bonus` | `0.1` | `reward_loss`'s entropy bonus. |
| `controller_entropy_bonus` | `0.05` | `controller_loss`'s entropy bonus. |
| `micro_batch` | `None` | Split each batch into slices of this many episodes. |

`RCC(cfg)` reads the architecture and places itself on `device`, and `Trainer(chain)` reads the rest off `chain.cfg`. 
Derived properties: `n_interactions`, `realization_shape`, `n_joint_realizations`, `classifier_input_dim`, `estimation_lr`.

## Objectives

| Function | Trains |
| --- | --- |
| `supervised_loss(belief, target)` | stage 2 (when the true target is available)|
| `reward_loss(goal_belief, selection, correct)` | stage 2 (when only reward is available)|
| `prediction_loss(predicted_rates, observed_rates)` | stages 1, 3 | 
| `embedding_norm_penalty(keys, queries)` | stage 1 |
| `controller_loss(value, predicted_value, log_prob, entropy)` | stage 4 |

## Example classifier training step

To *run* a chain you need a task to provide **observations**, **ctx_inds**, **teaching signal**.
[coggrid](https://github.com/johnschwarcz/coggrid) can be installed with `pip install -e ".[dev]"`.

```python
from coggrid import CogGridConfig, World, run_observers
from rcc import RCC, RCCConfig, supervised_loss

world = World(CogGridConfig(n_vars=50, n_contexts=2, n_realizations=4,  n_observations=3, embedding_dim=30, n_steps=20))
batch = world.sample_episodes(128, split="train")
target_belief = run_observers(batch)["joint"].belief   # the ideal observer's belief

chain = RCC(RCCConfig(n_vars=50, n_contexts=2, n_realizations=4, n_observations=3, hidden_dim=32))
optimizer = torch.optim.Adam(chain.inference_parameters(), lr=1e-3)
belief, interaction = chain(batch.observations, batch.ctx_inds)
loss = supervised_loss(belief, target_belief)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

<img src="docs/images/belief_accumulation.png" width="100%">

<img src="docs/images/belief_average.png" width="100%">

## Stage 4

The controller never sees an observation. It reads the interactions and picks a joint realization to put the world into.

```python
from rcc import intrinsic_value

policy, predicted_value = chain.controller(interaction.score)
actions, log_prob, entropy = chain.controller.act(policy)
value = intrinsic_value(predicted_observations, preferences)
```

`intrinsic_value` scores a set of predicted observation against preferences.

<img src="docs/images/policy.png" width="85%">

## Example training curves

<img src="docs/images/losses.png" width="100%">

## Example generalization

<img src="docs/images/training.png" width="70%">

## Citation

```bibtex
@article{schwarcz2026factorization,
  title  = {Factorization Regret mediates compositional generalization in latent space},
  author = {Schwarcz, John},
  year   = {2026},
  eprint = {2603.27134},
  archivePrefix = {arXiv}
}
```

## License

MIT — see [LICENSE](LICENSE).
