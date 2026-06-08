# DQ4FairIM-TC

> **Dynamic Fairness-Aware Seed Selection for Real-Time Information Diffusion**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Optional-orange?logo=pytorch)](https://pytorch.org/)

A deep reinforcement learning framework for **fair influence maximization (IM)** on continuously evolving social graphs. DQ4FairIM-TC extends [DQ4FairIM (Saxena et al., 2025)](https://arxiv.org/abs/2512.00545) with three key contributions:

- **Step-level temporal dynamics** — the graph is perturbed between every individual seed selection, not just between episodes.
- **Dual-objective reward** — jointly optimises maximin fairness, penalises inter-community disparity, and includes a time-critical latency penalty.
- **Full embedding backpropagation** — all five GNN weight matrices (W₁–W₅) are trained end-to-end via truncated BPTT to learn graph-topology-aware fairness representations.

An interactive live visualisation is included, rendering graph dynamics, seed placement, and IC propagation step by step.

---

## Results

| Dataset | Method | Outreach ↑ | Fairness ↑ | Disparity ↓ | Gini ↓ |
|---------|--------|-----------|-----------|------------|--------|
| FB-1000 | Degree *(best baseline)* | 0.5633 | 0.5595 | 0.0109 | 0.0048 |
| FB-1000 | **DQ4FairIM-TC** | 0.5581 | 0.5573 | **0.0011** | **0.0005** |
| HBA-300 | Parity *(best baseline)* | 0.5410 | 0.5269 | 0.0169 | 0.0079 |
| HBA-300 | **DQ4FairIM-TC** | **0.5525** | **0.5518** | **0.0042** | **0.0019** |

DQ4FairIM-TC achieves **89.9% disparity reduction** on the Facebook 1000-node subgraph, reaching a near-zero Gini coefficient of **0.0005**.

---

## Installation

```bash
git clone  https://github.com/Lakshya-23 DQ4FairIM-TC_Fairness-Aware_Influence_Maximization.git
cd DQ4FairIM_TC
pip install -r requirements.txt
```

Python 3.9 or higher is required. PyTorch is optional but provides a ~4× speedup; the NumPy fallback runs on any machine.

---

## Quick Start

```python
from src import (
    load_facebook_snap, FairIMEnvironment, DQNAgent,
    get_adjacency_matrix, evaluate_all_baselines
)

# Load the Facebook ego-network
samples = load_facebook_snap(sample_size=1000, n_samples=10, seed=42)
train_pool, test_samples = samples[:6], samples[6:]
g, communities = train_pool[0]

feature_dim = 3 + len(set(communities.values()))

agent = DQNAgent(
    feature_dim=feature_dim,
    embed_dim=64,
    epsilon_decay=0.9851,
    lambda_disparity=0.5,
)

# Train for one episode
env = FairIMEnvironment(
    base_graph=g,
    communities=communities,
    budget=10,
    phi=1.0,
    lambda_disparity=0.5,
    step_change_rate=0.005,
)

state = env.reset()
adj = get_adjacency_matrix(env.graph)
done = False

while not done:
    action = agent.select_action(state, adj, env.available_actions())
    state, reward, done, info = env.step(action)
    adj = get_adjacency_matrix(env.graph)
    agent.store(state, adj, action, reward, state, adj, done)
    agent.update()
```

Run `test.py` for a standalone training and evaluation script.

---

## Project Structure

```
fairimv3/
├── src/
│   ├── __init__.py          # Public API exports
│   ├── agent.py             # S2V + DQN (PyTorch & NumPy backends)
│   ├── baselines.py         # Degree, PageRank, Parity, Fair-PageRank, ...
│   ├── diffusion.py         # IC model, TC-IC model, Gini coefficient
│   ├── environment.py       # Fair-IM MDP with temporal graph dynamics
│   └── graph_utils.py       # HBA, SBM graph builders; SNAP Facebook loader
├── checkpoints/
│   └── agent_v3.pt          # Pre-trained checkpoint (Facebook dataset)
├── results/
│   ├── training_v3.png
│   ├── comparison_v3_FB_1000nodes.png
│   ├── comparison_v3_FB_5000nodes.png
│   ├── temporal_v3.png
│   └── ...
└── test.py                  # Standalone training/evaluation script
```

---

## Architecture

### Node Embeddings

Node embeddings are computed via an iterative message-passing process over T = 4 iterations:

$$\mu_v^{(t)} = \text{ReLU}\left(W_1 x_v + W_2 \sum_{u \in \mathcal{N}(v)} \mu_u^{(t-1)}\right)$$

The Q-value for action node *a* is:

$$Q(S, a) = W_3 \cdot \text{ReLU}\left(W_4 \sum_v \mu_v \;\|\; W_5 \mu_a\right)$$

All five weight matrices {W₁, W₂, W₃, W₄, W₅} are trained end-to-end via truncated BPTT through the final iteration, enabling the network to learn fairness-relevant structural features.

### Dual Fairness Reward

$$r_t = \Delta\sigma(S,a) + \phi \cdot \Delta\min_i \sigma_i(S,a) - \lambda_d \cdot \Delta(\max_i \sigma_i - \min_i \sigma_i) - \gamma \cdot \text{latency}(S,a)$$

The disparity term penalises seed choices that widen the community gap and rewards those that narrow it. The latency term penalises slower diffusion when operating under a time-critical deadline.

### Continuous Temporal Dynamics

- **At `reset()`** — remove and replace `temporal_change_rate` fraction of edges (default 2%).
- **After each `step()`** — remove and replace `step_change_rate` fraction of edges (default 0.5%).

The agent re-reads the adjacency matrix after every step, enabling real-time adaptation to graph changes.

---

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `phi` | 1.0 | Weight on maximin fairness term in reward |
| `lambda_disparity` | 0.5 | Weight on inter-community disparity penalty |
| `gamma_latency` | 0.5 | Latency penalty weight (active when deadline is set) |
| `temporal_change_rate` | 0.02 | Fraction of edges perturbed at each episode reset |
| `step_change_rate` | 0.005 | Fraction of edges perturbed after each seed selection |
| `budget` (k) | 10 | Number of seeds to select per episode |
| `embed_dim` | 64 | Graph embedding dimension |
| `n_s2v_iters` | 4 | Number of message-passing iterations |
| `epsilon_decay` | 0.9851 | ε-greedy decay rate (reaches ε_min ≈ 0.05 at episode 200) |

---

## Datasets

The SNAP Facebook dataset is downloaded automatically from `snap.stanford.edu` on first run. Community labels are derived from Louvain detection (two largest communities). Falls back to an SBM approximation if the download fails.

| Dataset | Nodes | Edges | Minority % | Source |
|---------|-------|-------|-----------|--------|
| HBA Synthetic | 300 | ~1,500 | 15% | Generated (`build_hba_graph`) |
| FB-1000 | 1,000 | ~8,400 | ~30% | SNAP Facebook ego-network |
| FB-5000 (full) | 4,039 | 88,234 | ~28% | SNAP Facebook ego-network |

---

## Evaluation Metrics

All metrics are computed via Monte Carlo IC simulation (300 samples at evaluation time):

| Metric | Formula | Goal |
|--------|---------|------|
| **Outreach** | σ(S) / \|V\| | ↑ Higher is better |
| **Maximin Fairness** | min_i σ_i(S) | ↑ Higher is better |
| **Disparity** | max_i σ_i − min_i σ_i | ↓ Lower is better |
| **Gini Coefficient** | Inequality across per-community outreach | ↓ Lower is better |
| **Temporal Resilience** | Performance sweep over step-change rates 0%–5% | — |

---

## Baselines

| Baseline | Description |
|----------|-------------|
| Degree | Top-k highest-degree nodes |
| PageRank | Top-k PageRank nodes |
| Parity | Degree-based, proportional community allocation |
| Fair-PageRank | PageRank with proportional community allocation |
| Greedy-Maximin | Greedy selection maximising maximin fairness *(slow)* |

---

## Citation

If you use this code, please cite the foundational works:

```bibtex
@article{saxena2025dq4fairim,
  title   = {DQ4FairIM: Fairness-aware Influence Maximization using Deep Reinforcement Learning},
  author  = {Saxena, A. and Yadav, H. K. and Rutten, B. and Jha, S. S.},
  journal = {arXiv:2512.00545},
  year    = {2025}
}

@article{ali2023fairtcim,
  title   = {FAIRTCIM: Fair Time-Critical Influence Maximization},
  author  = {Ali, J. and Cruciani, J. and Saxena, S. and Giannella, C.},
  journal = {IEEE Transactions on Knowledge and Data Engineering},
  volume  = {35},
  number  = {3},
  pages   = {2876--2889},
  year    = {2023}
}

@article{meena2025pdtfim,
  title   = {Privacy, Diversity, and Temporal Fairness in Influence Maximization},
  author  = {Meena, G. and Singh, K. and Kumar, S.},
  journal = {IEEE Transactions on Network Science and Engineering},
  year    = {2025}
}
```
