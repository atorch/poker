"""
A DQN-style learner with one Q output per action head (poker.actions), in big blinds.

- QNetwork maps features (poker.features) to Q-values for the learned heads (CHECK_CALL and
  RAISE_1..3). Q(FOLD) is not learned: with reward = the deal's chip change, folding is worth
  exactly -(chips committed so far), which the environment supplies as fold_values.
- Monte-Carlo targets: DealRecorder keeps each table's non-fold decisions during a deal; when
  the deal ends, every one of them is stored in the ReplayBuffer with the deal's reward as its
  target, and Learner.train_step regresses Q(s, a) toward it with an MSE loss.
- Temporal-difference targets (the alternative): TransitionRecorder links each decision to the
  learner's next decision in the same deal (or to the end of the deal), and
  Learner.expected_sarsa_targets computes reward + E[Q(next)] under the epsilon-greedy policy
  (gamma = 1, a deal is short), with Q from a target network that tracks the online one by
  Polyak averaging (Learner.soft_update).
- DQNAgent plays from a network (benchmark, interactive play). Models are saved as <stem>.pt
  (weights) plus <stem>.json (feature version, heads, layer sizes); load_agent checks the
  config and fails loudly on a mismatch.
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from poker.actions import HEAD_NAMES, N_HEADS, Head, game_action, legal_mask
from poker.features import FEATURE_VERSION, N_FEATURES, encode

LEARNED_HEADS = [head for head in Head if head != Head.FOLD]
MODEL_FORMAT = "poker-dqn"


class QNetwork(nn.Module):
    def __init__(self, n_features=N_FEATURES, hidden=(256, 256)):
        super().__init__()
        layers, size = [], n_features
        for width in hidden:
            layers += [nn.Linear(size, width), nn.ReLU()]
            size = width
        layers.append(nn.Linear(size, len(LEARNED_HEADS)))
        self.net = nn.Sequential(*layers)
        self.hidden = tuple(hidden)

    def forward(self, features):
        return self.net(features)

    @torch.no_grad()
    def learned_q(self, features):
        """Q-values (numpy, big blinds) of the learned heads for a batch of feature rows."""
        return self(torch.as_tensor(features)).numpy()


def full_q_values(learned_q, fold_values, masks):
    """Q over all HEADS: fold from fold_values, the rest from the network, -inf where illegal."""
    q = np.empty((len(learned_q), N_HEADS), dtype=np.float32)
    q[:, Head.FOLD] = fold_values
    q[:, [int(head) for head in LEARNED_HEADS]] = learned_q
    q[~masks] = -np.inf
    return q


def epsilon_greedy(q, masks, epsilon, rng):
    """Greedy heads, replaced by a uniformly random legal head with probability epsilon."""
    heads = q.argmax(axis=1)
    explore = rng.random(len(q)) < epsilon
    for i in np.flatnonzero(explore):
        heads[i] = rng.choice(np.flatnonzero(masks[i]))
    return heads


def softmax_policy(q, temperature):
    """Action probabilities over HEADS (rows of q); temperature 0 means greedy."""
    q = np.atleast_2d(q)
    if temperature <= 0:
        probabilities = np.zeros_like(q)
        probabilities[np.arange(len(q)), q.argmax(axis=1)] = 1.0
        return probabilities
    logits = (q - q.max(axis=1, keepdims=True)) / temperature
    weights = np.exp(logits)  # exp(-inf) = 0 for illegal heads
    return weights / weights.sum(axis=1, keepdims=True)


class ReplayBuffer:
    """A ring buffer of (features, head, target) for Monte-Carlo regression."""

    def __init__(self, capacity, n_features=N_FEATURES):
        self.features = np.zeros((capacity, n_features), dtype=np.float32)
        self.heads = np.zeros(capacity, dtype=np.int64)
        self.targets = np.zeros(capacity, dtype=np.float32)
        self.capacity = capacity
        self.size = 0
        self.position = 0

    def __len__(self):
        return self.size

    def add(self, features, heads, targets):
        for row, head, target in zip(features, heads, targets):
            self.features[self.position] = row
            self.heads[self.position] = head
            self.targets[self.position] = target
            self.position = (self.position + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, rng):
        indices = rng.integers(0, self.size, size=batch_size)
        return self.features[indices], self.heads[indices], self.targets[indices]


class DealRecorder:
    """Per-table decisions of the deal in progress; finished deals become Monte-Carlo targets."""

    def __init__(self, n_tables):
        self.pending = [[] for _ in range(n_tables)]

    def record(self, features, heads):
        for i, head in enumerate(heads):
            # Note: Q(fold) is exact, so fold decisions carry nothing for the network to learn
            if head != Head.FOLD:
                self.pending[i].append((features[i].copy(), int(head)))

    def finish(self, rewards, done):
        """(features, heads, targets) for every decision of the deals that just ended."""
        features, heads, targets = [], [], []
        for i in np.flatnonzero(done):
            for row, head in self.pending[i]:
                features.append(row)
                heads.append(head)
                targets.append(rewards[i])
            self.pending[i] = []
        return features, heads, targets


class TransitionBuffer:
    """A ring buffer of (features, head, reward, next features, next mask, next fold value, done)."""

    def __init__(self, capacity, n_features=N_FEATURES):
        self.fields = {
            "features": np.zeros((capacity, n_features), dtype=np.float32),
            "heads": np.zeros(capacity, dtype=np.int64),
            "rewards": np.zeros(capacity, dtype=np.float32),
            "next_features": np.zeros((capacity, n_features), dtype=np.float32),
            "next_masks": np.zeros((capacity, N_HEADS), dtype=bool),
            "next_fold_values": np.zeros(capacity, dtype=np.float32),
            "dones": np.zeros(capacity, dtype=bool),
        }
        self.capacity = capacity
        self.size = 0
        self.position = 0

    def __len__(self):
        return self.size

    def add(self, transitions):
        for transition in transitions:
            for name, value in transition.items():
                self.fields[name][self.position] = value
            self.position = (self.position + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, rng):
        indices = rng.integers(0, self.size, size=batch_size)
        return {name: values[indices] for name, values in self.fields.items()}


class TransitionRecorder:
    """
    Per-table pending decision, completed by the learner's next decision in the same deal
    (reward 0) or by the end of the deal (the deal's reward, done). Folds end the learner's deal
    with an exactly known value, so they are not stored.
    """

    def __init__(self, n_tables):
        self.pending = [None] * n_tables

    def complete(self, batch):
        """Transitions into the new pending decisions in `batch` (call before record)."""
        transitions = []
        for i, pending in enumerate(self.pending):
            if pending is not None:
                transitions.append({**pending, "rewards": 0.0, "next_features": batch["features"][i],
                                    "next_masks": batch["masks"][i], "next_fold_values": batch["fold_values"][i],
                                    "dones": False})
                self.pending[i] = None
        return transitions

    def record(self, features, heads):
        for i, head in enumerate(heads):
            self.pending[i] = None if head == Head.FOLD else {"features": features[i].copy(), "heads": int(head)}

    def finish(self, rewards, done):
        """Terminal transitions for the deals that just ended."""
        transitions = []
        for i in np.flatnonzero(done):
            if self.pending[i] is not None:
                transitions.append({**self.pending[i], "rewards": float(rewards[i]), "dones": True})
                self.pending[i] = None
        return transitions


class Learner:
    def __init__(self, hidden=(256, 256), learning_rate=3e-4, max_grad_norm=10.0):
        self.network = QNetwork(hidden=hidden)
        self.target_network = QNetwork(hidden=hidden)
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        self.max_grad_norm = max_grad_norm
        # Note: maps a head (1..4) to its column in the network's output (0..3)
        self.columns = torch.full((N_HEADS,), -1, dtype=torch.int64)
        for column, head in enumerate(LEARNED_HEADS):
            self.columns[int(head)] = column

    def train_step(self, features, heads, targets):
        """One MSE gradient step on Q(s, head) toward the targets; returns the loss."""
        columns = self.columns[torch.as_tensor(heads)]
        assert (columns >= 0).all(), "fold decisions should not be trained"
        q = self.network(torch.as_tensor(features)).gather(1, columns[:, None]).squeeze(1)
        loss = nn.functional.mse_loss(q, torch.as_tensor(targets))
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return loss.item()

    def expected_sarsa_targets(self, batch, epsilon):
        """reward + E[Q_target(next)] under epsilon-greedy over the legal next heads (0 if done)."""
        targets = batch["rewards"].astype(np.float32).copy()
        live = ~batch["dones"]
        if live.any():
            q = full_q_values(self.target_network.learned_q(batch["next_features"][live]),
                              batch["next_fold_values"][live], batch["next_masks"][live])
            legal = batch["next_masks"][live]
            best = q.max(axis=1)
            mean = np.where(legal, q, 0.0).sum(axis=1) / legal.sum(axis=1)
            targets[live] += (1 - epsilon) * best + epsilon * mean
        return targets

    @torch.no_grad()
    def soft_update(self, tau):
        """Move the target network a fraction tau toward the online network."""
        for target, online in zip(self.target_network.parameters(), self.network.parameters()):
            target.mul_(1 - tau).add_(online, alpha=tau)


def model_config(network):
    return {
        "format": MODEL_FORMAT,
        "feature_version": FEATURE_VERSION,
        "n_features": N_FEATURES,
        "heads": HEAD_NAMES,
        "learned_heads": [head.name for head in LEARNED_HEADS],
        "hidden": list(network.hidden),
    }


def save_model(network, path, **extra):
    """Write <stem>.pt and <stem>.json; returns the .pt path."""
    stem = Path(path).with_suffix("")
    stem.parent.mkdir(parents=True, exist_ok=True)
    torch.save(network.state_dict(), stem.with_suffix(".pt"))
    with open(stem.with_suffix(".json"), "w") as f:
        json.dump({**model_config(network), **extra}, f, indent=1)
    return stem.with_suffix(".pt")


def load_network(path):
    stem = Path(path).with_suffix("")
    with open(stem.with_suffix(".json")) as f:
        config = json.load(f)
    expected = model_config(QNetwork(hidden=config.get("hidden", ())))
    for key in ["format", "feature_version", "n_features", "heads", "learned_heads"]:
        if config.get(key) != expected[key]:
            raise ValueError(f"{stem}.pt was saved with {key}={config.get(key)!r}, but this code uses "
                             f"{expected[key]!r}; the model can't be loaded")
    network = QNetwork(hidden=config["hidden"])
    network.load_state_dict(torch.load(stem.with_suffix(".pt"), weights_only=True))
    network.eval()
    return network


class DQNAgent:
    """
    Plays from a QNetwork. temperature 0 is greedy; otherwise softmax over Q in big blinds.
    player_index is set by the caller (e.g. poker.benchmark.SeatedAgent).
    """

    def __init__(self, network, player_index=0, temperature=0.0):
        self.network = network
        self.player_index = player_index
        self.temperature = temperature

    def head_values(self, game_state):
        """Q over HEADS for the player to act (-inf where illegal)."""
        features = encode(game_state, self.player_index)[None]
        fold_value = -game_state.total_bet_by_player(self.player_index) / game_state.big_blind
        return full_q_values(self.network.learned_q(features), np.array([fold_value]),
                             legal_mask(game_state)[None])[0]

    def head_probabilities(self, game_state):
        return softmax_policy(self.head_values(game_state), self.temperature)[0]

    def action_probabilities(self, game_state):
        """{game action: probability} (used by poker.diagnostics)."""
        owed = game_state.minimum_legal_bet()
        return {game_action(head, owed): float(p) for head, p in enumerate(self.head_probabilities(game_state))
                if p > 0}

    def get_action(self, game_state, proba_random_action=0.0):
        probabilities = self.head_probabilities(game_state)
        head = int(np.random.choice(N_HEADS, p=probabilities)) if self.temperature > 0 else int(probabilities.argmax())
        return game_action(head, game_state.minimum_legal_bet())


def load_agent(path, temperature=0.0):
    return DQNAgent(load_network(path), temperature=temperature)
