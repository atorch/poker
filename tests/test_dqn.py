"""
Tests for poker.dqn: Q assembly, policies, Monte-Carlo targets, training, save/load and play.
"""
import json

import numpy as np
import pytest
import torch

from poker.actions import Head, N_HEADS
from poker.benchmark import run_benchmark
from poker.dqn import (
    DealRecorder, DQNAgent, Learner, QNetwork, ReplayBuffer, epsilon_greedy, full_q_values, load_agent,
    load_network, save_model, softmax_policy,
)
from poker.features import N_FEATURES
from poker.state import State


def test_full_q_values_and_policies():
    learned = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    masks = np.array([[True, True, True, False, False]])
    q = full_q_values(learned, np.array([-0.5]), masks)
    assert q[0].tolist() == [-0.5, 1.0, 2.0, -np.inf, -np.inf]
    assert softmax_policy(q, 0)[0].tolist() == [0, 0, 1, 0, 0]
    probabilities = softmax_policy(q, 1.0)[0]
    assert probabilities[3] == probabilities[4] == 0 and probabilities.sum() == pytest.approx(1)
    assert probabilities[2] > probabilities[1] > probabilities[0]

    rng = np.random.default_rng(0)
    many_q, many_masks = np.repeat(q, 2000, axis=0), np.repeat(masks, 2000, axis=0)
    assert (epsilon_greedy(many_q, many_masks, 0.0, rng) == Head.RAISE_1).all()
    explored = epsilon_greedy(many_q, many_masks, 1.0, rng)
    assert set(explored.tolist()) == {0, 1, 2}  # only legal heads


def test_deal_recorder_assigns_the_deal_reward_to_every_non_fold_decision():
    recorder = DealRecorder(n_tables=2)
    features = np.arange(2 * N_FEATURES, dtype=np.float32).reshape(2, N_FEATURES)
    recorder.record(features, [Head.CHECK_CALL, Head.RAISE_1])
    recorder.record(features + 1, [Head.RAISE_2, Head.FOLD])
    rewards, done = np.array([3.0, -1.5]), np.array([True, True])
    rows, heads, targets = recorder.finish(rewards, done)
    assert heads == [Head.CHECK_CALL, Head.RAISE_2, Head.RAISE_1]  # table 1's fold is not stored
    assert targets == [3.0, 3.0, -1.5]
    assert np.array_equal(rows[1], features[0] + 1)
    assert recorder.finish(rewards, done) == ([], [], [])


def test_deal_recorder_keeps_unfinished_deals():
    recorder = DealRecorder(n_tables=2)
    features = np.zeros((2, N_FEATURES), dtype=np.float32)
    recorder.record(features, [Head.CHECK_CALL, Head.CHECK_CALL])
    _, heads, _ = recorder.finish(np.array([1.0, 0.0]), np.array([True, False]))
    assert len(heads) == 1 and len(recorder.pending[1]) == 1


def test_replay_buffer_wraps_around():
    buffer = ReplayBuffer(capacity=3)
    rows = np.eye(5, N_FEATURES, dtype=np.float32)
    buffer.add(rows, [1, 2, 3, 4, 1], [0.0, 1.0, 2.0, 3.0, 4.0])
    assert len(buffer) == 3 and sorted(buffer.targets.tolist()) == [2.0, 3.0, 4.0]
    features, heads, targets = buffer.sample(8, np.random.default_rng(0))
    assert features.shape == (8, N_FEATURES) and set(targets.tolist()) <= {2.0, 3.0, 4.0}


def test_train_step_fits_targets():
    torch.manual_seed(0)
    learner = Learner(hidden=(32,), learning_rate=1e-2)
    rng = np.random.default_rng(0)
    features = rng.random((64, N_FEATURES), dtype=np.float32)
    heads = rng.integers(1, N_HEADS, size=64)
    targets = (heads == Head.RAISE_1).astype(np.float32) * 2 - 1
    first = learner.train_step(features, heads, targets)
    for _ in range(200):
        last = learner.train_step(features, heads, targets)
    assert last < 0.05 * first
    with pytest.raises(AssertionError):
        learner.train_step(features[:1], [Head.FOLD], targets[:1])


def test_save_and_load_round_trip(tmp_path):
    network = QNetwork(hidden=(16, 8))
    path = save_model(network, tmp_path / "model", step=5)
    assert path.suffix == ".pt" and json.loads(path.with_suffix(".json").read_text())["step"] == 5
    loaded = load_network(path)
    x = torch.randn(4, N_FEATURES)
    assert torch.equal(network(x), loaded(x))


def test_incompatible_models_fail_loudly(tmp_path):
    path = save_model(QNetwork(hidden=(8,)), tmp_path / "model")
    config = json.loads(path.with_suffix(".json").read_text())
    config["feature_version"] = -1
    path.with_suffix(".json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="feature_version"):
        load_network(path)


def test_agent_plays_legal_actions_from_a_saved_model(tmp_path):
    torch.manual_seed(0)
    path = save_model(QNetwork(hidden=(16,)), tmp_path / "model")
    agent = load_agent(path)
    state = State(n_players=3, initial_wealth=20, initial_dealer=0)
    assert agent.get_action(state) in state.legal_actions()
    assert sum(agent.action_probabilities(state).values()) == pytest.approx(1)
    assert DQNAgent(agent.network, temperature=0.5).get_action(state) in state.legal_actions()

    report = run_benchmark(str(path), opponents=["callstation", "tag"], n_decks=5)
    assert report["agent"] == "model.pt" and "fingerprint" in report


def batch_for(features, masks, fold_values):
    return {"features": features, "masks": masks, "fold_values": fold_values}


def test_transition_recorder_links_decisions_within_a_deal():
    from poker.dqn import TransitionRecorder

    recorder = TransitionRecorder(n_tables=2)
    first = np.zeros((2, N_FEATURES), dtype=np.float32)
    masks = np.ones((2, N_HEADS), dtype=bool)
    assert recorder.complete(batch_for(first, masks, np.zeros(2))) == []
    recorder.record(first, [Head.CHECK_CALL, Head.FOLD])
    # Table 0's deal goes on; table 1 folded, so its deal ends with nothing stored
    assert recorder.finish(np.array([0.0, -1.0]), np.array([False, True])) == []

    second = np.ones((2, N_FEATURES), dtype=np.float32)
    transitions = recorder.complete(batch_for(second, masks, np.array([-1.5, 0.0])))
    assert len(transitions) == 1 and transitions[0]["heads"] == Head.CHECK_CALL
    assert transitions[0]["rewards"] == 0 and not transitions[0]["dones"]
    assert transitions[0]["next_fold_values"] == -1.5 and transitions[0]["next_features"][0] == 1

    recorder.record(second, [Head.RAISE_1, Head.CHECK_CALL])
    transitions = recorder.finish(np.array([4.0, 0.0]), np.array([True, False]))
    assert len(transitions) == 1 and transitions[0]["rewards"] == 4.0 and transitions[0]["dones"]


def test_expected_sarsa_targets():
    torch.manual_seed(0)
    learner = Learner(hidden=(8,))
    next_features = np.random.default_rng(0).random((3, N_FEATURES), dtype=np.float32)
    masks = np.array([[True, True, True, False, False], [False, True, False, False, False], [False] * 5])
    batch = {"rewards": np.array([0.0, 0.0, 2.5]), "next_features": next_features, "next_masks": masks,
             "next_fold_values": np.array([-1.0, 0.0, 0.0]), "dones": np.array([False, False, True])}
    q = full_q_values(learner.target_network.learned_q(next_features), batch["next_fold_values"], masks)

    greedy = learner.expected_sarsa_targets(batch, epsilon=0.0)
    assert greedy[0] == pytest.approx(q[0].max()) and greedy[1] == pytest.approx(q[1, Head.CHECK_CALL])
    assert greedy[2] == 2.5  # terminal: just the reward
    uniform = learner.expected_sarsa_targets(batch, epsilon=1.0)
    assert uniform[0] == pytest.approx(q[0, :3].mean())


def test_soft_update_moves_the_target_network():
    learner = Learner(hidden=(8,))
    with torch.no_grad():
        for parameter in learner.network.parameters():
            parameter.add_(1.0)
    before = [p.clone() for p in learner.target_network.parameters()]
    learner.soft_update(0.25)
    for old, new, online in zip(before, learner.target_network.parameters(), learner.network.parameters()):
        assert torch.allclose(new, 0.75 * old + 0.25 * online)
