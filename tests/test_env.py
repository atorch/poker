"""
Tests for poker.env: chip conservation, deal boundaries, stacks and legal masks.
"""
import random

import numpy as np

from poker.actions import Head
from poker.env import VectorEnv, parse_pool
from poker.features import N_FEATURES, OFFSETS


def random_legal_heads(masks, rng):
    return np.array([rng.choice(np.flatnonzero(mask)) for mask in masks])


def test_parse_pool():
    assert parse_pool("callstation") == {"callstation": 1.0}
    assert parse_pool("callstation:1, maxraise:3") == {"callstation": 1.0, "maxraise": 3.0}
    pool = parse_pool("pool,tag:2")
    assert len(pool) == 7 and pool["tag"] == 3.0 and pool["random"] == 1.0


def test_set_opponents_applies_from_the_next_deal():
    env = VectorEnv(n_tables=4, opponents="callstation", seed=0)
    env.set_opponents("maxraise")
    rng = np.random.default_rng(0)
    for _ in range(100):
        env.step(random_legal_heads(env.observe()["masks"], rng))
    names = {type(player.agent).__name__ for table in env.tables for player in table.players if player is not None}
    assert names == {"MaxRaiseAgent"}


def test_observations_and_rewards():
    env = VectorEnv(n_tables=16, opponents="random,maxraise,callstation,tag", seed=0)
    rng = np.random.default_rng(0)
    finished = []
    for _ in range(200):
        batch = env.observe()
        assert batch["features"].shape == (16, N_FEATURES) and batch["masks"].any(axis=1).all()
        assert (batch["fold_values"] <= 0).all()
        # Note: folding is legal exactly when facing a bet
        owed = batch["features"][:, OFFSETS["owed"]]
        assert (batch["masks"][:, Head.FOLD] == (owed > 0)).all()
        rewards, done = env.step(random_legal_heads(batch["masks"], rng))
        finished.extend(rewards[done].tolist())
        assert (rewards[~done] == 0).all()
    assert env.deals_finished == len(finished) > 100
    # A deal can't win or lose more than the deepest stack (35 chips = 17.5 BB)
    assert max(abs(reward) for reward in finished) <= 17.5


def test_chips_are_conserved_and_stacks_honored():
    env = VectorEnv(n_tables=8, opponents="tag", stack_range=(5, 35), seed=1)
    rng = np.random.default_rng(1)
    for _ in range(100):
        for table in env.tables:
            assert all(5 <= stack <= 35 for stack in table.stacks)
            assert table.state.wealth == table.stacks  # wealth only changes when the deal ends
            assert table.state.current_player == table.seat and table.state.n_deals == 1
        # Note: finished tables start their next deal inside step(), so keep the current ones
        before = [(table.state, list(table.stacks), table.seat) for table in env.tables]
        rewards, done = env.step(random_legal_heads(env.observe()["masks"], rng))
        for i in np.flatnonzero(done):
            state, stacks, seat = before[i]
            assert sum(state.wealth) == sum(stacks)  # zero-sum deal
            assert rewards[i] * state.big_blind == state.wealth[seat] - stacks[seat]


def test_seeded_envs_are_reproducible():
    def run(seed):
        random.seed(0)
        np.random.seed(0)
        env = VectorEnv(n_tables=4, opponents="random", seed=seed)
        rng = np.random.default_rng(0)
        total = 0.0
        for _ in range(50):
            rewards, _ = env.step(random_legal_heads(env.observe()["masks"], rng))
            total += rewards.sum()
        return total

    assert run(3) == run(3)
