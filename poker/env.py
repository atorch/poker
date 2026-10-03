"""
A vectorized single-deal environment for training a learned agent against fixed opponents.

Each table plays one deal at a time: every player's stack is drawn uniformly from stack_range,
the dealer and the learner's seat are random, and each other seat gets an opponent drawn from
the opponent pool (scripted agent names or saved model paths, as in poker.benchmark). The
learner's pending decisions are batched across tables:

    env = VectorEnv(n_tables=256, opponents={"callstation": 1.0}, seed=0)
    batch = env.observe()               # features, legal masks and fold values, one row per table
    rewards, done = env.step(heads)     # one head (poker.actions.Head) per table

A deal ends when the hand is over, and its reward is the learner's chip change in big blinds.
Finished tables start their next deal immediately, so observe() always has one pending decision
per table. Deals where the learner never gets to act (everyone folds to its big blind) are
skipped.
"""
import random

import numpy as np

from poker.actions import N_HEADS, game_action, legal_mask
from poker.benchmark import SCRIPTED_AGENTS, agent_factory
from poker.cards import FULL_DECK
from poker.config import MAX_INITIAL_WEALTH, MIN_INITIAL_WEALTH
from poker.features import N_FEATURES, encode
from poker.state import State


# Note: "pool" stands for every scripted agent, equally weighted
POOL_ALIASES = {"pool": list(SCRIPTED_AGENTS)}


def parse_pool(text):
    """'callstation', 'callstation:1,maxraise:2' or 'pool' -> {name: weight}."""
    pool = {}
    for item in text.split(","):
        name, _, weight = item.strip().partition(":")
        for member in POOL_ALIASES.get(name, [name]):
            pool[member] = pool.get(member, 0.0) + (float(weight) if weight else 1.0)
    return pool


class Table:
    """One table: a deal in progress, with the learner in `seat`."""

    def __init__(self, opponent_factories, weights, stack_range, n_players, rng):
        self.opponent_factories = opponent_factories
        self.weights = weights
        self.stack_range = stack_range
        self.n_players = n_players
        self.rng = rng
        self.skipped_deals = 0

    def deal_is_over(self):
        return self.state.n_deals > 1 or self.state.terminal

    def new_deal(self):
        """Start deals until the learner has a decision to make."""
        while True:
            self.stacks = [self.rng.randint(*self.stack_range) for _ in range(self.n_players)]
            self.seat = self.rng.randrange(self.n_players)
            self.players = [
                None if seat == self.seat else self.rng.choices(self.opponent_factories, self.weights)[0](seat)
                for seat in range(self.n_players)
            ]
            self.state = State(
                n_players=self.n_players, initial_wealth=self.stacks, initial_dealer=self.rng.randrange(self.n_players),
                deck=self.rng.sample(FULL_DECK, k=len(FULL_DECK)),
            )
            self._play_opponents()
            if not self.deal_is_over():
                return
            self.skipped_deals += 1

    def _play_opponents(self):
        while not self.deal_is_over() and self.state.current_player != self.seat:
            self.state.update(self.players[self.state.current_player].get_action(self.state))

    def fold_value(self):
        """Q(fold) in big blinds: the chips the learner has already put in this deal are lost."""
        return -self.state.total_bet_by_player(self.seat) / self.state.big_blind

    def apply(self, head):
        """Apply the learner's head; returns the deal's reward (big blinds) if it ended, else None."""
        state = self.state
        assert legal_mask(state)[head], f"illegal head {head}"
        state.update(game_action(head, state.minimum_legal_bet()))
        self._play_opponents()
        if self.deal_is_over():
            return (self.state.wealth[self.seat] - self.stacks[self.seat]) / self.state.big_blind
        return None


class VectorEnv:
    def __init__(self, n_tables, opponents, stack_range=(MIN_INITIAL_WEALTH, MAX_INITIAL_WEALTH), n_players=3,
                 seed=0):
        self.rng = random.Random(seed)
        self.tables = [Table(None, None, stack_range, n_players, random.Random(self.rng.random()))
                       for _ in range(n_tables)]
        self.set_opponents(opponents)
        for table in self.tables:
            table.new_deal()
        self.deals_finished = 0

    def set_opponents(self, opponents):
        """Switch the opponent pool (a dict or a parse_pool string); deals in progress finish as they are."""
        if isinstance(opponents, str):
            opponents = parse_pool(opponents)
        self.opponents = dict(opponents)
        factories = [agent_factory(spec) for spec in self.opponents]
        weights = list(self.opponents.values())
        for table in self.tables:
            table.opponent_factories, table.weights = factories, weights

    @property
    def n_tables(self):
        return len(self.tables)

    @property
    def skipped_deals(self):
        return sum(table.skipped_deals for table in self.tables)

    def observe(self):
        """{"features": (n, N_FEATURES) float32, "masks": (n, N_HEADS) bool, "fold_values": (n,) float32}."""
        features = np.zeros((self.n_tables, N_FEATURES), dtype=np.float32)
        masks = np.zeros((self.n_tables, N_HEADS), dtype=bool)
        fold_values = np.zeros(self.n_tables, dtype=np.float32)
        for i, table in enumerate(self.tables):
            encode(table.state, table.seat, out=features[i])
            masks[i] = legal_mask(table.state)
            fold_values[i] = table.fold_value()
        return {"features": features, "masks": masks, "fold_values": fold_values}

    def step(self, heads):
        """Apply one head per table. Returns (rewards, done): rewards are 0 where the deal goes on."""
        rewards = np.zeros(self.n_tables, dtype=np.float32)
        done = np.zeros(self.n_tables, dtype=bool)
        for i, (table, head) in enumerate(zip(self.tables, heads)):
            reward = table.apply(int(head))
            if reward is not None:
                rewards[i], done[i] = reward, True
                self.deals_finished += 1
                table.new_deal()
        return rewards, done
