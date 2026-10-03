"""
Benchmark an agent against a fixed panel of opponents.

Two metrics, both from the point of view of the agent being benchmarked:

1. Chips per deal (duplicate): each deal starts from fixed stacks. Every shuffled deck is
   played once with the agent in each seat (opponents fill the other seats), so the agent
   gets every hand and every position for the same cards. This cancels most of the card
   luck, which makes this metric much less noisy than win rates. 0 means break-even.

2. Bust-out win rate: play full games until someone runs out of chips; the agent "wins" if
   it ends with the most chips (the metric used during training). 1/n_players is break-even.

Usage:
    uv run python -m poker.benchmark                                  # latest trained model
    uv run python -m poker.benchmark --agent maxraise                 # benchmark a baseline
    uv run python -m poker.benchmark --model models/foo.weights.h5 --decks 1000 --games 300
"""
import argparse
import contextlib
import io
import math
import random
import time

import numpy as np

from poker.baseline_agents import CallStationAgent, MaxRaiseAgent
from poker.cards import FULL_DECK
from poker.config import MAX_INITIAL_WEALTH, MIN_INITIAL_WEALTH, TYPICAL_INITIAL_WEALTH
from poker.consistent_random_agent import ConsistentRandomAgent
from poker.random_agent import RandomAgent
from poker.skillful_random_agent import SkillfulRandomAgent
from poker.state import State

OPPONENTS = {
    "random": RandomAgent,
    "skillful": SkillfulRandomAgent,
    "consistent": ConsistentRandomAgent,
    "maxraise": MaxRaiseAgent,
    "callstation": CallStationAgent,
}


class SeatedAgent:
    """Lets one agent object (e.g. one loaded model) play from any seat."""

    def __init__(self, agent, seat):
        self.agent = agent
        self.player_index = seat

    def get_action(self, game_state):
        self.agent.player_index = self.player_index
        return self.agent.get_action(game_state, proba_random_action=0.0)


def mean_and_ci(values, z=1.96):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return float(values.mean()) if len(values) else 0.0, float("nan")
    return float(values.mean()), z * float(values.std(ddof=1)) / math.sqrt(len(values))


def play_one_deal(players, deck, initial_wealth, dealer):
    """Play a single deal from fixed stacks; return each player's chip change."""
    state = State(
        n_players=len(players), initial_wealth=initial_wealth, initial_dealer=dealer, deck=list(deck)
    )
    while state.n_deals == 1 and not state.terminal:
        state.update(players[state.current_player].get_action(state))
    return [wealth - initial_wealth for wealth in state.wealth]


def chips_per_deal_duplicate(make_agent, make_opponent, n_decks, n_players=3, initial_wealth=TYPICAL_INITIAL_WEALTH):
    """Mean chips won per deal by the agent, averaging each deck over all agent seats."""
    per_deck = []
    for _ in range(n_decks):
        deck = random.sample(FULL_DECK, k=len(FULL_DECK))
        dealer = random.randrange(n_players)
        results = []
        for agent_seat in range(n_players):
            players = [
                make_agent(seat) if seat == agent_seat else make_opponent(seat) for seat in range(n_players)
            ]
            results.append(play_one_deal(players, deck, initial_wealth, dealer)[agent_seat])
        per_deck.append(np.mean(results))
    return mean_and_ci(per_deck)


def bust_out_win_rate(make_agent, make_opponent, n_games, n_players=3, max_deals=500):
    """Fraction of full games (until someone busts) the agent finishes with the most chips."""
    wins = []
    for _ in range(n_games):
        agent_seat = random.randrange(n_players)
        players = [make_agent(seat) if seat == agent_seat else make_opponent(seat) for seat in range(n_players)]
        state = State(
            n_players=n_players,
            initial_wealth=np.random.randint(MIN_INITIAL_WEALTH, MAX_INITIAL_WEALTH + 1),
            initial_dealer=random.randrange(n_players),
        )
        while not state.terminal and state.n_deals < max_deals:
            state.update(players[state.current_player].get_action(state))
        wins.append(float(int(np.argmax(state.wealth)) == agent_seat))
    return mean_and_ci(wins)


def load_model_agent(model_path, hidden_layers=(128, 128), temperature=1.0):
    from poker.agent import Agent

    with contextlib.redirect_stdout(io.StringIO()):
        agent = Agent(player_index=0, n_players=3, hidden_layers=hidden_layers, temperature=temperature)
        loaded = agent.load_model(model_path)
    if not loaded:
        raise FileNotFoundError(f"Could not load model from {model_path}")
    return agent


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent", default="model", choices=["model"] + sorted(OPPONENTS))
    parser.add_argument("--model", default="models/player_0_latest.weights.h5")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--opponents", nargs="+", default=list(OPPONENTS), choices=sorted(OPPONENTS))
    parser.add_argument("--decks", type=int, default=500, help="decks for the duplicate chips/deal metric")
    parser.add_argument("--games", type=int, default=200, help="games for the bust-out win rate metric")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.agent == "model":
        model_agent = load_model_agent(args.model, temperature=args.temperature)
        make_agent = lambda seat: SeatedAgent(model_agent, seat)
        agent_name = args.model
    else:
        make_agent = lambda seat: SeatedAgent(OPPONENTS[args.agent](player_index=seat), seat)
        agent_name = args.agent

    print(f"Benchmarking {agent_name} (stacks of {TYPICAL_INITIAL_WEALTH} for chips/deal; "
          f"{args.decks} decks x 3 seats, {args.games} bust-out games)")
    print(f"{'vs 2x':<12} {'chips/deal (duplicate)':>24} {'bust-out win rate':>20} {'time':>7}")
    for opponent_name in args.opponents:
        make_opponent = lambda seat: SeatedAgent(OPPONENTS[opponent_name](player_index=seat), seat)
        start = time.time()
        chips, chips_ci = chips_per_deal_duplicate(make_agent, make_opponent, args.decks)
        win_rate, win_ci = bust_out_win_rate(make_agent, make_opponent, args.games)
        print(f"{opponent_name:<12} {chips:>+15.2f} ± {chips_ci:<6.2f} {100 * win_rate:>12.1f}% ± {100 * win_ci:<4.1f}"
              f" {time.time() - start:>6.0f}s")


if __name__ == "__main__":
    main()
