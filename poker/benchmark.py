"""
Benchmark an agent against a fixed panel of opponents.

Metrics, all from the point of view of the agent being benchmarked:

1. Chips per deal (duplicate): each deal starts from fixed stacks. Every shuffled deck is
   played once with the agent in each seat (opponents fill the other seats), so the agent
   gets every hand and every position for the same cards. This cancels most of the card
   luck, which makes this metric much less noisy than win rates. 0 means break-even.
   The decks come from a dedicated random generator seeded by (seed, opponent), so every
   agent benchmarked with the same seed plays exactly the same decks (paired comparisons).

2. Style statistics from the same deals: VPIP, PFR, postflop aggression frequency,
   fold-to-bet, went-to-showdown and won-at-showdown (see StyleStats).

3. Bust-out win rate (optional, --games 0 skips it): play full games until someone runs out
   of chips; the agent "wins" if it ends with the most chips. 1/n_players is break-even.

4. The M1 acceptance check from TICKET_learner_redesign.md, and a behaviour fingerprint
   from poker.diagnostics (preflop hand selection and a few reference spots).

Agents and opponents are given as specs: a scripted agent name (see SCRIPTED_AGENTS) or the
path of a saved model (poker.dqn `.pt`, or an old Keras `.h5` / `.weights.h5`).

Usage:
    uv run python -m poker.benchmark                                     # default model
    uv run python -m poker.benchmark --agent maxraise                    # benchmark a baseline
    uv run python -m poker.benchmark --agent models/foo.weights.h5 --decks 5000 --games 0
    uv run python -m poker.benchmark --opponents maxraise models/old.h5  # models as opponents
    uv run python -m poker.benchmark --json results.json                 # full results as JSON
"""
import argparse
import contextlib
import io
import json
import math
import os
import random
import time
import zlib
from functools import partial

import numpy as np

from poker.baseline_agents import CallStationAgent, MaxRaiseAgent, TagAgent
from poker.cards import FULL_DECK
from poker.config import MAX_INITIAL_WEALTH, MIN_INITIAL_WEALTH, TYPICAL_INITIAL_WEALTH
from poker.consistent_random_agent import ConsistentRandomAgent
from poker.random_agent import RandomAgent
from poker.skillful_random_agent import SkillfulRandomAgent
from poker.state import GameStage, State

SCRIPTED_AGENTS = {
    "random": RandomAgent,
    "skillful": SkillfulRandomAgent,
    "consistent": ConsistentRandomAgent,
    "maxraise": MaxRaiseAgent,
    "callstation": CallStationAgent,
    "tag": TagAgent,
    "tag-calldown": partial(TagAgent, fold_postflop=False),
}
DEFAULT_PANEL = list(SCRIPTED_AGENTS)
DEFAULT_MODEL = "models/player_0_latest.weights.h5"

# M1 acceptance criteria (TICKET_learner_redesign.md), on chips per deal at stacks of 20:
#  opponent -> (description, test on (mean, 95% CI half-width))
M1_CRITERIA = {
    "maxraise": (">= +0.5", lambda mean, ci: mean >= 0.5),
    "callstation": (">= +0.5", lambda mean, ci: mean >= 0.5),
    "tag": ("CI > 0", lambda mean, ci: mean - ci > 0),
    "tag-calldown": ("CI > 0", lambda mean, ci: mean - ci > 0),
    "random": ("> 0", lambda mean, ci: mean > 0),
    "skillful": ("> 0", lambda mean, ci: mean > 0),
    "consistent": ("> 0", lambda mean, ci: mean > 0),
}


class SeatedAgent:
    """Lets one agent object (e.g. one loaded model) play from any seat."""

    def __init__(self, agent, seat):
        self.agent = agent
        self.player_index = seat

    def get_action(self, game_state):
        self.agent.player_index = self.player_index
        return self.agent.get_action(game_state, proba_random_action=0.0)

    def action_probabilities(self, game_state):
        """{action: probability} if the agent exposes its policy, otherwise None."""
        if not hasattr(self.agent, "action_probabilities"):
            return None
        self.agent.player_index = self.player_index
        return self.agent.action_probabilities(game_state)


def load_model_agent(model_path, hidden_layers=(128, 128), temperature=1.0):
    from poker.agent import Agent

    with contextlib.redirect_stdout(io.StringIO()):
        agent = Agent(player_index=0, n_players=3, hidden_layers=hidden_layers, temperature=temperature)
        loaded = agent.load_model(model_path)
    if not loaded:
        raise FileNotFoundError(f"Could not load model from {model_path}")
    return agent


def agent_factory(spec, temperature=None):
    """
    make_agent(seat) -> SeatedAgent for a spec: a scripted agent name or a saved model path
    (a poker.dqn `.pt` model, or an old Keras `.h5` / `.weights.h5` model).

    A model is loaded once and shared across seats. Callables are passed through, so code that
    already has a live agent (e.g. a trainer) can pass its own make_agent. temperature=None uses
    each model type's default: greedy for poker.dqn models, softmax at 1.0 for the old models.
    """
    if callable(spec):
        return spec
    if spec in SCRIPTED_AGENTS:
        constructor = SCRIPTED_AGENTS[spec]
        return lambda seat: SeatedAgent(constructor(player_index=seat), seat)
    if str(spec).endswith(".pt"):
        from poker.dqn import load_agent

        model_agent = load_agent(spec, temperature=0.0 if temperature is None else temperature)
    else:
        model_agent = load_model_agent(spec, temperature=1.0 if temperature is None else temperature)
    return lambda seat: SeatedAgent(model_agent, seat)


def display_name(spec):
    return os.path.basename(spec) if isinstance(spec, str) and spec not in SCRIPTED_AGENTS else str(spec)


def mean_and_ci(values, z=1.96):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return float(values.mean()) if len(values) else 0.0, float("nan")
    return float(values.mean()), z * float(values.std(ddof=1)) / math.sqrt(len(values))


def seed_for(seed, *labels):
    """A deterministic 32-bit seed for (seed, labels), independent of PYTHONHASHSEED."""
    return zlib.crc32(":".join(str(part) for part in (seed,) + labels).encode())


class StyleStats:
    """
    Standard poker style statistics for one player, accumulated over deals:
    - vpip: fraction of deals where they voluntarily put chips in preflop (call or raise)
    - pfr: fraction of deals where they raised preflop
    - afq: postflop aggression frequency, (bets + raises) / (bets + raises + calls + folds)
    - fold_to_bet: fraction of decisions facing a bet (any street) where they folded
    - wtsd: fraction of deals where they saw the flop that they took to showdown
    - wsd: fraction of their showdowns where they won at least a share of the pot
    """

    KEYS = ["vpip", "pfr", "afq", "fold_to_bet", "wtsd", "wsd"]
    LABELS = ["VPIP", "PFR", "AFq", "F2B", "WTSD", "W$SD"]

    def __init__(self):
        self.counts = {name: 0 for name in [
            "deals", "vpip", "pfr", "postflop_aggressive", "postflop_passive", "faced_bet", "folded_to_bet",
            "saw_flop", "showdowns", "won_at_showdown",
        ]}
        self._start_deal()

    def _start_deal(self):
        self._vpip = self._pfr = self._folded_preflop = False

    def record_decision(self, state, action):
        """Call before the action is applied to `state`."""
        owed = state.minimum_legal_bet()
        if owed > 0:
            self.counts["faced_bet"] += 1
            self.counts["folded_to_bet"] += action < 0

        if state.game_stage == GameStage.PRE_FLOP:
            self._vpip |= action > 0
            self._pfr |= action > owed
            self._folded_preflop |= action < 0
        elif action > owed:
            self.counts["postflop_aggressive"] += 1
        elif action < 0 or (owed > 0 and action == owed):
            self.counts["postflop_passive"] += 1

    def record_deal_end(self, state, seat):
        """Call once the deal is over (State keeps the last deal's result in last_deal_*)."""
        counts = self.counts
        counts["deals"] += 1
        counts["vpip"] += self._vpip
        counts["pfr"] += self._pfr
        counts["saw_flop"] += len(state.last_deal_public_cards) >= 3 and not self._folded_preflop
        if not state.last_deal_won_by_fold and seat not in state.last_deal_folded_players:
            counts["showdowns"] += 1
            counts["won_at_showdown"] += seat in state.last_deal_winners
        self._start_deal()

    def summary(self):
        counts = self.counts

        def ratio(numerator, denominator):
            return numerator / denominator if denominator else None

        return {
            "vpip": ratio(counts["vpip"], counts["deals"]),
            "pfr": ratio(counts["pfr"], counts["deals"]),
            "afq": ratio(counts["postflop_aggressive"], counts["postflop_aggressive"] + counts["postflop_passive"]),
            "fold_to_bet": ratio(counts["folded_to_bet"], counts["faced_bet"]),
            "wtsd": ratio(counts["showdowns"], counts["saw_flop"]),
            "wsd": ratio(counts["won_at_showdown"], counts["showdowns"]),
        }


def play_one_deal(players, deck, initial_wealth, dealer, on_decision=None):
    """
    Play a single deal from fixed stacks.

    Returns (each player's chip change, the final State). on_decision(seat, state, action) is
    called before each action is applied.
    """
    state = State(
        n_players=len(players), initial_wealth=initial_wealth, initial_dealer=dealer, deck=list(deck)
    )
    while state.n_deals == 1 and not state.terminal:
        seat = state.current_player
        action = players[seat].get_action(state)
        if on_decision is not None:
            on_decision(seat, state, action)
        state.update(action)
    return [wealth - initial_wealth for wealth in state.wealth], state


def chips_per_deal_duplicate(make_agent, make_opponent, n_decks, rng, n_players=3,
                             initial_wealth=TYPICAL_INITIAL_WEALTH, style=None):
    """
    Mean chips won per deal by the agent (and its 95% CI), averaging each deck over all agent seats.

    Decks and dealers are drawn only from `rng` (a random.Random), so they don't depend on how
    the agents use randomness. If `style` (a StyleStats) is given, the agent's decisions are
    recorded in it.
    """
    per_deck = []
    for _ in range(n_decks):
        deck = rng.sample(FULL_DECK, k=len(FULL_DECK))
        dealer = rng.randrange(n_players)
        results = []
        for agent_seat in range(n_players):
            players = [
                make_agent(seat) if seat == agent_seat else make_opponent(seat) for seat in range(n_players)
            ]

            def on_decision(seat, state, action, agent_seat=agent_seat):
                if style is not None and seat == agent_seat:
                    style.record_decision(state, action)

            chip_changes, final_state = play_one_deal(players, deck, initial_wealth, dealer, on_decision)
            if style is not None:
                style.record_deal_end(final_state, agent_seat)
            results.append(chip_changes[agent_seat])
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


def check_m1(opponent_results):
    """Evaluate the M1 criteria that apply to the opponents in `opponent_results`."""
    checks = {}
    for name, (description, test) in M1_CRITERIA.items():
        if name in opponent_results:
            result = opponent_results[name]
            checks[name] = bool(test(result["chips_per_deal"], result["chips_ci"]))
    return {
        "passed": sum(checks.values()),
        "checked": len(checks),
        "total": len(M1_CRITERIA),
        "pass": len(checks) == len(M1_CRITERIA) and all(checks.values()),
        "checks": checks,
    }


def run_benchmark(agent, opponents=DEFAULT_PANEL, n_decks=500, n_games=0, seed=0,
                  initial_wealth=TYPICAL_INITIAL_WEALTH, fingerprint=True, temperature=None, agent_name=None):
    """
    Benchmark `agent` (a spec or a make_agent callable) against each opponent spec; return a
    JSON-serializable report (format it with format_report).

    The global random / np.random states are restored afterwards, so a trainer that calls this
    periodically keeps its own random stream.
    """
    random_state, np_random_state = random.getstate(), np.random.get_state()
    try:
        return _run_benchmark(agent, opponents, n_decks, n_games, seed, initial_wealth, fingerprint,
                              temperature, agent_name)
    finally:
        random.setstate(random_state)
        np.random.set_state(np_random_state)


def _run_benchmark(agent, opponents, n_decks, n_games, seed, initial_wealth, fingerprint, temperature,
                   agent_name):
    make_agent = agent_factory(agent, temperature=temperature)
    report = {
        "agent": agent_name or display_name(agent),
        "initial_wealth": initial_wealth,
        "n_decks": n_decks,
        "n_games": n_games,
        "seed": seed,
        "opponents": {},
    }

    for opponent in opponents:
        name = opponent if isinstance(opponent, str) else str(opponent)
        make_opponent = agent_factory(opponent, temperature=temperature)
        # Note: agents' own randomness is reseeded per opponent, so results don't depend on panel order
        random.seed(seed_for(seed, name, "agents"))
        np.random.seed(seed_for(seed, name, "agents"))
        start = time.time()

        style = StyleStats()
        chips, chips_ci = chips_per_deal_duplicate(
            make_agent, make_opponent, n_decks, random.Random(seed_for(seed, name, "decks")),
            initial_wealth=initial_wealth, style=style,
        )
        result = {"chips_per_deal": chips, "chips_ci": chips_ci, "style": style.summary()}
        if n_games > 0:
            win_rate, win_ci = bust_out_win_rate(make_agent, make_opponent, n_games)
            result.update(bust_out_win_rate=win_rate, bust_out_ci=win_ci)
        result["seconds"] = time.time() - start
        report["opponents"][name] = result

    report["m1"] = check_m1(report["opponents"])

    if fingerprint:
        from poker.diagnostics import fingerprint as compute_fingerprint

        report["fingerprint"] = compute_fingerprint(make_agent)

    return report


def format_style_value(value):
    return "  -  " if value is None else f"{value:5.2f}"


def format_report(report, title=True):
    """Compact text table for a report from run_benchmark (about 15 lines)."""
    lines = []
    if title:
        games = f", {report['n_games']} bust-out games" if report["n_games"] else ""
        lines.append(f"Benchmark: {report['agent']} | stacks {report['initial_wealth']} | "
                     f"{report['n_decks']} decks x 3 seats{games} | seed {report['seed']}")

    has_bust_out = any("bust_out_win_rate" in result for result in report["opponents"].values())
    header = f"{'vs 2x':<14} {'chips/deal':>15}"
    if has_bust_out:
        header += f" {'bust-out':>14}"
    header += "  " + " ".join(f"{label:>5}" for label in StyleStats.LABELS) + "   M1"
    lines.append(header)

    for name, result in report["opponents"].items():
        line = f"{display_name(name)[:14]:<14} {result['chips_per_deal']:>+7.2f} ± {result['chips_ci']:<5.2f}"
        if has_bust_out:
            if "bust_out_win_rate" in result:
                line += f" {100 * result['bust_out_win_rate']:>6.1f}% ± {100 * result['bust_out_ci']:<4.1f}"
            else:
                line += " " * 15
        line += "  " + " ".join(format_style_value(result["style"][key]) for key in StyleStats.KEYS)
        if name in report["m1"]["checks"]:
            description = M1_CRITERIA[name][0]
            line += f"   {'pass' if report['m1']['checks'][name] else 'FAIL'} ({description})"
        lines.append(line)

    m1 = report["m1"]
    if m1["checked"]:
        verdict = "PASS" if m1["pass"] else "FAIL"
        missing = m1["total"] - m1["checked"]
        lines.append(f"M1: {verdict}, {m1['passed']} of {m1['checked']} criteria met"
                     + (f" ({missing} not run)" if missing else ""))

    if "fingerprint" in report:
        from poker.diagnostics import format_fingerprint

        lines.extend(format_fingerprint(report["fingerprint"]))
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent", default=DEFAULT_MODEL,
                        help=f"scripted agent name ({', '.join(SCRIPTED_AGENTS)}) or model path")
    parser.add_argument("--temperature", type=float, default=None,
                        help="softmax temperature for model agents (default: greedy for .pt models, 1.0 for old models)")
    parser.add_argument("--opponents", nargs="+", default=DEFAULT_PANEL,
                        help="scripted agent names and/or model paths")
    parser.add_argument("--decks", type=int, default=500, help="decks for the duplicate chips/deal metric")
    parser.add_argument("--games", type=int, default=200, help="games for the bust-out win rate (0 skips it)")
    parser.add_argument("--no-fingerprint", action="store_true", help="skip the behaviour fingerprint")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", metavar="PATH", help="also write the full report as JSON")
    args = parser.parse_args()

    report = run_benchmark(
        args.agent, opponents=args.opponents, n_decks=args.decks, n_games=args.games, seed=args.seed,
        fingerprint=not args.no_fingerprint, temperature=args.temperature,
    )
    print("\n".join(format_report(report)))
    seconds = sum(result["seconds"] for result in report["opponents"].values())
    print(f"({seconds:.0f}s)")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
