"""
Behaviour fingerprint of an agent: a handful of numbers that say whether it uses its cards.

1. Preflop hand selection: the agent's action probabilities for each of the 169 starting-hand
   classes when it is first to act preflop (the dealer, facing the $2 big blind, stacks of 20).
   Summarized as rank correlations with preflop equity against 2 random hands: P(raise) should
   rise with equity and P(fold) should fall. Rank correlations only measure order (P(raise) going
   from 0.73 to 0.78 across all hands still correlates well), so the size of the effect is also
   reported as a spread: P(raise) for the top 20% of hands (by equity, weighted by combinations)
   minus P(raise) for the bottom 50%, and P(fold) for the bottom 50% minus the top 20%. Plus the
   combo-weighted P(raise) and P(fold): how often it opens with a raise or folds when first in.
2. A few fixed spots on the flop (REFERENCE_SPOTS), e.g. top pair versus nothing, first to act
   and facing a bet.

Agents that expose action_probabilities(state) are read exactly; others are sampled.
make_agent is a factory seat -> agent, as in poker.benchmark.
"""
import random
from collections import Counter

import numpy as np

from poker.cards import parse_cards, stacked_deck
from poker.config import TYPICAL_INITIAL_WEALTH
from poker.preflop_equity import HAND_CLASSES, class_combos, preflop_equity, representative_hole_cards
from poker.state import GameStage, State

NAMED_HANDS = ["AA", "AKs", "T9s", "72o"]

# Dealer at seat 0, so seat 1 posts the small blind and acts first after the flop.
#  Preflop: the dealer calls $2, the small blind completes, the big blind checks.
LIMPED_PREFLOP = [2, 1, 0]
REFERENCE_SPOTS = [
    {"name": "top pair, first to act", "seat": 1, "hole": "As Kd", "flop": "Ah 7c 2d", "actions": LIMPED_PREFLOP},
    {"name": "nothing, first to act", "seat": 1, "hole": "9s 8d", "flop": "Ah Kc 2d", "actions": LIMPED_PREFLOP},
    {"name": "top pair, facing $2 bet", "seat": 2, "hole": "As Kd", "flop": "Ah 7c 2d",
     "actions": LIMPED_PREFLOP + [2]},
    {"name": "nothing, facing $2 bet", "seat": 2, "hole": "9s 8d", "flop": "Ah Kc 2d",
     "actions": LIMPED_PREFLOP + [2]},
]


def _ranks(values):
    """Ranks with ties averaged (for Spearman correlation)."""
    values = np.asarray(values, dtype=float)
    ranks = np.empty(len(values))
    ranks[values.argsort(kind="mergesort")] = np.arange(len(values))
    for value in np.unique(values):
        tied = values == value
        ranks[tied] = ranks[tied].mean()
    return ranks


def spearman(x, y):
    """Spearman rank correlation, or None if either input is constant."""
    rank_x, rank_y = _ranks(x), _ranks(y)
    if rank_x.std() == 0 or rank_y.std() == 0:
        return None
    return float(np.corrcoef(rank_x, rank_y)[0, 1])


def action_distribution(make_agent, seat, state, n_samples=50):
    """{action: probability} for the agent in `seat` at `state` (sampled if the policy isn't exposed)."""
    agent = make_agent(seat)
    probabilities = agent.action_probabilities(state) if hasattr(agent, "action_probabilities") else None
    if probabilities is not None:
        return probabilities
    # Note: a fresh agent per sample, because some scripted agents commit to a style per deal
    counts = Counter(make_agent(seat).get_action(state) for _ in range(n_samples))
    return {action: count / n_samples for action, count in counts.items()}


def fold_call_raise(probabilities, owed):
    """Collapse {action: probability} into P(fold), P(check or call), P(bet or raise)."""
    fold = sum(p for action, p in probabilities.items() if action < 0)
    raise_ = sum(p for action, p in probabilities.items() if action > owed)
    return {"fold": fold, "call": max(0.0, 1.0 - fold - raise_), "raise": raise_}


def first_to_act_state(hole_cards, rng, initial_wealth=TYPICAL_INITIAL_WEALTH):
    """A fresh deal with `hole_cards` dealt to seat 0, the dealer, who acts first preflop."""
    deck = stacked_deck([hole_cards, None, None], rng=rng)
    return State(n_players=3, initial_wealth=initial_wealth, initial_dealer=0, deck=deck)


def spot_state(spot, rng, initial_wealth=TYPICAL_INITIAL_WEALTH):
    holes = [None, None, None]
    holes[spot["seat"]] = parse_cards(spot["hole"])
    deck = stacked_deck(holes, parse_cards(spot["flop"]), rng=rng)
    state = State(n_players=3, initial_wealth=initial_wealth, initial_dealer=0, deck=deck)
    for action in spot["actions"]:
        state.update(action)
    assert state.game_stage == GameStage.FLOP and state.current_player == spot["seat"], spot["name"]
    return state


def preflop_profile(make_agent, n_samples=50, seed=0):
    """{hand class: {"fold", "call", "raise"}} when first to act preflop."""
    rng = random.Random(seed)
    profile = {}
    for name in HAND_CLASSES:
        state = first_to_act_state(representative_hole_cards(name), rng)
        profile[name] = fold_call_raise(action_distribution(make_agent, 0, state, n_samples),
                                        state.minimum_legal_bet())
    return profile


def equity_groups(equity, weights, top_fraction=0.2, bottom_fraction=0.5):
    """
    Boolean masks over hand classes: the classes that start within the top_fraction (and the
    bottom_fraction) of combos when sorted by equity.
    """
    equity = np.asarray(equity, dtype=float)

    def first_fraction(order, fraction):
        combos_before = np.cumsum(weights[order]) - weights[order]
        mask = np.zeros(len(equity), dtype=bool)
        mask[order[combos_before < fraction - 1e-9]] = True
        return mask

    return (first_fraction(np.argsort(-equity, kind="mergesort"), top_fraction),
            first_fraction(np.argsort(equity, kind="mergesort"), bottom_fraction))


def fingerprint(make_agent, n_samples=50, seed=0):
    """JSON-serializable behaviour fingerprint (format it with format_fingerprint)."""
    profile = preflop_profile(make_agent, n_samples=n_samples, seed=seed)
    equity = [preflop_equity(name, n_opponents=2) for name in HAND_CLASSES]
    weights = np.array([class_combos(name) for name in HAND_CLASSES], dtype=float)
    weights /= weights.sum()

    def column(key):
        return np.array([profile[name][key] for name in HAND_CLASSES])

    top, bottom = equity_groups(equity, weights)

    def group_mean(key, mask):
        return float(weights[mask] @ column(key)[mask] / weights[mask].sum())

    rng = random.Random(seed)
    spots = {}
    for spot in REFERENCE_SPOTS:
        state = spot_state(spot, rng)
        spots[spot["name"]] = fold_call_raise(
            action_distribution(make_agent, spot["seat"], state, n_samples), state.minimum_legal_bet()
        )

    return {
        "preflop": {
            "raise_vs_equity": spearman(column("raise"), equity),
            "fold_vs_equity": spearman(column("fold"), equity),
            "raise_spread": group_mean("raise", top) - group_mean("raise", bottom),
            "fold_spread": group_mean("fold", bottom) - group_mean("fold", top),
            "p_raise": float(weights @ column("raise")),
            "p_fold": float(weights @ column("fold")),
            "by_class": {name: [round(profile[name][key], 4) for key in ("fold", "call", "raise")]
                         for name in HAND_CLASSES},
        },
        "spots": spots,
    }


def _format_correlation(value):
    return "n/a" if value is None else f"{value:+.2f}"


def _format_fcr(probabilities):
    return "/".join(f"{probabilities[key]:.2f}" for key in ("fold", "call", "raise"))


def format_fingerprint(fp):
    """Compact text (5 lines) for a fingerprint."""
    preflop = fp["preflop"]
    hands = "  ".join(
        f"{name} {_format_fcr(dict(zip(('fold', 'call', 'raise'), preflop['by_class'][name])))}"
        for name in NAMED_HANDS
    )
    spots = [f"{name}: {_format_fcr(probabilities)}" for name, probabilities in fp["spots"].items()]
    return [
        f"Preflop first in (fold/call/raise): P(raise) {preflop['p_raise']:.2f}, P(fold) {preflop['p_fold']:.2f};"
        f" rank corr with equity: raise {_format_correlation(preflop['raise_vs_equity'])},"
        f" fold {_format_correlation(preflop['fold_vs_equity'])};"
        f" spread (top 20% vs bottom 50%): raise {preflop['raise_spread']:+.2f}, fold {preflop['fold_spread']:+.2f}",
        f"  {hands}",
        "Flop spots (fold/call/raise; call includes check):",
    ] + [f"  {'   '.join(spots[i:i + 2])}" for i in range(0, len(spots), 2)]
