"""
Preflop equity of each of the 169 starting-hand classes against 1 and 2 random hands.

Equity is the expected share of the pot at showdown when nobody folds (ties split the pot),
estimated by Monte Carlo and stored in poker/data/preflop_equity.json. Regenerate with:

    uv run python -m poker.preflop_equity --rollouts 20000

Hand classes are written like "AA", "AKs" (suited) and "AKo" (offsuit), high card first.
"""
import argparse
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from functools import lru_cache

from poker.cards import FULL_DECK, RANK_SYMBOLS, Card, Rank, Suit
from poker.hands import best_hand_strength

TABLE_PATH = os.path.join(os.path.dirname(__file__), "data", "preflop_equity.json")

_RANKS_HIGH_TO_LOW = list(reversed(Rank))

# Pairs AA..22, then suited AKs..32s, then offsuit AKo..32o
HAND_CLASSES = (
    [RANK_SYMBOLS[rank] * 2 for rank in _RANKS_HIGH_TO_LOW]
    + [RANK_SYMBOLS[high] + RANK_SYMBOLS[low] + "s" for high in _RANKS_HIGH_TO_LOW for low in _RANKS_HIGH_TO_LOW
       if high > low]
    + [RANK_SYMBOLS[high] + RANK_SYMBOLS[low] + "o" for high in _RANKS_HIGH_TO_LOW for low in _RANKS_HIGH_TO_LOW
       if high > low]
)


def hand_class(hole_cards):
    """The starting-hand class of two hole cards, e.g. "AKs"."""
    high, low = sorted(hole_cards, key=lambda card: card.rank, reverse=True)
    ranks = RANK_SYMBOLS[high.rank] + RANK_SYMBOLS[low.rank]
    if high.rank == low.rank:
        return ranks
    return ranks + ("s" if high.suit == low.suit else "o")


def class_combos(hand_class_name):
    """Number of two-card combinations in the class (6 for pairs, 4 suited, 12 offsuit)."""
    if len(hand_class_name) == 2:
        return 6
    return 4 if hand_class_name.endswith("s") else 12


def representative_hole_cards(hand_class_name):
    """One concrete pair of hole cards in the class."""
    high = Rank(RANK_SYMBOLS.index(hand_class_name[0]))
    low = Rank(RANK_SYMBOLS.index(hand_class_name[1]))
    second_suit = Suit.HEARTS if hand_class_name.endswith("s") else Suit.SPADES
    return [Card(high, Suit.HEARTS), Card(low, second_suit)]


@lru_cache(maxsize=1)
def load_table(path=TABLE_PATH):
    with open(path) as f:
        return json.load(f)


def preflop_equity(hole_cards_or_class, n_opponents=2):
    """Equity against n_opponents (1 or 2) random hands, from the stored table."""
    name = hole_cards_or_class if isinstance(hole_cards_or_class, str) else hand_class(hole_cards_or_class)
    return load_table()["equity"][str(n_opponents)][name]


def estimate_equity(hand_class_name, n_opponents, n_rollouts, seed):
    """Monte-Carlo equity of a hand class against n_opponents random hands."""
    rng = random.Random(f"{seed}:{hand_class_name}:{n_opponents}")
    hole = representative_hole_cards(hand_class_name)
    rest = [card for card in FULL_DECK if card not in hole]
    total_share = 0.0
    for _ in range(n_rollouts):
        dealt = rng.sample(rest, 2 * n_opponents + 5)
        board = dealt[-5:]
        my_strength = best_hand_strength(board, hole)[0]
        opponent_strengths = [best_hand_strength(board, dealt[2 * i: 2 * i + 2])[0] for i in range(n_opponents)]
        best = max(opponent_strengths)
        if my_strength > best:
            total_share += 1.0
        elif my_strength == best:
            total_share += 1.0 / (1 + opponent_strengths.count(best))
    return total_share / n_rollouts


def _estimate_task(args):
    return args[:2], estimate_equity(*args)


def compute_table(n_rollouts, seed=0, processes=None, opponent_counts=(1, 2)):
    tasks = [(name, n_opponents, n_rollouts, seed) for n_opponents in opponent_counts for name in HAND_CLASSES]
    equity = {str(n): {} for n in opponent_counts}
    with ProcessPoolExecutor(max_workers=processes) as pool:
        for (name, n_opponents), value in pool.map(_estimate_task, tasks, chunksize=4):
            equity[str(n_opponents)][name] = round(value, 4)
    return {
        "meta": {
            "description": "Preflop equity (expected pot share at showdown, ties split) against 1 and 2 "
                           "random hands, by starting-hand class",
            "rollouts_per_class": n_rollouts,
            "seed": seed,
            "generated": date.today().isoformat(),
        },
        "equity": equity,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rollouts", type=int, default=20000, help="Monte-Carlo rollouts per hand class")
    parser.add_argument("--processes", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=TABLE_PATH)
    args = parser.parse_args()

    table = compute_table(args.rollouts, seed=args.seed, processes=args.processes)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(table, f, indent=1)
    heads_up, three_way = table["equity"]["1"], table["equity"]["2"]
    print(f"Wrote {args.out}: AA {heads_up['AA']:.3f} / {three_way['AA']:.3f}, "
          f"72o {heads_up['72o']:.3f} / {three_way['72o']:.3f} (vs 1 / 2 random hands)")


if __name__ == "__main__":
    main()
