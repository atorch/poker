from collections import Counter
from enum import IntEnum
from itertools import combinations
from operator import attrgetter

from poker.cards import Rank


def sort_hand(hand):

    return sorted(hand, key=attrgetter("rank", "suit"))


def is_straight(sorted_hand):

    if sorted_hand[-1].rank == Rank.ACE and sorted_hand[0].rank == Rank.TWO:

        # Note: in this case the sorted hand is [two, ... , ace]
        straight_tiebreaker = sorted_hand[-2].rank
        return is_straight(sorted_hand[:-1])[0], straight_tiebreaker

    straight_tiebreaker = sorted_hand[-1].rank

    for i, card in enumerate(sorted_hand[:-1]):

        next_card = sorted_hand[i + 1]

        if card.rank + 1 != next_card.rank:
            return False, straight_tiebreaker

    return True, straight_tiebreaker


class HandCategory(IntEnum):

    HIGH_CARD = 0
    PAIR = 1
    TWO_PAIR = 2
    THREE_OF_A_KIND = 3
    STRAIGHT = 4
    FLUSH = 5
    FULL_HOUSE = 6
    FOUR_OF_A_KIND = 7
    STRAIGHT_FLUSH = 8


def encode_strength(category, tiebreak_ranks):
    """
    Pack a hand category and its ordered tiebreak ranks into a single int.

    The encoding is lexicographic: category first, then each tiebreak rank in order
    (base 13, five slots), so comparing ints is the same as comparing
    (category, tiebreak_ranks) tuples. Two hands tie only if they are equal in poker.
    """
    padded = list(tiebreak_ranks) + [0] * (5 - len(tiebreak_ranks))
    value = int(category)
    for rank in padded:
        value = value * len(Rank) + int(rank)
    return value


def best_hand_strength(public_cards, hole_cards):

    available_cards = public_cards + hole_cards

    return max(
        (strength(candidate_hand) for candidate_hand in combinations(available_cards, 5)),
        key=lambda strength_and_description: strength_and_description[0],
    )


def strength(hand):
    """
    Return (strength, description) for a five-card hand.

    Higher strength wins; equal strength is a genuine tie (split pot).
    Tiebreaks follow standard poker rules, including kickers.
    """

    unique_suits = set(card.suit for card in hand)

    sorted_hand = sort_hand(hand)
    straight, straight_tiebreaker = is_straight(sorted_hand)
    is_flush = len(unique_suits) == 1

    # Note: ranks ordered by (count, rank) descending, e.g. two pair J J 7 7 A -> [J, 7, A]
    rank_counter = Counter(card.rank for card in hand)
    groups = sorted(rank_counter.items(), key=lambda item: (item[1], item[0]), reverse=True)
    ranks_by_group = [rank for rank, _ in groups]
    counts = [count for _, count in groups]
    high_to_low = sorted((card.rank for card in hand), reverse=True)

    if straight and is_flush:
        return encode_strength(HandCategory.STRAIGHT_FLUSH, [straight_tiebreaker]), "a straight flush"

    if counts == [4, 1]:
        quads, kicker = ranks_by_group
        return (
            encode_strength(HandCategory.FOUR_OF_A_KIND, ranks_by_group),
            f"four {quads.name}s, {kicker.name} kicker",
        )

    if counts == [3, 2]:
        trips, pair = ranks_by_group
        return (
            encode_strength(HandCategory.FULL_HOUSE, ranks_by_group),
            f"a full house ({trips.name}s full of {pair.name}s)",
        )

    if is_flush:
        return (
            encode_strength(HandCategory.FLUSH, high_to_low),
            f"a {hand[0].suit.name} flush, {high_to_low[0].name} high",
        )

    if straight:
        return (
            encode_strength(HandCategory.STRAIGHT, [straight_tiebreaker]),
            f"a straight ending in a {straight_tiebreaker.name}",
        )

    if counts == [3, 1, 1]:
        return (
            encode_strength(HandCategory.THREE_OF_A_KIND, ranks_by_group),
            f"three {ranks_by_group[0].name}s, {ranks_by_group[1].name} kicker",
        )

    if counts == [2, 2, 1]:
        higher_pair, lower_pair, kicker = ranks_by_group
        return (
            encode_strength(HandCategory.TWO_PAIR, ranks_by_group),
            f"two pair, {higher_pair.name}s and {lower_pair.name}s, {kicker.name} kicker",
        )

    if counts == [2, 1, 1, 1]:
        return (
            encode_strength(HandCategory.PAIR, ranks_by_group),
            f"a pair of {ranks_by_group[0].name}s, {ranks_by_group[1].name} kicker",
        )

    return encode_strength(HandCategory.HIGH_CARD, high_to_low), f"{high_to_low[0].name} high"
