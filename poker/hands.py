from collections import Counter
from enum import IntEnum
from itertools import combinations
from operator import attrgetter

from poker.cards import Rank


def sort_hand(hand):

    return sorted(hand, key=attrgetter("rank", "suit"))


def is_straight(sorted_hand):

    ranks = [card.rank for card in sorted_hand]

    # Note: the ace plays low only in the wheel (A 2 3 4 5), which ranks as a five-high straight.
    #  (This used to strip the ace and recurse, which also stripped a second or third ace, so
    #  A A 2 3 4 and A A A 2 3 were scored as ace-high straights.)
    if ranks == [Rank.TWO, Rank.THREE, Rank.FOUR, Rank.FIVE, Rank.ACE]:
        return True, Rank.FIVE

    straight_tiebreaker = ranks[-1]

    return all(rank + 1 == next_rank for rank, next_rank in zip(ranks, ranks[1:])), straight_tiebreaker


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


def hand_category(strength_value):
    """The HandCategory of a strength value produced by encode_strength."""

    return HandCategory(strength_value // len(Rank) ** 5)


def leading_rank(strength_value):
    """The first tiebreak rank of a strength value (e.g. the rank of the pair for PAIR)."""

    return Rank(strength_value // len(Rank) ** 4 % len(Rank))


def made_hand_category(cards):
    """
    The best HandCategory that any five of `cards` make (fewer than five cards: pairs,
    two pair, trips and quads only, since straights and flushes need five cards).
    """

    if len(cards) >= 5:
        return hand_category(best_hand_strength(list(cards), [])[0])

    counts = sorted(Counter(card.rank for card in cards).values(), reverse=True) + [0, 0]
    if counts[0] == 4:
        return HandCategory.FOUR_OF_A_KIND
    if counts[0] == 3:
        return HandCategory.THREE_OF_A_KIND
    if counts[0] == 2 and counts[1] == 2:
        return HandCategory.TWO_PAIR
    if counts[0] == 2:
        return HandCategory.PAIR
    return HandCategory.HIGH_CARD


def best_hand_strength(public_cards, hole_cards):
    """
    (strength, description) of the best five-card hand among the cards (five to seven of them).

    Computed directly from rank and suit counts, which is much faster than evaluating every
    five-card combination; best_hand_strength_by_combinations is the (slow) reference.
    """

    return _best_hand(list(public_cards) + list(hole_cards))


def best_hand_strength_by_combinations(public_cards, hole_cards):

    available_cards = list(public_cards) + list(hole_cards)

    return max(
        (strength(candidate_hand) for candidate_hand in combinations(available_cards, 5)),
        key=lambda strength_and_description: strength_and_description[0],
    )


def _highest_straight(rank_set):
    """The top rank of the highest straight in a set of ranks (FIVE for the wheel), or None."""

    for high in range(Rank.ACE, Rank.SIX - 1, -1):
        if all(high - offset in rank_set for offset in range(5)):
            return Rank(high)
    if {Rank.ACE, Rank.TWO, Rank.THREE, Rank.FOUR, Rank.FIVE} <= rank_set:
        return Rank.FIVE
    return None


def _best_hand(cards):

    suit_counter = Counter(card.suit for card in cards)
    flush_suit, flush_count = suit_counter.most_common(1)[0]
    flush_ranks = sorted((card.rank for card in cards if card.suit == flush_suit), reverse=True) \
        if flush_count >= 5 else None

    if flush_ranks is not None:
        straight_flush_high = _highest_straight(set(flush_ranks))
        if straight_flush_high is not None:
            return encode_strength(HandCategory.STRAIGHT_FLUSH, [straight_flush_high]), "a straight flush"

    rank_counter = Counter(card.rank for card in cards)
    # Note: ranks ordered by (count, rank) descending, as in strength()
    by_group = sorted(rank_counter, key=lambda rank: (rank_counter[rank], rank), reverse=True)
    quads = [rank for rank in by_group if rank_counter[rank] == 4]
    trips = [rank for rank in by_group if rank_counter[rank] == 3]
    pairs = [rank for rank in by_group if rank_counter[rank] == 2]

    def kickers(excluded, n):
        return sorted((rank for rank in rank_counter if rank not in excluded), reverse=True)[:n]

    if quads:
        kicker = kickers(quads[:1], 1)[0]
        return (
            encode_strength(HandCategory.FOUR_OF_A_KIND, [quads[0], kicker]),
            f"four {quads[0].name}s, {kicker.name} kicker",
        )

    if trips and (len(trips) > 1 or pairs):
        pair = max(trips[1:] + pairs)
        return (
            encode_strength(HandCategory.FULL_HOUSE, [trips[0], pair]),
            f"a full house ({trips[0].name}s full of {pair.name}s)",
        )

    if flush_ranks is not None:
        return (
            encode_strength(HandCategory.FLUSH, flush_ranks[:5]),
            f"a {flush_suit.name} flush, {flush_ranks[0].name} high",
        )

    straight_high = _highest_straight(set(rank_counter))
    if straight_high is not None:
        return (
            encode_strength(HandCategory.STRAIGHT, [straight_high]),
            f"a straight ending in a {straight_high.name}",
        )

    if trips:
        first, second = kickers(trips[:1], 2)
        return (
            encode_strength(HandCategory.THREE_OF_A_KIND, [trips[0], first, second]),
            f"three {trips[0].name}s, {first.name} kicker",
        )

    if len(pairs) >= 2:
        higher_pair, lower_pair = pairs[:2]
        kicker = kickers([higher_pair, lower_pair], 1)[0]
        return (
            encode_strength(HandCategory.TWO_PAIR, [higher_pair, lower_pair, kicker]),
            f"two pair, {higher_pair.name}s and {lower_pair.name}s, {kicker.name} kicker",
        )

    if pairs:
        first, second, third = kickers(pairs[:1], 3)
        return (
            encode_strength(HandCategory.PAIR, [pairs[0], first, second, third]),
            f"a pair of {pairs[0].name}s, {first.name} kicker",
        )

    high_to_low = kickers([], 5)
    return encode_strength(HandCategory.HIGH_CARD, high_to_low), f"{high_to_low[0].name} high"


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
