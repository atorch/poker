from collections import Counter
from operator import attrgetter


def sort_hand(hand):

    return sorted(hand, key=attrgetter("rank"))


def strength(hand):

    # TODO Finish this function

    sorted_hand = sort_hand(hand)

    unique_suits = set(card.suit for card in hand)

    # Note: the hand is a flush (either a straight flush or a regular flush)
    if len(unique_suits) == 1:
        # TODO Distinguish flush and straight flush (and stronger and weaker flushes)
        return 10

    rank_counter = Counter([card.rank for card in hand])
    most_common_ranks = rank_counter.most_common()

    if most_common_ranks[0][1] == 4:
        # Four of a kind
        return 20

    if most_common_ranks[0][1] == 3 and most_common_ranks[1][1] == 2:
        # Full house
        return 15

    if most_common_ranks[0][1] == 3:
        # Three of a kind
        return 5

    if most_common_ranks[0][1] == 2 and most_common_ranks[1][1] == 2:
        # Two pair
        return 4

    if most_common_ranks[0][1] == 2:
        # Pair
        return 2

    # High card
    return 0
