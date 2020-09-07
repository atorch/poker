from collections import Counter
from operator import attrgetter


def sort_hand(hand):

    return sorted(hand, key=attrgetter("rank"))


def is_straight(sorted_hand):

    for i, card in enumerate(sorted_hand[:-1]):

        next_card = sorted_hand[i + 1]

        if card.rank + 1 != next_card.rank:
            return False

    return True


def strength(hand):

    # TODO Finish this function

    unique_suits = set(card.suit for card in hand)

    sorted_hand = sort_hand(hand)
    straight = is_straight(sorted_hand)

    # Note: the hand is a flush (either a straight flush or a regular flush)
    if len(unique_suits) == 1:

        if straight:
            # Straight flush
            return 1000

        # Regular flush
        return 500

    rank_counter = Counter([card.rank for card in hand])
    most_common_ranks = rank_counter.most_common()

    if most_common_ranks[0][1] == 4:
        # Four of a kind
        return 700

    if most_common_ranks[0][1] == 3 and most_common_ranks[1][1] == 2:
        # Full house
        return 600

    if straight:
        # Regular straight (not a straight flush)
        return 400

    if most_common_ranks[0][1] == 3:
        # Three of a kind
        return 300

    if most_common_ranks[0][1] == 2 and most_common_ranks[1][1] == 2:
        # Two pair
        return 200

    if most_common_ranks[0][1] == 2:
        # Pair
        return 100 + most_common_ranks[0][0]

    # High card
    return sorted_hand[-1].rank
