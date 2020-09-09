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

    first_most_common_rank = most_common_ranks[0][0]
    first_most_common_count = most_common_ranks[0][1]
    second_most_common_count = most_common_ranks[1][1]

    if first_most_common_count == 4:
        # Four of a kind (with ties broken by rank)
        # TODO Are rank ties broken by the fifth card (the kicker?)
        return 700 + first_most_common_rank

    if first_most_common_count == 3 and second_most_common_count == 2:
        # Full house
        return 600

    if straight:
        # Regular straight (not a straight flush)
        # TODO Tiebreaking for straights (stronger/weaker straights), including ace-low straights
        return 400

    if first_most_common_count == 3:
        # Three of a kind (with ties broken by rank)
        # TODO Do Texas holdem rules break ties using the kicker?
        return 300 + first_most_common_rank

    if first_most_common_count == 2 and second_most_common_count == 2:
        # Two pair (with ties broken first by the rank of the high pair, then by the rank of the low pair)
        return 200

    if first_most_common_count == 2:
        # Pair (with ties broken by the rank of the pair)
        return 100 + first_most_common_rank

    # High card
    return sorted_hand[-1].rank
