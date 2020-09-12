from collections import Counter
from itertools import combinations
from operator import attrgetter

from poker.cards import Rank


def sort_hand(hand):

    return sorted(hand, key=attrgetter("rank"))


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


def best_hand_strength(public_cards, hole_cards):

    available_cards = public_cards + hole_cards

    return max(
        strength(candidate_hand) for candidate_hand in combinations(available_cards, 5)
    )


def strength(hand):

    # TODO Finish this function

    unique_suits = set(card.suit for card in hand)

    sorted_hand = sort_hand(hand)
    straight, straight_tiebreaker = is_straight(sorted_hand)

    # Note: the hand is a flush (either a straight flush or a regular flush)
    if len(unique_suits) == 1:

        if straight:
            # Straight flush (ties broken by the rank of the last card in the straight)
            return 1000 + straight_tiebreaker, "straight flush"

        # Regular flush (ties broken by the rank of the high card)
        return 500 + sorted_hand[-1].rank, "flush"

    rank_counter = Counter([card.rank for card in hand])
    most_common_ranks = rank_counter.most_common()

    first_most_common_rank = most_common_ranks[0][0]
    first_most_common_count = most_common_ranks[0][1]
    second_most_common_count = most_common_ranks[1][1]

    if first_most_common_count == 4:
        # Four of a kind (with ties broken by rank)
        # TODO Are rank ties broken by the fifth card (the kicker?)
        return 700 + first_most_common_rank, "four of a kind"

    if first_most_common_count == 3 and second_most_common_count == 2:
        # Full house
        return 600 + first_most_common_rank, "full house"

    if straight:
        # Regular straight
        return 400 + straight_tiebreaker, "straight"

    if first_most_common_count == 3:
        # Three of a kind (with ties broken by rank)
        # TODO Do Texas holdem rules break ties using the kicker?
        return 300 + first_most_common_rank, "three of a kind"

    if first_most_common_count == 2 and second_most_common_count == 2:
        # Two pair (with ties broken first by the rank of the high pair, then by the rank of the low pair)
        return 200, "two pair"

    if first_most_common_count == 2:
        # Pair (with ties broken by the rank of the pair)
        return 100 + first_most_common_rank, f"pair of {first_most_common_rank.name}"

    # High card
    return int(sorted_hand[-1].rank), f"high card ({sorted_hand[-1].rank.name})"
