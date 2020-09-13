from collections import Counter
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
            return 1000 + straight_tiebreaker, "a straight flush"

        # Regular flush (ties broken by the rank of the high card)
        description = f"a {hand[0].suit.name} flush"
        return 500 + sorted_hand[-1].rank, description

    rank_counter = Counter([card.rank for card in hand])
    most_common_ranks = rank_counter.most_common()

    first_most_common_rank = most_common_ranks[0][0]
    second_most_common_rank = most_common_ranks[1][0]
    first_most_common_count = most_common_ranks[0][1]
    second_most_common_count = most_common_ranks[1][1]

    if first_most_common_count == 4:
        description = f"four {first_most_common_rank.name}"
        return 700 + first_most_common_rank, description

    if first_most_common_count == 3 and second_most_common_count == 2:
        description = f"a full house ({first_most_common_rank.name} full of {second_most_common_rank.name})"
        return 600 + first_most_common_rank, description

    if straight:
        description = f"a straight ending in a {straight_tiebreaker.name}"
        return 400 + straight_tiebreaker, description

    if first_most_common_count == 3:
        description = f"three of a kind ({first_most_common_rank.name})"
        return 300 + first_most_common_rank, description

    if first_most_common_count == 2 and second_most_common_count == 2:
        return 200, "two pair"

    if first_most_common_count == 2:
        description = f"a pair of {first_most_common_rank.name}"
        return 100 + first_most_common_rank, description

    description = f"{sorted_hand[-1].rank.name} high"
    return int(sorted_hand[-1].rank), description
