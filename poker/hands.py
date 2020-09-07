from operator import attrgetter


def sort_hand(hand):

    return sorted(hand, key=attrgetter("rank"))


def strength(hand):

    # TODO
    return
