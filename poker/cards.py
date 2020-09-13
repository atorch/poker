from collections import namedtuple
from enum import Enum, IntEnum


class Suit(IntEnum):

    # Note: suits are all of equal value in poker:
    #  no suit is stronger than any other
    HEARTS = 0
    DIAMONDS = 1
    CLUBS = 2
    SPADES = 3


class Rank(IntEnum):

    # Note: aces are strongest, but can also be used as the low card in straights
    TWO = 0
    THREE = 1
    FOUR = 2
    FIVE = 3
    SIX = 4
    SEVEN = 5
    EIGHT = 6
    NINE = 7
    TEN = 8
    JACK = 9
    QUEEN = 10
    KING = 11
    ACE = 12


class Card:
    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit

    def __eq__(self, other):
        return self.rank == other.rank and self.suit == other.suit

    def __repr__(self):
        # Example: "SIX of HEARTS"
        return f"{self.rank.name} of {self.suit.name}"


FULL_DECK = tuple(Card(rank, suit) for rank in Rank for suit in Suit)


def card_index(card):
    return card.suit + card.rank * len(Suit)
