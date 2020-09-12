from collections import namedtuple
from enum import Enum, IntEnum


class Suit(Enum):

    # Note: suits are all of equal value in poker:
    #  no suit is stronger than any other
    HEARTS = 0
    DIAMONDS = 1
    CLUBS = 2
    SPADES = 3


class Rank(IntEnum):

    # Note: aces are strongest, but can also be used as the low card in straights
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14


class Card:
    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit

    def __eq__(self, other):
        return self.rank == other.rank and self.suit == other.suit

    def __repr__(self):
        # Example: "SIX of Suit.HEARTS"
        return f"{self.rank.name} of {self.suit}"


FULL_DECK = tuple(Card(rank, suit) for rank in Rank for suit in Suit)
