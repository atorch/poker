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


Card = namedtuple("Card", ["rank", "suit"])
