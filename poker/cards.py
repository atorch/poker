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


RANK_SYMBOLS = "23456789TJQKA"
SUIT_SYMBOLS = "hdcs"


def parse_cards(text):
    """Parse a compact card list like "Js 7h Ad" (T = ten) into Card objects."""
    return [
        Card(Rank(RANK_SYMBOLS.index(token[0])), Suit(SUIT_SYMBOLS.index(token[1])))
        for token in text.split()
    ]


def stacked_deck(hole_cards_by_player, board=(), rng=None):
    """
    A deck that deals the given hole cards (player 0 first), then the given board cards.

    State pops cards off the end of the deck, so the dealt cards go at the end in reverse
    order. Players or board cards that are None are dealt at random from the rest of the
    deck (shuffled with `rng`, a random.Random; unshuffled if rng is None).
    """
    fixed = [card for hole in hole_cards_by_player if hole is not None for card in hole] + list(board)
    rest = [card for card in FULL_DECK if card not in fixed]
    if rng is not None:
        rng.shuffle(rest)

    dealt_in_order = []
    for hole in hole_cards_by_player:
        dealt_in_order.extend(hole if hole is not None else [rest.pop(), rest.pop()])
    dealt_in_order.extend(board)
    return rest + list(reversed(dealt_in_order))
