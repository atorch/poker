from poker.cards import Suit, Rank, Card
from poker.hands import sort_hand


def test_sort_hand():

    unsorted_hand = [
        Card(Rank.ACE, Suit.HEARTS),
        Card(Rank.ACE, Suit.SPADES),
        Card(Rank.THREE, Suit.CLUBS),
        Card(Rank.SEVEN, Suit.DIAMONDS),
        Card(Rank.TWO, Suit.CLUBS),
    ]

    sorted_hand = sort_hand(unsorted_hand)

    assert sorted_hand[0] == Card(Rank.TWO, Suit.CLUBS)
    assert sorted_hand[1] == Card(Rank.THREE, Suit.CLUBS)
    assert sorted_hand[2] == Card(Rank.SEVEN, Suit.DIAMONDS)
    assert sorted_hand[3].rank == sorted_hand[4].rank == Rank.ACE
