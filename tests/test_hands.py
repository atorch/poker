from poker.cards import Suit, Rank, Card
from poker.hands import strength, sort_hand


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


def test_hand_strengh():

    flush = [
        Card(Rank.ACE, Suit.HEARTS),
        Card(Rank.JACK, Suit.HEARTS),
        Card(Rank.THREE, Suit.HEARTS),
        Card(Rank.SEVEN, Suit.HEARTS),
        Card(Rank.TWO, Suit.HEARTS),
    ]

    three_of_a_kind = [
        Card(Rank.SEVEN, Suit.HEARTS),
        Card(Rank.SEVEN, Suit.CLUBS),
        Card(Rank.SEVEN, Suit.DIAMONDS),
        Card(Rank.TWO, Suit.HEARTS),
        Card(Rank.JACK, Suit.HEARTS),
    ]

    pair_of_jacks = [
        Card(Rank.JACK, Suit.CLUBS),
        Card(Rank.THREE, Suit.DIAMONDS),
        Card(Rank.SEVEN, Suit.HEARTS),
        Card(Rank.TWO, Suit.HEARTS),
        Card(Rank.JACK, Suit.HEARTS),
    ]

    high_card = [
        Card(Rank.JACK, Suit.CLUBS),
        Card(Rank.EIGHT, Suit.HEARTS),
        Card(Rank.TWO, Suit.HEARTS),
        Card(Rank.THREE, Suit.DIAMONDS),
        Card(Rank.QUEEN, Suit.HEARTS),
    ]

    assert (
        strength(flush)
        > strength(three_of_a_kind)
        > strength(pair_of_jacks)
        > strength(high_card)
    )
