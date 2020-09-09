from poker.cards import Suit, Rank, Card
from poker.hands import is_straight, strength, sort_hand


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


def test_straights():

    straight = [
        Card(Rank.THREE, Suit.CLUBS),
        Card(Rank.FOUR, Suit.CLUBS),
        Card(Rank.FIVE, Suit.CLUBS),
        Card(Rank.SIX, Suit.SPADES),
        Card(Rank.SEVEN, Suit.DIAMONDS),
    ]

    non_straight = [
        Card(Rank.THREE, Suit.CLUBS),
        Card(Rank.FOUR, Suit.CLUBS),
        Card(Rank.FIVE, Suit.CLUBS),
        Card(Rank.SIX, Suit.SPADES),
        Card(Rank.EIGHT, Suit.DIAMONDS),
    ]

    another_non_straight = [
        Card(Rank.ACE, Suit.HEARTS),
        Card(Rank.ACE, Suit.SPADES),
        Card(Rank.THREE, Suit.CLUBS),
        Card(Rank.SEVEN, Suit.DIAMONDS),
        Card(Rank.TWO, Suit.CLUBS),
    ]

    assert is_straight(sort_hand(straight))
    assert not is_straight(sort_hand(non_straight))
    assert not is_straight(sort_hand(another_non_straight))

    # TODO Also test low straights (ace, two, three, ...)


def test_hand_strengh():

    straight_flush = [
        Card(Rank.ACE, Suit.HEARTS),
        Card(Rank.JACK, Suit.HEARTS),
        Card(Rank.QUEEN, Suit.HEARTS),
        Card(Rank.KING, Suit.HEARTS),
        Card(Rank.TEN, Suit.HEARTS),
    ]

    four_aces = [
        Card(Rank.ACE, Suit.HEARTS),
        Card(Rank.ACE, Suit.SPADES),
        Card(Rank.ACE, Suit.CLUBS),
        Card(Rank.ACE, Suit.DIAMONDS),
        Card(Rank.JACK, Suit.HEARTS),
    ]

    four_queens = [
        Card(Rank.QUEEN, Suit.HEARTS),
        Card(Rank.QUEEN, Suit.SPADES),
        Card(Rank.QUEEN, Suit.CLUBS),
        Card(Rank.QUEEN, Suit.DIAMONDS),
        Card(Rank.JACK, Suit.HEARTS),
    ]

    flush = [
        Card(Rank.ACE, Suit.HEARTS),
        Card(Rank.JACK, Suit.HEARTS),
        Card(Rank.THREE, Suit.HEARTS),
        Card(Rank.SEVEN, Suit.HEARTS),
        Card(Rank.TWO, Suit.HEARTS),
    ]

    ace_high_straight = [
        Card(Rank.ACE, Suit.HEARTS),
        Card(Rank.JACK, Suit.SPADES),
        Card(Rank.QUEEN, Suit.HEARTS),
        Card(Rank.KING, Suit.HEARTS),
        Card(Rank.TEN, Suit.HEARTS),
    ]

    three_sevens = [
        Card(Rank.SEVEN, Suit.HEARTS),
        Card(Rank.SEVEN, Suit.CLUBS),
        Card(Rank.SEVEN, Suit.DIAMONDS),
        Card(Rank.TWO, Suit.HEARTS),
        Card(Rank.JACK, Suit.HEARTS),
    ]

    three_sixes = [
        Card(Rank.SIX, Suit.HEARTS),
        Card(Rank.SIX, Suit.CLUBS),
        Card(Rank.SIX, Suit.DIAMONDS),
        Card(Rank.FIVE, Suit.HEARTS),
        Card(Rank.JACK, Suit.SPADES),
    ]

    two_pair = [
        Card(Rank.JACK, Suit.CLUBS),
        Card(Rank.JACK, Suit.HEARTS),
        Card(Rank.THREE, Suit.DIAMONDS),
        Card(Rank.THREE, Suit.HEARTS),
        Card(Rank.TWO, Suit.HEARTS),
    ]

    pair_of_jacks = [
        Card(Rank.THREE, Suit.DIAMONDS),
        Card(Rank.SEVEN, Suit.HEARTS),
        Card(Rank.TWO, Suit.HEARTS),
        Card(Rank.JACK, Suit.CLUBS),
        Card(Rank.JACK, Suit.HEARTS),
    ]

    pair_of_eights = [
        Card(Rank.THREE, Suit.DIAMONDS),
        Card(Rank.SEVEN, Suit.HEARTS),
        Card(Rank.TWO, Suit.HEARTS),
        Card(Rank.EIGHT, Suit.CLUBS),
        Card(Rank.EIGHT, Suit.HEARTS),
    ]

    pair_of_sevens = [
        Card(Rank.THREE, Suit.DIAMONDS),
        Card(Rank.JACK, Suit.HEARTS),
        Card(Rank.TWO, Suit.HEARTS),
        Card(Rank.SEVEN, Suit.CLUBS),
        Card(Rank.SEVEN, Suit.HEARTS),
    ]

    queen_high = [
        Card(Rank.JACK, Suit.CLUBS),
        Card(Rank.EIGHT, Suit.HEARTS),
        Card(Rank.TWO, Suit.HEARTS),
        Card(Rank.THREE, Suit.DIAMONDS),
        Card(Rank.QUEEN, Suit.HEARTS),
    ]

    jack_high = [
        Card(Rank.JACK, Suit.CLUBS),
        Card(Rank.EIGHT, Suit.HEARTS),
        Card(Rank.TWO, Suit.HEARTS),
        Card(Rank.THREE, Suit.DIAMONDS),
        Card(Rank.FOUR, Suit.HEARTS),
    ]

    nine_high = [
        Card(Rank.NINE, Suit.CLUBS),
        Card(Rank.EIGHT, Suit.HEARTS),
        Card(Rank.TWO, Suit.HEARTS),
        Card(Rank.THREE, Suit.DIAMONDS),
        Card(Rank.FOUR, Suit.HEARTS),
    ]

    assert (
        strength(straight_flush)
        > strength(four_aces)
        > strength(four_queens)
        > strength(flush)
        > strength(ace_high_straight)
        > strength(three_sevens)
        > strength(three_sixes)
        > strength(two_pair)
        > strength(pair_of_jacks)
        > strength(pair_of_eights)
        > strength(pair_of_sevens)
        > strength(queen_high)
        > strength(jack_high)
        > strength(nine_high)
    )
