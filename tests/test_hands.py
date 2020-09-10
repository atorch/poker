from poker.cards import Suit, Rank, Card
from poker.hands import best_hand_strength, is_straight, strength, sort_hand


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
        Card(Rank.SIX, Suit.SPADES),
        Card(Rank.SEVEN, Suit.DIAMONDS),
        Card(Rank.THREE, Suit.CLUBS),
        Card(Rank.FOUR, Suit.CLUBS),
        Card(Rank.FIVE, Suit.CLUBS),
    ]

    ace_low_straight = [
        Card(Rank.THREE, Suit.CLUBS),
        Card(Rank.FOUR, Suit.CLUBS),
        Card(Rank.FIVE, Suit.CLUBS),
        Card(Rank.ACE, Suit.SPADES),
        Card(Rank.TWO, Suit.DIAMONDS),
    ]

    almost_ace_low_straight = [
        Card(Rank.THREE, Suit.CLUBS),
        Card(Rank.FOUR, Suit.CLUBS),
        Card(Rank.SIX, Suit.CLUBS),
        Card(Rank.ACE, Suit.SPADES),
        Card(Rank.TWO, Suit.DIAMONDS),
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

    assert is_straight(sort_hand(straight)) == (True, Rank.SEVEN)
    assert is_straight(sort_hand(ace_low_straight)) == (True, Rank.FIVE)
    assert not is_straight(sort_hand(almost_ace_low_straight))[0]
    assert not is_straight(sort_hand(non_straight))[0]
    assert not is_straight(sort_hand(another_non_straight))[0]


def test_best_hand_strength():

    # Note: the hole cards are a player's private cards
    hole_cards = [
        Card(Rank.ACE, Suit.HEARTS),
        Card(Rank.JACK, Suit.HEARTS),
    ]

    three_public_cards = [
        Card(Rank.QUEEN, Suit.HEARTS),
        Card(Rank.KING, Suit.HEARTS),
        Card(Rank.TEN, Suit.HEARTS),
    ]

    five_public_cards = [
        Card(Rank.SEVEN, Suit.CLUBS),
        Card(Rank.QUEEN, Suit.HEARTS),
        Card(Rank.KING, Suit.HEARTS),
        Card(Rank.TEN, Suit.HEARTS),
        Card(Rank.TWO, Suit.CLUBS),
    ]

    royal_straight_flush = [
        Card(Rank.ACE, Suit.HEARTS),
        Card(Rank.JACK, Suit.HEARTS),
        Card(Rank.QUEEN, Suit.HEARTS),
        Card(Rank.KING, Suit.HEARTS),
        Card(Rank.TEN, Suit.HEARTS),
    ]

    assert best_hand_strength(three_public_cards, hole_cards) == strength(royal_straight_flush)
    assert best_hand_strength(five_public_cards, hole_cards) == strength(royal_straight_flush)


def test_hand_strengh():

    royal_straight_flush = [
        Card(Rank.ACE, Suit.HEARTS),
        Card(Rank.JACK, Suit.HEARTS),
        Card(Rank.QUEEN, Suit.HEARTS),
        Card(Rank.KING, Suit.HEARTS),
        Card(Rank.TEN, Suit.HEARTS),
    ]

    straight_flush = [
        Card(Rank.NINE, Suit.HEARTS),
        Card(Rank.JACK, Suit.HEARTS),
        Card(Rank.QUEEN, Suit.HEARTS),
        Card(Rank.KING, Suit.HEARTS),
        Card(Rank.TEN, Suit.HEARTS),
    ]

    ace_low_straight_flush = [
        Card(Rank.ACE, Suit.HEARTS),
        Card(Rank.TWO, Suit.HEARTS),
        Card(Rank.THREE, Suit.HEARTS),
        Card(Rank.FOUR, Suit.HEARTS),
        Card(Rank.FIVE, Suit.HEARTS),
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

    flush_ace_high = [
        Card(Rank.ACE, Suit.HEARTS),
        Card(Rank.JACK, Suit.HEARTS),
        Card(Rank.THREE, Suit.HEARTS),
        Card(Rank.SEVEN, Suit.HEARTS),
        Card(Rank.TWO, Suit.HEARTS),
    ]

    flush_king_high = [
        Card(Rank.KING, Suit.HEARTS),
        Card(Rank.JACK, Suit.HEARTS),
        Card(Rank.THREE, Suit.HEARTS),
        Card(Rank.SEVEN, Suit.HEARTS),
        Card(Rank.EIGHT, Suit.HEARTS),
    ]

    ace_high_straight = [
        Card(Rank.ACE, Suit.HEARTS),
        Card(Rank.JACK, Suit.SPADES),
        Card(Rank.QUEEN, Suit.HEARTS),
        Card(Rank.KING, Suit.HEARTS),
        Card(Rank.TEN, Suit.HEARTS),
    ]

    king_high_straight = [
        Card(Rank.NINE, Suit.HEARTS),
        Card(Rank.JACK, Suit.SPADES),
        Card(Rank.QUEEN, Suit.HEARTS),
        Card(Rank.KING, Suit.HEARTS),
        Card(Rank.TEN, Suit.HEARTS),
    ]

    six_high_straight = [
        Card(Rank.SIX, Suit.HEARTS),
        Card(Rank.TWO, Suit.SPADES),
        Card(Rank.THREE, Suit.HEARTS),
        Card(Rank.FOUR, Suit.HEARTS),
        Card(Rank.FIVE, Suit.HEARTS),
    ]

    ace_low_straight = [
        Card(Rank.ACE, Suit.HEARTS),
        Card(Rank.TWO, Suit.SPADES),
        Card(Rank.THREE, Suit.HEARTS),
        Card(Rank.FOUR, Suit.HEARTS),
        Card(Rank.FIVE, Suit.HEARTS),
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
        strength(royal_straight_flush)
        > strength(straight_flush)
        > strength(ace_low_straight_flush)
        > strength(four_aces)
        > strength(four_queens)
        > strength(flush_ace_high)
        > strength(flush_king_high)
        > strength(ace_high_straight)
        > strength(king_high_straight)
        > strength(six_high_straight)
        > strength(ace_low_straight)
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
