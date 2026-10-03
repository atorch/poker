from collections import Counter
from itertools import combinations
from random import Random

import pytest

from poker.cards import Suit, Rank, Card, FULL_DECK
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

    assert best_hand_strength(three_public_cards, hole_cards) == strength(
        royal_straight_flush
    )
    assert best_hand_strength(five_public_cards, hole_cards) == strength(
        royal_straight_flush
    )


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
        strength(royal_straight_flush)[0]
        > strength(straight_flush)[0]
        > strength(ace_low_straight_flush)[0]
        > strength(four_aces)[0]
        > strength(four_queens)[0]
        > strength(flush_ace_high)[0]
        > strength(flush_king_high)[0]
        > strength(ace_high_straight)[0]
        > strength(king_high_straight)[0]
        > strength(six_high_straight)[0]
        > strength(ace_low_straight)[0]
        > strength(three_sevens)[0]
        > strength(three_sixes)[0]
        > strength(two_pair)[0]
        > strength(pair_of_jacks)[0]
        > strength(pair_of_eights)[0]
        > strength(pair_of_sevens)[0]
        > strength(queen_high)[0]
        > strength(jack_high)[0]
        > strength(nine_high)[0]
    )


def test_two_pair_tie_breaking():
    """
    Test that two-pair hands are correctly ordered by their higher pair.
    (Lower pair and kicker tiebreaks are covered in test_kickers_and_lower_pairs_break_ties.)
    """
    # Aces and Twos (AA22x) - higher pair is Aces
    aces_and_twos = [
        Card(Rank.ACE, Suit.HEARTS),
        Card(Rank.ACE, Suit.SPADES),
        Card(Rank.TWO, Suit.CLUBS),
        Card(Rank.TWO, Suit.DIAMONDS),
        Card(Rank.FIVE, Suit.HEARTS),
    ]

    # Kings and Queens (KKQQx) - higher pair is Kings
    kings_and_queens = [
        Card(Rank.KING, Suit.HEARTS),
        Card(Rank.KING, Suit.SPADES),
        Card(Rank.QUEEN, Suit.CLUBS),
        Card(Rank.QUEEN, Suit.DIAMONDS),
        Card(Rank.FIVE, Suit.HEARTS),
    ]

    # Jacks and Threes (JJ33x) - higher pair is Jacks
    jacks_and_threes = [
        Card(Rank.JACK, Suit.HEARTS),
        Card(Rank.JACK, Suit.SPADES),
        Card(Rank.THREE, Suit.CLUBS),
        Card(Rank.THREE, Suit.DIAMONDS),
        Card(Rank.FIVE, Suit.HEARTS),
    ]

    # Verify ordering: AA22 > KKQQ > JJ33
    strength_aces_twos = strength(aces_and_twos)[0]
    strength_kings_queens = strength(kings_and_queens)[0]
    strength_jacks_threes = strength(jacks_and_threes)[0]

    assert strength_aces_twos > strength_kings_queens
    assert strength_kings_queens > strength_jacks_threes
    assert strength_aces_twos > strength_jacks_threes


def cards(text):
    """Parse a compact hand like "Js 7h Ad" into Card objects (T = ten)."""
    ranks = {"2": Rank.TWO, "3": Rank.THREE, "4": Rank.FOUR, "5": Rank.FIVE, "6": Rank.SIX,
             "7": Rank.SEVEN, "8": Rank.EIGHT, "9": Rank.NINE, "T": Rank.TEN, "J": Rank.JACK,
             "Q": Rank.QUEEN, "K": Rank.KING, "A": Rank.ACE}
    suits = {"h": Suit.HEARTS, "d": Suit.DIAMONDS, "c": Suit.CLUBS, "s": Suit.SPADES}
    return [Card(ranks[token[0]], suits[token[1]]) for token in text.split()]


def test_board_two_pair_ace_kicker_wins_outright():
    # Regression: this showdown was scored as a three-way tie (kickers were ignored)
    board = cards("7s Jh Ts Js 7h")
    king_kicker = best_hand_strength(board, cards("3h Kd"))[0]
    board_plays = best_hand_strength(board, cards("2d 8h"))[0]
    ace_kicker = best_hand_strength(board, cards("6d Ac"))[0]

    assert ace_kicker > king_kicker > board_plays


@pytest.mark.parametrize(
    "better, worse",
    [
        ("Jh Js 7c 7d 5h", "Jc Jd 3c 3d Ah"),  # two pair: lower pair matters
        ("Jh Js 7c 7d Ah", "Jc Jd 7h 7s Kh"),  # two pair: kicker matters
        ("9h 9s Ac 4d 3h", "9c 9d Kc Qd Jh"),  # pair: first kicker
        ("9h 9s Ac Kd 3h", "9c 9d Ah Qd Jh"),  # pair: second kicker
        ("9h 9s Ac Kd 4h", "9c 9d Ah Kh 3c"),  # pair: third kicker
        ("Ah Kd 9c 5s 4h", "As Kc 9d 5h 3c"),  # high card: fifth card
        ("5h 5s 5c Ad 2h", "5h 5s 5c Kd Qh"),  # trips: kicker
        ("Kh Ks Kc 3d 3h", "Kh Ks Kc 2d 2h"),  # full house: pair breaks ties on a trips board
        ("Qh Qs Qc Qd Ah", "Qh Qs Qc Qd Kh"),  # quads: kicker
        ("Ah Jh 9h 6h 4h", "As Js 9s 6s 3s"),  # flush: lower cards
    ],
)
def test_kickers_and_lower_pairs_break_ties(better, worse):
    assert strength(cards(better))[0] > strength(cards(worse))[0]


@pytest.mark.parametrize(
    "hand, other",
    [
        ("Ah Kd 9c 5s 4h", "As Kc 9d 5h 4c"),  # identical ranks, different suits
        ("6h 5d 4c 3s 2h", "6s 5c 4d 3h 2c"),  # same straight
        ("5h 4d 3c 2s Ah", "5s 4c 3d 2h As"),  # same wheel
    ],
)
def test_genuine_ties(hand, other):
    assert strength(cards(hand))[0] == strength(cards(other))[0]


def test_wheel_is_lowest_straight():
    assert strength(cards("6h 5d 4c 3s 2h"))[0] > strength(cards("5h 4d 3c 2s Ah"))[0]
    assert strength(cards("5h 4d 3c 2s Ah"))[0] > strength(cards("Ah Ad Ac Ks Qh"))[0]


def reference_strength(hand):
    """Independent, deliberately simple evaluator used to cross-check strength()."""
    ranks = sorted((int(card.rank) for card in hand), reverse=True)
    counts = Counter(ranks)
    by_group = tuple(sorted(counts, key=lambda rank: (counts[rank], rank), reverse=True))
    shape = sorted(counts.values(), reverse=True)
    flush = len({card.suit for card in hand}) == 1
    distinct = sorted(set(ranks), reverse=True)
    straight_high = None
    if len(distinct) == 5 and distinct[0] - distinct[4] == 4:
        straight_high = distinct[0]
    elif distinct == [12, 3, 2, 1, 0]:
        straight_high = 3

    if straight_high is not None and flush:
        return (8, (straight_high,))
    if shape == [4, 1]:
        return (7, by_group)
    if shape == [3, 2]:
        return (6, by_group)
    if flush:
        return (5, tuple(ranks))
    if straight_high is not None:
        return (4, (straight_high,))
    if shape == [3, 1, 1]:
        return (3, by_group)
    if shape == [2, 2, 1]:
        return (2, by_group)
    if shape == [2, 1, 1, 1]:
        return (1, by_group)
    return (0, tuple(ranks))


def test_strength_ordering_matches_reference_evaluator():
    rng = Random(0)
    for _ in range(3000):
        deck = rng.sample(FULL_DECK, 10)
        hand, other = deck[:5], deck[5:]

        ours = strength(hand)[0], strength(other)[0]
        reference = reference_strength(hand), reference_strength(other)

        assert (ours[0] > ours[1]) == (reference[0] > reference[1])
        assert (ours[0] == ours[1]) == (reference[0] == reference[1])


def test_best_hand_strength_matches_reference_on_seven_cards():
    rng = Random(1)
    for _ in range(300):
        deck = rng.sample(FULL_DECK, 9)
        board, first_hole, second_hole = deck[:5], deck[5:7], deck[7:]

        ours = [best_hand_strength(board, hole)[0] for hole in (first_hole, second_hole)]
        reference = [
            max(reference_strength(five) for five in combinations(board + hole, 5))
            for hole in (first_hole, second_hole)
        ]

        assert (ours[0] > ours[1]) == (reference[0] > reference[1])
        assert (ours[0] == ours[1]) == (reference[0] == reference[1])
