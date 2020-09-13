from collections import Counter

from poker.cards import Card, card_index, FULL_DECK, Rank, Suit


def test_card_comparison():

    ace_of_hearts = Card(Rank.ACE, Suit.HEARTS)
    ten_of_hearts = Card(Rank.TEN, Suit.HEARTS)
    king_of_spades = Card(Rank.KING, Suit.SPADES)
    three_of_clubs = Card(Rank.THREE, Suit.CLUBS)

    assert (
        ace_of_hearts.rank
        > king_of_spades.rank
        > ten_of_hearts.rank
        > three_of_clubs.rank
    )

    assert ace_of_hearts.suit == ten_of_hearts.suit
    assert ace_of_hearts.suit != king_of_spades.suit


def test_deck():

    assert len(FULL_DECK) == 52

    suit_counter = Counter([card.suit for card in FULL_DECK])

    for suit in Suit:
        assert suit_counter[suit] == 13

    rank_counter = Counter([card.rank for card in FULL_DECK])

    for rank in Rank:
        assert rank_counter[rank] == 4


def test_card_index():

    for index, card in enumerate(FULL_DECK):
        assert index == card_index(card)
