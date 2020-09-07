from poker.cards import Suit, Rank, Card


def test_card_comparison():

    ace_of_hearts = Card(Rank.ACE, Suit.HEARTS)
    ten_of_hearts = Card(Rank.TEN, Suit.HEARTS)
    king_of_spades = Card(Rank.KING, Suit.SPADES)
    three_of_clubs = Card(Rank.THREE, Suit.CLUBS)

    assert ace_of_hearts.rank > king_of_spades.rank > ten_of_hearts.rank > three_of_clubs.rank

    assert ace_of_hearts.suit == ten_of_hearts.suit
    assert ace_of_hearts.suit != king_of_spades.suit
