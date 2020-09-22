from poker.cards import Card, Rank, Suit
from poker.state import GameStage, State
from poker.utils import argmax


def test_state_with_low_wealth():

    initial_wealth = 5
    big_blind = 2
    small_blind = 1

    # Note: cards are dealt (popped) off of the _end_ of the list
    deck = [
        Card(Rank.KING, Suit.HEARTS),
        Card(Rank.KING, Suit.DIAMONDS),
        Card(Rank.KING, Suit.CLUBS),
        Card(Rank.THREE, Suit.HEARTS),
        Card(Rank.NINE, Suit.CLUBS),
        Card(Rank.TWO, Suit.SPADES),
        Card(Rank.SEVEN, Suit.SPADES),
        Card(Rank.TWO, Suit.CLUBS),
        Card(Rank.SEVEN, Suit.CLUBS),
        Card(Rank.TWO, Suit.DIAMONDS),
        Card(Rank.SEVEN, Suit.DIAMONDS),
        Card(Rank.ACE, Suit.HEARTS),
        Card(Rank.ACE, Suit.DIAMONDS),
    ]

    state = State(
        n_players=4,
        initial_wealth=initial_wealth,
        big_blind=big_blind,
        small_blind=small_blind,
        deck=deck,
    )

    state.update(big_blind)
    state.update(big_blind)

    # Note: the small blind completes
    state.update(big_blind - small_blind)
    assert state.game_stage == GameStage.FLOP

    for _ in range(state.n_players):
        state.update(small_blind)

    assert state.game_stage == GameStage.TURN

    for _ in range(state.n_players):
        state.update(small_blind)

    assert state.game_stage == GameStage.RIVER

    for _ in range(state.n_players):
        # Note: the players all knock (bet zero)
        state.update(0)

    # Note: player index 0 had a pair of aces as their hole cards, and they win the round. The next
    #  player to be big blind has only $1 left (and the big blind is $2), so they are forced to go all in
    #  as part of the blind
    assert state.wealth[0] == initial_wealth + (state.n_players - 1) * (
        big_blind + 2 * small_blind
    )
    assert (
        state.wealth[2]
        == state.wealth[3]
        == initial_wealth - (big_blind + 2 * small_blind)
        == 1
    )

    assert state.game_stage == GameStage.PRE_FLOP

    state.update(small_blind)
    state.update(small_blind)

    assert state.game_stage == GameStage.FLOP

    assert state.dealer == 1
    assert state.current_player == 2

    # Note: the low wealth players are all in, so bets > 0 are not allowed
    assert state.maximum_legal_bet() == 0

    for _ in range(state.n_players):
        state.update(0)

    assert state.game_stage == GameStage.TURN

    state.update(0)

    for _ in range(state.n_players - 1):
        state.update(-1)

    # Note: player 2 wins because everyone else has folded
    assert state.wealth[2] == 1 + small_blind * (state.n_players - 1)
    assert state.wealth[3] == 0


def test_minimum_legal_bet():

    initial_dealer = 0

    state = State(n_players=3)

    # Note: the first person to act is the player after the big blind,
    #  and their minimum legal bet is therefore the big blind amount
    assert state.minimum_legal_bet() == state.big_blind

    state.update(-1)

    min_legal_bet = state.minimum_legal_bet()
    assert min_legal_bet == (state.big_blind - state.small_blind)
    state.update(min_legal_bet)

    # Note: the big blind is allowed to bet zero dollars (to "knock")
    assert state.minimum_legal_bet() == 0


def test_ties():

    initial_wealth = 200.0
    initial_dealer = 0

    # Note: there will be a full house (aces and kings) on the board
    #  Cards are dealt (popped) off of the _end_ of the list
    deck = [
        Card(Rank.ACE, Suit.HEARTS),
        Card(Rank.ACE, Suit.DIAMONDS),
        Card(Rank.KING, Suit.HEARTS),
        Card(Rank.KING, Suit.DIAMONDS),
        Card(Rank.KING, Suit.CLUBS),
        Card(Rank.TWO, Suit.HEARTS),
        Card(Rank.SEVEN, Suit.HEARTS),
        Card(Rank.TWO, Suit.SPADES),
        Card(Rank.SEVEN, Suit.SPADES),
        Card(Rank.TWO, Suit.CLUBS),
        Card(Rank.SEVEN, Suit.CLUBS),
    ]

    # Note: we pass in a deck so that the order in which cards are dealt is known and deterministic
    state = State(
        n_players=3,
        initial_wealth=initial_wealth,
        initial_dealer=initial_dealer,
        deck=deck,
    )

    action_bet = 1
    action_fold = -1
    action_knock = 0

    state.update(state.big_blind)
    state.update(state.small_blind)
    state.update(action_knock)
    assert state.game_stage == GameStage.FLOP

    for _ in range(state.n_players):
        state.update(action_bet)

    assert state.game_stage == GameStage.TURN

    for _ in range(state.n_players):
        state.update(action_bet)

    assert state.game_stage == GameStage.RIVER

    amount_bet_by_player_who_folds = (
        sum(state.bets_by_stage[0][state.current_player])
        + sum(state.bets_by_stage[1][state.current_player])
        + sum(state.bets_by_stage[2][state.current_player])
    )

    state.update(action_fold)
    assert state.has_folded[1]

    # Note: the other two players stay in the game. They tie and split the pot
    state.update(action_bet)
    state.update(action_bet)

    amount_won_by_each_winner = amount_bet_by_player_who_folds / (state.n_players - 1)
    assert state.wealth[0] == initial_wealth + amount_won_by_each_winner
    assert state.wealth[2] == initial_wealth + amount_won_by_each_winner
    assert sum(state.wealth) == initial_wealth * state.n_players


def test_state():

    initial_wealth = 200
    initial_dealer = 0
    state = State(
        n_players=3, initial_wealth=initial_wealth, initial_dealer=initial_dealer
    )

    n_cards_in_deck_after_initial_deal = 52 - state.n_players * 2
    assert len(state.shuffled_deck) == n_cards_in_deck_after_initial_deal

    assert state.game_stage == GameStage.PRE_FLOP
    assert len(state.public_cards) == 0

    assert all(player_has_folded is False for player_has_folded in state.has_folded)

    # Note: any action < 0 means the current player is folding
    action_fold = -1

    # Note: player index 1 is the small blind, player index 2 is the big blind,
    #  so player index 0 (the dealer) is the first to act pre flop
    assert state.current_player == state.dealer

    state.update(action_fold)
    assert state.has_folded[state.dealer]

    assert state.current_player == 1
    ten_dollars = 10
    complete_blind_plus_ten = (state.big_blind - state.small_blind) + ten_dollars

    state.update(complete_blind_plus_ten)
    assert state.bets_by_stage[GameStage.PRE_FLOP][1] == [
        state.small_blind,
        complete_blind_plus_ten,
    ]
    assert state.game_stage == GameStage.PRE_FLOP

    # Note: the big blind player calls. At this point,
    #  the dealer has folded, and the other two have both bet the big blind plus $10.
    #  We move to the next stage (the flop) and the dealer deals three public cards
    state.update(ten_dollars)
    assert state.bets_by_stage[GameStage.PRE_FLOP][2] == [state.big_blind, ten_dollars]

    assert state.game_stage == GameStage.FLOP
    assert len(state.public_cards) == 3
    assert len(state.shuffled_deck) == n_cards_in_deck_after_initial_deal - 3

    # Note: the player after the dealer is the first to act after the flop
    assert state.current_player == 1

    # Note: the second player bets $10, then the dealer bets $20,
    #  the first player has already folded and does nothing, and then the second player calls
    state.update(ten_dollars)
    state.update(2 * ten_dollars)
    state.update(ten_dollars)

    # Note: the dealer folded before making any bets
    assert state.bets_by_stage[GameStage.FLOP][state.dealer] == []
    assert state.bets_by_stage[GameStage.FLOP][1] == [ten_dollars, ten_dollars]
    assert state.bets_by_stage[GameStage.FLOP][2] == [2 * ten_dollars]

    assert state.game_stage == GameStage.TURN
    assert len(state.public_cards) == 4

    assert state.current_player == 1
    state.update(ten_dollars)

    assert state.current_player == 2
    state.update(action_fold)

    total_bets_by_player_2 = ten_dollars * 3 + state.big_blind
    assert state.wealth[2] == initial_wealth - total_bets_by_player_2
    assert state.wealth[1] == initial_wealth + total_bets_by_player_2
    assert state.wealth[0] == initial_wealth

    assert sum(state.wealth) == initial_wealth * state.n_players

    assert len(state.public_cards) == 0
    assert len(state.shuffled_deck) == n_cards_in_deck_after_initial_deal

    # Note: the dealer shifts to the next player
    assert state.dealer == 1

    assert all(player_has_folded is False for player_has_folded in state.has_folded)

    assert sum(state.wealth) == initial_wealth * state.n_players

    action_knock = 0
    action_small_bet = 10
    action_big_bet = 20

    assert state.game_stage == GameStage.PRE_FLOP

    # Note: all players bet (or complete) the big blind, and we move to the next stage
    state.update(state.big_blind)
    state.update(state.small_blind)
    state.update(action_knock)

    assert state.game_stage == GameStage.FLOP
    assert len(state.public_cards) == 3

    for _ in range(state.n_players):
        # Note: all players bet $10 after the flop
        #  and we move to the next stage
        state.update(action_small_bet)

    assert state.game_stage == GameStage.TURN
    assert len(state.public_cards) == 4

    for _ in range(state.n_players):
        # Note: all players bet $10 after the turn
        #  and we move to the next stage
        state.update(action_small_bet)

    assert state.game_stage == GameStage.RIVER
    assert len(state.public_cards) == 5

    # Note: even though the stage has not ended, all public cards have
    #  been dealt and we can calculate hand strengths
    hand_strengths, hand_descriptions = state.calculate_best_hand_strengths()
    winning_players = argmax(hand_strengths)

    # Note: there may be multiple winning players (a tie)
    #  We assert on the first winning player's wealth
    wealth_before_winning = state.wealth[winning_players[0]]

    # Note: no players have folded, so all hand strengths must be positive
    assert max(hand_strengths) > 0

    state.update(action_small_bet)
    state.update(action_big_bet)
    state.update(action_big_bet)

    # Note: this player initially made a small bet, and the next person raised them with a large bet
    #  The first person to act now completes their bet (calls the raise)
    state.update(action_big_bet - action_small_bet)

    # Note: the bets were small during the flop and turn, and big during the river
    amount_won = (
        (state.n_players - len(winning_players))
        * (2 * action_small_bet + action_big_bet + state.big_blind)
        / len(winning_players)
    )
    assert state.wealth[winning_players[0]] == wealth_before_winning + amount_won
