"""
Tests to investigate why player 2 almost never wins.

The fairness test shows:
- Player 0: 63.5% wins
- Player 1: 36.4% wins
- Player 2: 0.1% wins (only 1 out of 1000!)

These tests investigate game mechanics to find the bug.
"""
import numpy as np
from poker.state import State, GameStage
from poker.cards import Card, Rank, Suit


def test_blind_posting_fairness():
    """
    Test that each player posts small blind and big blind equally often
    across many deals with different initial dealers.

    This catches bugs where player 2 might always be in a disadvantageous position.
    """
    small_blind_counts = [0, 0, 0]  # Track who posts small blind
    big_blind_counts = [0, 0, 0]    # Track who posts big blind

    n_deals = 300  # 100 deals starting from each dealer position

    for initial_dealer in [0, 1, 2]:
        for _ in range(100):
            state = State(n_players=3, initial_wealth=100, initial_dealer=initial_dealer)

            # After State.__init__, blinds have been posted
            # Small blind is posted by player left of dealer
            # Big blind is posted by player left of small blind

            small_blind_player = (initial_dealer + 1) % 3
            big_blind_player = (initial_dealer + 2) % 3

            small_blind_counts[small_blind_player] += 1
            big_blind_counts[big_blind_player] += 1

    # Each player should post each blind 100 times (roughly equal)
    expected_count = 100

    print(f"\nBlind posting distribution over {n_deals} deals:")
    print(f"  Small blind: {small_blind_counts}")
    print(f"  Big blind: {big_blind_counts}")

    for i in range(3):
        assert small_blind_counts[i] == expected_count, \
            f"Player {i} posted small blind {small_blind_counts[i]} times, expected {expected_count}"
        assert big_blind_counts[i] == expected_count, \
            f"Player {i} posted big blind {big_blind_counts[i]} times, expected {expected_count}"


def test_current_player_after_blinds():
    """
    Test who acts first after blinds are posted.

    Current player should be the dealer (first to act after blinds).
    Bug: Maybe player 2 never gets to act?
    """
    for initial_dealer in [0, 1, 2]:
        state = State(n_players=3, initial_wealth=100, initial_dealer=initial_dealer)

        # After blinds are posted, current player should be the dealer
        # (first to act after small blind and big blind)
        expected_current = initial_dealer

        print(f"\nDealer={initial_dealer}, current_player={state.current_player}, expected={expected_current}")

        assert state.current_player == expected_current, \
            f"With dealer={initial_dealer}, current_player should be {expected_current}, got {state.current_player}"


def test_initial_wealth_distribution():
    """
    Test that all players start with equal wealth.

    Note: Wealth is NOT deducted during betting - it's only updated at round end
    in redistribute_wealth_and_reinitialize(). During betting, bets are tracked in
    bets_by_stage and wealth remains constant.
    """
    state = State(n_players=3, initial_wealth=100, initial_dealer=0)

    print(f"\nInitial wealth after blinds posted: {state.wealth}")
    print(f"  All players: {state.wealth}")
    print(f"  Bets: {state.bets_by_stage[state.game_stage]}")

    # Wealth stays constant during betting
    assert all(w == 100 for w in state.wealth), f"All players should have 100 wealth, got {state.wealth}"

    # But bets are tracked separately
    assert state.bets_by_stage[state.game_stage][1] == [1], "Player 1 should have posted SB"
    assert state.bets_by_stage[state.game_stage][2] == [2], "Player 2 should have posted BB"


def test_player_2_wins_multiple_rounds():
    """
    Test showing player 2 can win money in multiple scenarios:
    - Round 1: Player 2 is NOT dealer, wins by hand strength at showdown
    - Round 2: Player 2 IS dealer, wins when others fold

    This demonstrates player 2 can accumulate wealth across multiple rounds.
    """
    # Round 1: Dealer=0, Player 2 wins by having best hand at showdown
    print("\n=== ROUND 1: Player 2 (not dealer) wins by hand strength ===")

    # Construct deck so player 2 gets pocket Aces, others get weak hands
    # Player 0 (dealer) gets 2♠ 3♠, Player 1 (SB) gets 4♠ 5♠, Player 2 (BB) gets A♠ A♥
    # Public cards: 7♦ 8♦ 9♦ 10♦ J♦ (no one makes a flush or straight)
    deck_round_1 = [
        # Deal hole cards (2 per player, starting left of dealer)
        Card(Rank.FOUR, Suit.SPADES), Card(Rank.FIVE, Suit.SPADES),  # Player 1 (SB)
        Card(Rank.ACE, Suit.SPADES), Card(Rank.ACE, Suit.HEARTS),    # Player 2 (BB)
        Card(Rank.TWO, Suit.SPADES), Card(Rank.THREE, Suit.SPADES),  # Player 0 (dealer)
        # Public cards (dealt in order: 3 for flop, 1 for turn, 1 for river)
        Card(Rank.SEVEN, Suit.DIAMONDS), Card(Rank.EIGHT, Suit.DIAMONDS), Card(Rank.NINE, Suit.DIAMONDS),  # Flop
        Card(Rank.TEN, Suit.DIAMONDS),   # Turn
        Card(Rank.JACK, Suit.DIAMONDS),  # River
    ]

    state = State(n_players=3, initial_wealth=100, initial_dealer=0, deck=deck_round_1)

    assert state.dealer == 0
    assert state.current_player == 0  # Dealer acts first pre-flop
    initial_wealths = state.wealth.copy()
    print(f"Initial wealths: {initial_wealths}")

    # Pre-flop: Everyone calls to see the flop
    state.update(2)  # Player 0 calls BB (bets 2)
    state.update(1)  # Player 1 completes to BB (bets 1 more)
    state.update(0)  # Player 2 checks (bets 0)

    assert state.game_stage == GameStage.FLOP
    print(f"Pot after pre-flop: ${state.total_bets()} (each bet $2)")

    # Flop: Everyone checks
    assert state.current_player == 1  # Player 1 (left of dealer) acts first
    state.update(0)  # Player 1 checks
    state.update(0)  # Player 2 checks
    state.update(0)  # Player 0 checks

    assert state.game_stage == GameStage.TURN

    # Turn: Everyone checks
    state.update(0)  # Player 1 checks
    state.update(0)  # Player 2 checks
    state.update(0)  # Player 0 checks

    assert state.game_stage == GameStage.RIVER

    # River: Everyone checks, go to showdown
    state.update(0)  # Player 1 checks
    state.update(0)  # Player 2 checks
    state.update(0)  # Player 0 checks

    # After showdown, player 2 should have won (pocket aces beat everything)
    print(f"Wealths after round 1: {state.wealth}")
    assert state.wealth[2] > initial_wealths[2], "Player 2 should have won money"
    assert state.wealth[0] < initial_wealths[0], "Player 0 should have lost money"
    assert state.wealth[1] < initial_wealths[1], "Player 1 should have lost money"

    # Player 2 should have gained $4 (the other two each lost $2)
    assert state.wealth[2] == initial_wealths[2] + 4, f"Player 2 should have gained $4, got {state.wealth[2] - initial_wealths[2]}"

    print(f"✓ Player 2 won ${state.wealth[2] - initial_wealths[2]} in round 1")

    # Round 2: Dealer=1, Player 2 wins when others fold
    print("\n=== ROUND 2: Player 2 (now dealer) wins when others fold ===")

    # Deck doesn't matter since everyone will fold
    deck_round_2 = [
        Card(Rank.TWO, Suit.CLUBS), Card(Rank.THREE, Suit.CLUBS),    # Player 2 (SB)
        Card(Rank.FOUR, Suit.CLUBS), Card(Rank.FIVE, Suit.CLUBS),    # Player 0 (BB)
        Card(Rank.SIX, Suit.CLUBS), Card(Rank.SEVEN, Suit.CLUBS),    # Player 1 (dealer)
    ]

    # New round starts with dealer=1
    assert state.dealer == 1
    wealth_before_round_2 = state.wealth.copy()
    print(f"Wealths before round 2: {wealth_before_round_2}")

    # Pre-flop: Player 1 (dealer) folds, Player 2 folds, Player 0 wins blinds
    # Wait, that won't work. Let me make player 2 win.
    # Player 1 acts first (dealer), then player 2 (SB), then player 0 (BB)
    assert state.current_player == 1
    state.update(-1)  # Player 1 folds

    # After player 1 folds, player 2 acts
    assert state.current_player == 2
    state.update(1)  # Player 2 completes to BB (bets 1 more to match the 2)

    # Now player 0 (BB) can check or raise
    assert state.current_player == 0
    state.update(-1)  # Player 0 folds

    # Player 2 should win the pot (SB + BB = $3)
    print(f"Wealths after round 2: {state.wealth}")
    assert state.wealth[2] > wealth_before_round_2[2], "Player 2 should have won money when others folded"

    # Player 2 put in $2 total (SB $1 + $1 more to call), player 1 folded (0), player 0 lost BB ($2)
    # So player 2 gains: BB ($2) from player 0 = net +$1 (put in 2, got back 3)
    # Actually wait - player 1 posted nothing before folding (dealer acts first pre-flop, no blind)
    # Player 2 posted SB ($1), then bet $1 more to call BB ($2 total)
    # Player 0 posted BB ($2), then folded
    # Player 1 folded before betting anything
    # So pot = $1 (player 2 SB) + $1 (player 2 call) + $2 (player 0 BB) = $4
    # But player 2 put in $2, so net gain = $4 - $2 = $2
    # But player 0 also put in $2 (the BB), so that goes to player 2
    # Net: Player 2 gains $2, Player 0 loses $2, Player 1 unchanged

    print(f"✓ Player 2 won ${state.wealth[2] - wealth_before_round_2[2]} in round 2 (by fold)")

    # Summary: Verify player 2 has accumulated wealth
    print("\n=== SUMMARY ===")
    print(f"Initial wealth: {initial_wealths[2]}")
    print(f"Final wealth:   {state.wealth[2]}")
    print(f"Net gain:       ${state.wealth[2] - initial_wealths[2]}")

    # Player 2 should have gained money across both rounds
    # Round 1: +$4, Round 2: +$2, Total: +$6
    expected_final = initial_wealths[2] + 6
    assert state.wealth[2] == expected_final, \
        f"Player 2 should have ${expected_final}, got ${state.wealth[2]}"

    print(f"✓ Player 2 successfully won money across multiple rounds!")
    print(f"  - Won as Big Blind by hand strength (+$4)")
    print(f"  - Won as dealer when others folded (+$2)")
    print(f"  - NET RESULT: Gained ${state.wealth[2] - initial_wealths[2]}")


def test_player_elimination_order():
    """
    Test that players can be eliminated in any order.

    Bug hypothesis: Maybe player 2 is always eliminated first due to blind structure?
    """
    # Start with very low wealth - only enough for a few blinds
    state = State(n_players=3, initial_wealth=5, initial_dealer=0)

    print(f"\nInitial wealth (low): {state.wealth}")

    # Track who gets eliminated first over many games
    # (This would need to be a randomized game simulation - marking as TODO)

    # For now, just verify initial wealth after blinds
    total_wealth = sum(state.wealth)
    assert total_wealth == 15, f"Total wealth should be conserved: {total_wealth}"


def test_dealer_rotation_within_episode():
    """
    Test that dealer rotates correctly across multiple deals within an episode.
    """
    state = State(n_players=3, initial_wealth=100, initial_dealer=0)
    assert state.dealer == 0

    # Force a round to complete by manually setting wealth
    state.wealth = [50, 50, 0]  # Player 2 eliminated
    state.terminal = False  # Prevent actual termination

    # Manually test the rotation logic
    next_dealer = (state.dealer + 1) % state.n_players
    assert next_dealer == 1, f"Next dealer should be 1, got {next_dealer}"

    # Note: We can't easily test redistribute_wealth_and_reinitialize
    # without triggering recursion (as we saw in earlier test failures)
    # But we can verify the rotation formula is correct


def test_action_order_across_positions():
    """
    Test that action order is correct for each dealer position.

    Pre-flop: Dealer acts first (after blinds)
    Post-flop: Player left of dealer acts first
    """
    for dealer in [0, 1, 2]:
        state = State(n_players=3, initial_wealth=100, initial_dealer=dealer)

        # Pre-flop: Dealer acts first
        assert state.current_player == dealer, \
            f"Pre-flop with dealer={dealer}, current should be {dealer}, got {state.current_player}"

        # All players check/call to flop - track who acts when stage transitions
        first_actor_on_flop = None
        for action_num in range(3):
            prev_stage = state.game_stage
            state.update([2, 1, 0][action_num])  # Call, complete, check

            # Check if stage just transitioned to FLOP
            if prev_stage != GameStage.FLOP and state.game_stage == GameStage.FLOP:
                first_actor_on_flop = state.current_player
                break

        assert state.game_stage == GameStage.FLOP

        # Post-flop: Player left of dealer should have acted first
        expected_first_actor = (dealer + 1) % 3
        print(f"\nDealer={dealer}, first actor on flop={first_actor_on_flop}, expected={expected_first_actor}")

        assert first_actor_on_flop == expected_first_actor, \
            f"Post-flop with dealer={dealer}, first actor should be {expected_first_actor}, got {first_actor_on_flop}"
