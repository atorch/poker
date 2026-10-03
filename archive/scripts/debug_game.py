from poker.state import State, GameStage
from poker.random_agent import RandomAgent
import numpy as np

# Play ONE game with verbose logging
agents = [RandomAgent(i, [-1, 0, 1, 2]) for i in range(3)]
state = State(n_players=3, initial_wealth=20, initial_dealer=0, verbose=False)

print(f'=== ROUND 1 ===')
print(f'Dealer: {state.dealer}, Current: {state.current_player}')
print(f'Bets: {state.bets_by_stage[GameStage.PRE_FLOP]}')
print(f'Wealth: {state.wealth}')

action_count = 0
round_num = 1
for step in range(200):
    if state.terminal:
        break

    # Track round transitions
    prev_dealer = state.dealer
    prev_stage = state.game_stage

    player = state.current_player
    action = agents[player].get_action(state, proba_random_action=1.0)

    state.update(action)

    # Detect round transition (new deal)
    if state.dealer != prev_dealer and not state.terminal:
        round_num += 1
        print(f'\n=== ROUND {round_num} (dealer: {prev_dealer} -> {state.dealer}) ===')
        print(f'Current: {state.current_player}')
        print(f'Bets: {state.bets_by_stage[GameStage.PRE_FLOP]}')
        print(f'Wealth: {state.wealth}')

        # Check: is it player 2's turn to act first?
        if state.current_player == 2:
            print(f'  WARNING: Player 2 acts first (dealer={state.dealer})')

print(f'\nGame ended after {step+1} steps')
print(f'Final wealth: {state.wealth}')
print(f'Winner: Player {np.argmax(state.wealth)}')
