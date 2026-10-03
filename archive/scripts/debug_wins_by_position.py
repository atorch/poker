from poker.state import State
from poker.random_agent import RandomAgent
import numpy as np

# Track: when each player wins, what was the final dealer position?
wins_by_player = {0: [], 1: [], 2: []}
total_games = 100

for game in range(total_games):
    agents = [RandomAgent(i, [-1, 0, 1, 2]) for i in range(3)]
    state = State(n_players=3, initial_wealth=20, initial_dealer=game % 3)

    for step in range(200):
        if state.terminal:
            break

        player = state.current_player
        action = agents[player].get_action(state)
        state.update(action)

    if state.terminal:
        winner = np.argmax(state.wealth)
        final_dealer = state.dealer
        wins_by_player[winner].append(final_dealer)

print(f'Results from {total_games} games:')
print()
for player in range(3):
    n_wins = len(wins_by_player[player])
    win_rate = 100.0 * n_wins / total_games
    print(f'Player {player}: {n_wins} wins ({win_rate:.1f}%)')

    if n_wins > 0:
        dealer_counts = {}
        for dealer in wins_by_player[player]:
            dealer_counts[dealer] = dealer_counts.get(dealer, 0) + 1
        print(f'  Final dealer distribution: {dealer_counts}')
    print()

# Check: are all games being won?
total_wins = sum(len(wins_by_player[p]) for p in range(3))
print(f'Total games with a winner: {total_wins}/{total_games}')
