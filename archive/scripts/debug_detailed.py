from poker.state import State, GameStage
from poker.random_agent import RandomAgent
import numpy as np

# Focus on ONE round where dealer=2, trace every action
agents = [RandomAgent(i, [-1, 0, 1, 2]) for i in range(3)]

# Keep restarting until we get dealer=2
for attempt in range(100):
    state = State(n_players=3, initial_wealth=20, initial_dealer=2, verbose=False)

    if state.dealer == 2:
        print(f'=== Starting with Dealer=2 ===')
        print(f'After blinds: current_player={state.current_player}')
        print(f'Bets: {state.bets_by_stage[GameStage.PRE_FLOP]}')
        print(f'SB should be player 0, BB should be player 1')
        print()

        # Play until stage transitions to FLOP
        action_num = 0
        while state.game_stage == GameStage.PRE_FLOP and not state.terminal and action_num < 10:
            action_num += 1
            player = state.current_player
            action = agents[player].get_action(state)
            print(f'PRE_FLOP Action {action_num}: Player {player} does {action}')
            state.update(action)
            print(f'  After: current={state.current_player}, stage={state.game_stage.name}')

        if state.game_stage == GameStage.FLOP:
            print(f'\n=== Transitioned to FLOP ===')
            print(f'First to act on flop: Player {state.current_player}')
            print(f'Expected: Player 0 (left of dealer)')

            if state.current_player != 0:
                print(f'❌ BUG: Player {state.current_player} acts first, should be Player 0!')
            else:
                print(f'✓ Correct')

            # Play through FLOP
            action_num = 0
            while state.game_stage == GameStage.FLOP and not state.terminal and action_num < 10:
                action_num += 1
                player = state.current_player
                action = agents[player].get_action(state)
                print(f'FLOP Action {action_num}: Player {player} does {action}')
                state.update(action)
                print(f'  After: current={state.current_player}, stage={state.game_stage.name}')

        break
