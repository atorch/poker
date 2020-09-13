import numpy as np

from poker.cards import Card, card_index, FULL_DECK, Rank, Suit
from poker.state import GameStage, State


class SimpleQFunctionPlayer:
    def __init__(self, player_index=0, actions=[-1, 0, 1, 2, 3]):

        self.player_index = player_index

        n_game_stages = len(GameStage)
        n_cards = len(FULL_DECK)

        self.actions = actions
        n_actions = len(self.actions)

        # Note: this is a very crude (incomplete) representation of the player's state:
        #  It tracks the game stage and the player's hole cards (i.e. their private cards),
        #  and nothing else. It is also a wasteful representation, because many of the 52*52
        #  possible combinations of cards are duplicates (and some are impossible)
        self.q = np.zeros((n_game_stages, n_cards, n_cards, n_actions))

    def get_private_state(self, game_state):

        game_stage = game_state.game_stage
        first_hole_card = game_state.hole_cards[self.player_index][0]
        second_hole_card = game_state.hole_cards[self.player_index][1]

        first_hole_card_index = card_index(first_hole_card)
        second_hole_card_index = card_index(second_hole_card)

        return game_stage, first_hole_card_index, second_hole_card_index

    def random_legal_action_index(self, minimum_legal_bet):

        action_probabilities = []
        for action in self.actions:
            if action < 0 or (action >= 0 and action >= minimum_legal_bet):
                action_probabilities.append(1.0)
            else:
                action_probabilities.append(0.0)

        action_probabilities = np.array(action_probabilities) / sum(
            action_probabilities
        )
        return np.random.choice(range(len(self.actions)), p=action_probabilities)

    def get_action(self, game_state, proba_random_action=0.8):

        private_state = self.get_private_state(game_state)
        q_at_private_state = self.q[private_state].copy()

        minimum_legal_bet = game_state.minimum_legal_bet()

        if np.random.uniform() < proba_random_action:

            action_index = self.random_legal_action_index(minimum_legal_bet)
            return self.actions[action_index], action_index

        for index, action in enumerate(self.actions):
            if action >= 0 and action < minimum_legal_bet:
                # Note: we temporarily set q to -Inf at illegal actions, so that
                #  those actions cannot be returned by argmax
                q_at_private_state[index] = -np.inf

        action_index = np.argmax(q_at_private_state)
        return self.actions[action_index], action_index


def describe_learned_q_function(q):

    # Note: the value function takes the max over actions (the last index in the q function)
    value = np.max(q, axis=-1)

    argmax_value = np.argmax(value)
    argmax_value = np.unravel_index(argmax_value, value.shape)

    print(f"Argmax of value function: {argmax_value} with value {value[argmax_value]}")
    first_card_index = argmax_value[0]
    second_card_index = argmax_value[1]
    print(
        f"Hole cards at argmax of value: {FULL_DECK[first_card_index]}, {FULL_DECK[second_card_index]}"
    )

    argmin_value = np.argmin(value)
    argmin_value = np.unravel_index(argmin_value, value.shape)

    print(f"Argmin of value function: {argmin_value} with value {value[argmin_value]}")
    first_card_index = argmin_value[0]
    second_card_index = argmin_value[1]
    print(
        f"Hole cards at argmin of value: {FULL_DECK[first_card_index]}, {FULL_DECK[second_card_index]}"
    )

    argmax_q = np.argmax(q)
    argmax_q = np.unravel_index(argmax_q, q.shape)

    print(f"Argmax of Q: {argmax_q} with value {q[argmax_q]}")
    first_card_index = argmax_q[0]
    second_card_index = argmax_q[1]
    print(
        f"Hole cards at argmax of Q: {FULL_DECK[first_card_index]}, {FULL_DECK[second_card_index]}"
    )

    argmin_q = np.argmin(q)
    argmin_q = np.unravel_index(argmin_q, q.shape)

    print(f"Argmin of Q: {argmin_q} with value {q[argmin_q]}")
    first_card_index = argmin_q[0]
    second_card_index = argmin_q[1]
    print(
        f"Hole cards at argmin of Q: {FULL_DECK[first_card_index]}, {FULL_DECK[second_card_index]}"
    )

    index_ace_hearts = card_index(Card(Rank.ACE, Suit.HEARTS))
    index_ace_clubs = card_index(Card(Rank.ACE, Suit.CLUBS))

    q_with_ace = q[GameStage.RIVER, index_ace_hearts, index_ace_clubs, :]
    print(f"Q function at the river with two aces (hearts and clubs): {q_with_ace}")

    # Note: in this naive implementation, these two states are treated differently,
    #  even though they should be identical (because the order of your private cards does not matter)
    # TODO Only one of these is possible with sorting
    q_with_ace = q[GameStage.RIVER, index_ace_clubs, index_ace_hearts, :]
    print(f"Q function at the river with two aces (clubs and hearts): {q_with_ace}")


def run_sarsa(n_players, n_steps=100, learning_rate=0.01):

    # This is (roughly) Sutton and Barto Figure 6.9

    players = [SimpleQFunctionPlayer(player_index) for player_index in range(n_players)]

    initial_wealth = 100
    state = State(n_players=n_players, initial_wealth=initial_wealth, verbose=True)

    learning_player = state.current_player

    private_state = players[learning_player].get_private_state(state)
    action, action_index = players[learning_player].get_action(state)

    cumulative_reward = 0

    for step in range(n_steps):

        # if step % 100_000 == 0:
        #     print(step)

        wealth_before_action = state.wealth[learning_player]

        # TODO Need to check whether state is terminal,
        #  For example, you just folded your last dollar and you are out of the game
        #  If so, there is no continuation value
        state.update(action)

        # Note: we need to make the other players act so that we can get back to the learning player
        #  Even though the other (non-learning) players are acting, it
        #  is possible that the learning player is forced
        #  to act (bet the small blind or big blind) in this block
        while state.current_player != learning_player:

            action, _ = players[state.current_player].get_action(state)
            state.update(action)

        # Note: we calculate the player's reward _after_ the other players act
        # TODO This can create odd rewards when the other players fold and
        #  the learning player is either small or big blind in the next round
        wealth_after_action = state.wealth[learning_player]
        reward = wealth_after_action - wealth_before_action
        cumulative_reward += reward
        print(f" *** Reward for player {learning_player} is ${reward} ***")

        # TODO Put this in pytest
        assert state.wealth[learning_player] == initial_wealth + cumulative_reward

        next_private_state = players[learning_player].get_private_state(state)

        # TODO Probability of random action needs to decrease#
        #  over time, as the player learns their Q function
        next_action, next_action_index = players[learning_player].get_action(state)

        continuation_value = players[learning_player].q[next_private_state][
            next_action_index
        ]

        # TODO Need to check whether next state is terminal
        #  If so, there is no continuation value
        updated_guess_for_q = reward + continuation_value

        players[learning_player].q[private_state][action_index] = players[
            learning_player
        ].q[private_state][action_index] + learning_rate * (
            updated_guess_for_q
            - players[learning_player].q[private_state][action_index]
        )

        action = next_action
        action_index = next_action_index
        private_state = next_private_state

    describe_learned_q_function(players[learning_player].q)


def main(n_players=3):

    run_sarsa(n_players)


if __name__ == "__main__":
    main()
