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

        # Note: this is an index into the player's Q function
        #  The player is _not_ allowed to see the other player's private cards!
        game_stage = game_state.game_stage
        first_hole_card = game_state.hole_cards[self.player_index][0]
        second_hole_card = game_state.hole_cards[self.player_index][1]

        first_hole_card_index = card_index(first_hole_card)
        second_hole_card_index = card_index(second_hole_card)

        return game_stage, first_hole_card_index, second_hole_card_index

    def random_legal_action_index(self, minimum_legal_bet, maximum_legal_bet):

        action_probabilities = []
        for action in self.actions:

            # Note: negative actions indicate folding, which is always a legal action
            if action < 0 or (minimum_legal_bet <= action <= maximum_legal_bet):
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
        maximum_legal_bet = game_state.maximum_legal_bet()

        if np.random.uniform() < proba_random_action:

            action_index = self.random_legal_action_index(
                minimum_legal_bet, maximum_legal_bet
            )
            return self.actions[action_index], action_index

        for index, action in enumerate(self.actions):
            if (0 <= action < minimum_legal_bet) or (action > maximum_legal_bet):
                # Note: we temporarily set q to -Inf at illegal actions, so that
                #  those actions cannot be returned by argmax
                q_at_private_state[index] = -np.inf

        action_index = np.argmax(q_at_private_state)
        return self.actions[action_index], action_index

    def update_q(self, private_state, action_index, updated_guess_for_q, learning_rate):

        self.q[private_state][action_index] = self.q[private_state][
            action_index
        ] + learning_rate * (updated_guess_for_q - self.q[private_state][action_index])


def describe_learned_q_function(q):

    argmax_q = np.argmax(q)
    argmax_q = np.unravel_index(argmax_q, q.shape)

    first_card_index = argmax_q[0]
    second_card_index = argmax_q[1]
    hole_cards = f"{FULL_DECK[first_card_index]}, {FULL_DECK[second_card_index]}"
    print(f"Argmax of Q: {argmax_q} with value {q[argmax_q]}, hole cards {hole_cards}")

    argmin_q = np.argmin(q)
    argmin_q = np.unravel_index(argmin_q, q.shape)

    first_card_index = argmin_q[0]
    second_card_index = argmin_q[1]
    hole_cards = f"{FULL_DECK[first_card_index]}, {FULL_DECK[second_card_index]}"
    print(f"Argmin of Q: {argmin_q} with value {q[argmin_q]}, hole cards {hole_cards}")

    # Note: the value function takes the max over actions (the last index in the q function)
    value = np.max(q, axis=-1)

    argmin_value = np.argmin(value)
    argmin_value = np.unravel_index(argmin_value, value.shape)

    first_card_index = argmin_value[0]
    second_card_index = argmin_value[1]
    print(
        f"Argmin of value function: {argmin_value} with value {value[argmin_value]}, hole cards {hole_cards}"
    )

    index_ace_hearts = card_index(Card(Rank.ACE, Suit.HEARTS))
    index_ace_clubs = card_index(Card(Rank.ACE, Suit.CLUBS))
    index_ace_spades = card_index(Card(Rank.ACE, Suit.SPADES))

    q_with_ace = q[GameStage.RIVER, index_ace_hearts, index_ace_clubs, :]
    print(f"Q function at the river with two aces (hearts and clubs): {q_with_ace}")

    q_with_ace = q[GameStage.RIVER, index_ace_hearts, index_ace_spades, :]
    print(f"Q function at the river with two aces (hearts and spades): {q_with_ace}")


def run_sarsa(n_players, n_episodes=20_000, learning_rate=0.01):

    # This is (roughly) Sutton and Barto Figure 6.9

    players = [SimpleQFunctionPlayer(player_index) for player_index in range(n_players)]

    for episode in range(n_episodes):

        # Note: the probability of random (exploratory) actions decreases over time
        proba_random_action = 0.02 + 0.98 * np.exp(-episode / 1000)

        initial_wealth = 100
        state = State(n_players=n_players, initial_wealth=initial_wealth, verbose=False)

        learning_player = state.current_player

        private_state = players[learning_player].get_private_state(state)
        action, action_index = players[learning_player].get_action(
            state, proba_random_action
        )

        cumulative_reward = 0

        for player_index in range(n_players):
            if player_index != learning_player:
                # Note: at the beginning of every episode, we push updates to the q function
                #  from the learning player to all other players
                players[player_index].q = players[learning_player].q.copy()

        while not state.terminal:

            wealth_before_action = state.wealth[learning_player]

            state.update(action)

            # Note: we need to make the other players act so that we can get back to the learning player
            #  Even though the other (non-learning) players are acting, it
            #  is possible that the learning player is forced
            #  to act (bet the small blind or big blind) in this block
            while state.current_player != learning_player:

                action, _ = players[state.current_player].get_action(
                    state, proba_random_action
                )
                state.update(action)

                # TODO Is this needed?
                if state.terminal:
                    break

            # Note: we calculate the player's reward _after_ the other players act
            # TODO This can create odd rewards when the other players fold and
            #  the learning player is either small or big blind in the next round
            wealth_after_action = state.wealth[learning_player]
            reward = wealth_after_action - wealth_before_action
            cumulative_reward += reward
            # print(f" *** Reward for player {learning_player} is ${reward} ***")

            # TODO Put this in pytest
            assert state.wealth[learning_player] == initial_wealth + cumulative_reward

            next_private_state = players[learning_player].get_private_state(state)

            next_action, next_action_index = players[learning_player].get_action(
                state, proba_random_action
            )

            continuation_value = players[learning_player].q[next_private_state][
                next_action_index
            ]

            if state.terminal:
                print(f"Reached a terminal state: player wealths are {state.wealth}")
                updated_guess_for_q = reward

            else:
                updated_guess_for_q = reward + continuation_value

            players[learning_player].update_q(
                private_state, action_index, updated_guess_for_q, learning_rate
            )

            action = next_action
            action_index = next_action_index
            private_state = next_private_state

    describe_learned_q_function(players[learning_player].q)


def main(n_players=3):

    run_sarsa(n_players)


if __name__ == "__main__":
    main()
