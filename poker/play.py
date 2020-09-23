import numpy as np

from poker.cards import Card, card_index, FULL_DECK, Rank, Suit
from poker.state import GameStage, State
from poker.q_function import get_q_function_model


class QFunctionPlayer:
    def __init__(self, player_index=0, actions=[-1, 0, 1, 2, 3]):

        self.player_index = player_index

        self.actions = actions

        self.model = get_q_function_model(n_actions=len(self.actions))

    def describe_learned_q_function(self):

        first_hole_card = Card(Rank.ACE, Suit.HEARTS)
        second_hole_card = Card(Rank.ACE, Suit.CLUBS)

        private_state = GameStage.RIVER, first_hole_card.rank, first_hole_card.suit, second_hole_card.rank, second_hole_card.suit

        for action in self.actions:

            q = self.predicted_q(private_state, action)
            print(f"Q function at the river with two aces (hearts and clubs), action {action}: {q}")

    def get_private_state(self, game_state):

        # Note: this is an index into the player's Q function
        #  The player is _not_ allowed to see the other player's private cards!
        game_stage = game_state.game_stage
        first_hole_card = game_state.hole_cards[self.player_index][0]
        second_hole_card = game_state.hole_cards[self.player_index][1]

        return game_stage, first_hole_card.rank, first_hole_card.suit, second_hole_card.rank, second_hole_card.suit

    def random_legal_action(self, minimum_legal_bet, maximum_legal_bet):

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
        return np.random.choice(self.actions, p=action_probabilities)

    def get_model_input(self, private_state, action):

        # TODO Magic numbers for model input size
        model_input = np.zeros((1, 6))
        model_input[0, 0:5] = np.expand_dims(np.array(private_state), 0)
        model_input[0, -1] = action

        return model_input

    def predicted_q(self, private_state, action):

        model_input = self.get_model_input(private_state, action)
        return self.model.predict(model_input)[0, 0]

    def update_q(self, private_state, action, updated_guess_for_q):

        model_input = self.get_model_input(private_state, action)
        y = np.array([updated_guess_for_q])

        self.model.fit(x=model_input, y=y, epochs=1, batch_size=1, steps_per_epoch=1, verbose=0)

    def get_action(self, game_state, proba_random_action=0.8):

        minimum_legal_bet = game_state.minimum_legal_bet()
        maximum_legal_bet = game_state.maximum_legal_bet()

        if np.random.uniform() < proba_random_action:

            return self.random_legal_action(
                minimum_legal_bet, maximum_legal_bet
            )

        for index, action in enumerate(self.actions):

            private_state = self.get_private_state(game_state)
            private_state = np.expand_dims(np.array(private_state), 0)

            # TODO Put this in a function
            model_input = np.zeros((len(self.actions), 6))
            model_input[0:len(self.actions), 0:5] = np.repeat(private_state, len(self.actions), 0)
            model_input[:, 5] = self.actions

            # Note: the model returns predicted action-values of shape (len(self.actions), 1)
            q_at_private_state = self.model.predict(model_input)[:, 0]

            if (0 <= action < minimum_legal_bet) or (action > maximum_legal_bet):
                # Note: we temporarily set q to -Inf at illegal actions, so that
                #  those actions cannot be returned by argmax
                q_at_private_state[index] = -np.inf

        action_index = np.argmax(q_at_private_state)
        return self.actions[action_index]


def run_sarsa(n_players, n_episodes=2):

    # This is (roughly) Sutton and Barto Figure 6.9
    # page 130, TODO compare to page 131
    # page 244

    players = [QFunctionPlayer(player_index) for player_index in range(n_players)]

    for episode in range(n_episodes):

        # Note: the probability of random (exploratory) actions decreases over time
        proba_random_action = 0.02 + 0.98 * np.exp(-episode / 1000)

        initial_wealth = 100
        state = State(n_players=n_players, initial_wealth=initial_wealth, verbose=False)

        learning_player = state.current_player

        private_state = players[learning_player].get_private_state(state)
        action = players[learning_player].get_action(
            state, proba_random_action
        )

        cumulative_reward = 0

        # TODO Push model weights to the other players
        # for player_index in range(n_players):
        #     if player_index != learning_player:
        #         # Note: at the beginning of every episode, we push updates to the q function
        #         #  from the learning player to all other players
        #         players[player_index].q = players[learning_player].q.copy()

        while not state.terminal:

            wealth_before_action = state.wealth[learning_player]

            state.update(action)

            # Note: we need to make the other players act so that we can get back to the learning player
            #  Even though the other (non-learning) players are acting, it
            #  is possible that the learning player is forced
            #  to act (bet the small blind or big blind) in this block
            while state.current_player != learning_player:

                action = players[state.current_player].get_action(
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

            next_action = players[learning_player].get_action(
                state, proba_random_action
            )

            continuation_value = players[learning_player].predicted_q(next_private_state, next_action)

            if state.terminal:
                print(f"Reached a terminal state: player wealths are {state.wealth}")
                updated_guess_for_q = reward

            else:
                updated_guess_for_q = reward + continuation_value

            players[learning_player].update_q(
                private_state, action, updated_guess_for_q
            )

            action = next_action
            private_state = next_private_state

    players[learning_player].describe_learned_q_function()


def main(n_players=3):

    run_sarsa(n_players)


if __name__ == "__main__":
    main()
