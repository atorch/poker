import numpy as np

from poker.cards import Card, Rank, Suit
from poker.state import GameStage
from poker.q_function import get_model


class Agent:
    def __init__(self, player_index=0, actions=[-1, 0, 1, 2, 3]):

        self.player_index = player_index

        self.actions = actions

        self.len_private_state = 13
        self.n_inputs = self.len_private_state + 1

        # Note: the number of inputs is equal to the length of the private state vector plus one (for the actions)
        self.model = get_model(
            n_actions=len(self.actions), n_inputs=self.n_inputs
        )

    def describe_learned_q_function(self):

        first_hole_cards = [
            Card(Rank.TWO, Suit.HEARTS),
            Card(Rank.SEVEN, Suit.HEARTS),
            Card(Rank.ACE, Suit.HEARTS),
        ]
        second_hole_cards = [Card(Rank.SEVEN, Suit.CLUBS), Card(Rank.ACE, Suit.CLUBS)]

        public_cards = [
            Rank.ACE,
            Suit.DIAMONDS,
            Rank.NINE,
            Suit.DIAMONDS,
            Rank.FOUR,
            Suit.CLUBS,
        ]

        print(f"Value below assume the first three public cards are {public_cards}")

        for first_hole_card in first_hole_cards:
            for second_hole_card in second_hole_cards:
                for stage in [GameStage.FLOP, GameStage.RIVER]:

                    private_state = [
                        stage,
                        first_hole_card.rank,
                        first_hole_card.suit,
                        second_hole_card.rank,
                        second_hole_card.suit,
                    ] + public_cards

                    for action in self.actions:

                        q = self.predicted_q(private_state, action)
                        print(
                            f"Value at stage {stage.name} with {first_hole_card}, {second_hole_card}, action {action}: {q}"
                        )

    def get_private_state(self, game_state):

        # Note: the player is not allowed to see the other player's private cards!

        game_stage = game_state.game_stage
        first_hole_card = game_state.hole_cards[self.player_index][0]
        second_hole_card = game_state.hole_cards[self.player_index][1]

        public_cards = [-1, -1, -1, -1, -1, -1]

        if game_state.game_stage != GameStage.PRE_FLOP:
            public_cards = [
                game_state.public_cards[0].rank,
                game_state.public_cards[0].suit,
                game_state.public_cards[1].rank,
                game_state.public_cards[1].suit,
                game_state.public_cards[2].rank,
                game_state.public_cards[2].suit,
            ]

        own_wealth = game_state.wealth[self.player_index]
        total_bet_by_self = game_state.total_bet_by_player(self.player_index)

        private_state = [
            game_stage,
            first_hole_card.rank,
            first_hole_card.suit,
            second_hole_card.rank,
            second_hole_card.suit,
            own_wealth,
            total_bet_by_self,
        ] + public_cards

        return private_state

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

    def get_model_input(self, private_state, actions):

        model_input = np.zeros((len(actions), self.n_inputs))
        private_state = np.expand_dims(np.array(private_state), 0)
        model_input[0:len(actions), 0:self.len_private_state] = np.repeat(
            private_state, len(actions), 0
        )
        model_input[:, -1] = actions

        return model_input

    def predicted_q(self, private_state, action):

        model_input = self.get_model_input(private_state, actions=[action])
        return self.model.predict(model_input)[0, 0]

    def update_q(self, private_state, action, updated_guess_for_q):

        model_input = self.get_model_input(private_state, actions=[action])
        y = np.array([updated_guess_for_q])

        self.model.fit(
            x=model_input, y=y, epochs=1, batch_size=1, steps_per_epoch=1, verbose=0
        )

    def get_action(self, game_state, proba_random_action=0.8):

        minimum_legal_bet = game_state.minimum_legal_bet()
        maximum_legal_bet = game_state.maximum_legal_bet()

        if np.random.uniform() < proba_random_action:

            return self.random_legal_action(minimum_legal_bet, maximum_legal_bet)

        private_state = self.get_private_state(game_state)

        model_input = self.get_model_input(private_state, self.actions)

        # Note: the model returns predicted action-values of shape (len(self.actions), 1)
        q_at_private_state = self.model.predict(model_input)[:, 0]

        for index, action in enumerate(self.actions):

            if (0 <= action < minimum_legal_bet) or (action > maximum_legal_bet):
                # Note: we temporarily set q to -Inf at illegal actions, so that
                #  those actions cannot be returned by argmax
                q_at_private_state[index] = -np.inf

        # TODO Likely a better idea to let players play randomly,
        #  even when they aren't exploring. Optimal strategy is likely _not_ deterministic play!
        action_index = np.argmax(q_at_private_state)
        return self.actions[action_index]
