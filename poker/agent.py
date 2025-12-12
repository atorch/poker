import os
import numpy as np

from poker.cards import Card, Rank, Suit
from poker.state import GameStage, State
from poker.q_function import get_model


def softmax_with_temperature(q_values, temperature=1.0):
    """
    Compute softmax probabilities over Q-values with temperature parameter.

    Args:
        q_values: Array of Q-values (can include -inf for illegal actions)
        temperature: Temperature parameter (higher = more uniform, lower = more greedy)
            - temperature → 0: approaches argmax (deterministic)
            - temperature = 1: standard softmax
            - temperature → ∞: approaches uniform random

    Returns:
        Array of probabilities (sums to 1, illegal actions have probability 0)
    """
    # Handle illegal actions (Q = -inf)
    legal_mask = np.isfinite(q_values)

    if not np.any(legal_mask):
        raise ValueError("All actions are illegal!")

    # Scale Q-values by temperature
    scaled_q = q_values / temperature

    # Compute softmax only over legal actions (numerical stability)
    legal_q = scaled_q[legal_mask]
    max_q = np.max(legal_q)
    exp_q = np.exp(legal_q - max_q)  # Subtract max for numerical stability

    # Create probability distribution
    probs = np.zeros_like(q_values)
    probs[legal_mask] = exp_q / np.sum(exp_q)

    return probs


class Agent:
    def __init__(self, player_index=0, actions=[-1, 0, 1, 2, 3], n_players=3, temperature=1.0):

        self.player_index = player_index
        self.n_players = n_players
        self.temperature = temperature

        self.actions = actions

        # State: game_stage(1) + hole_cards(4) + own_wealth(1) + own_bets(1) + pot_size(1) + public_cards(6) + opponent_wealths(n_players-1) + opponent_active(n_players-1)
        self.len_private_state = 14 + 2 * (n_players - 1)
        self.n_inputs = self.len_private_state + 1

        # Note: the number of inputs is equal to the length of the private state vector plus one (for the actions)
        self.model = get_model(n_actions=len(self.actions), n_inputs=self.n_inputs)

    def save_model(self, filepath):
        """Save the trained model weights to disk."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save_weights(filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load trained model weights from disk."""
        if os.path.exists(filepath):
            try:
                self.model.load_weights(filepath)
                print(f"Model loaded from {filepath}")
                return True
            except ValueError as e:
                print(f"Warning: Could not load model from {filepath}")
                print(f"  Error: {e}")
                print(f"  This is likely due to a state representation change.")
                print(f"  Starting with fresh model instead.")
                return False
        else:
            print(f"No model found at {filepath}, starting fresh")
            return False

    def describe_learned_q_function(self, n_iter=20):

        for _ in range(n_iter):

            game_state = State(n_players=3)

            private_cards = game_state.hole_cards[self.player_index]
            private_state = self.get_private_state(game_state)

            model_input = self.get_model_input(private_state, self.actions)

            q = self.model.predict(model_input)[:, 0]

            print(
                f"Value at stage {game_state.game_stage.name} with private cards {private_cards}: {q}"
            )

    def sanity_check_learned_strategy(self):
        """
        Sanity check that the learned Q-function makes poker sense.
        Tests specific scenarios that should have obvious optimal Q-value ordering.
        Note: With softmax policy, actions are sampled probabilistically, but we still
        want Q-values to be ordered correctly (e.g., Q(bet) > Q(fold) for pocket aces).
        """
        print("\n" + "="*70)
        print("SANITY CHECK: Testing learned Q-function on key scenarios")
        print("="*70)

        # Test 1: Premium hands should prefer betting over folding
        print("\n[Test 1] Premium starting hands should have highest Q-value for non-fold actions:")
        premium_hands = [
            ([Card(Rank.ACE, Suit.HEARTS), Card(Rank.ACE, Suit.SPADES)], "Pocket Aces"),
            ([Card(Rank.KING, Suit.HEARTS), Card(Rank.KING, Suit.SPADES)], "Pocket Kings"),
            ([Card(Rank.ACE, Suit.HEARTS), Card(Rank.KING, Suit.HEARTS)], "Ace-King suited"),
        ]

        for hole_cards, hand_name in premium_hands:
            game_state = State(n_players=3)
            game_state.hole_cards[self.player_index] = hole_cards
            private_state = self.get_private_state(game_state)
            model_input = self.get_model_input(private_state, self.actions)
            q_values = self.model.predict(model_input, verbose=0)[:, 0]

            action_strs = [f"fold" if a < 0 else f"bet ${a}" for a in self.actions]
            highest_q_idx = np.argmax(q_values)
            highest_q_action = self.actions[highest_q_idx]

            print(f"  {hand_name:20s} -> Highest Q: {action_strs[highest_q_idx]:10s} ", end="")
            if highest_q_action < 0:
                print("❌ Q(fold) is highest (BAD!)")
            else:
                print("✓ Q(non-fold) is highest (good)")

        # Test 2: Worst hands should consider folding
        print("\n[Test 2] Worst starting hands - checking Q-value ordering when facing bets:")
        worst_hands = [
            ([Card(Rank.SEVEN, Suit.HEARTS), Card(Rank.TWO, Suit.CLUBS)], "7-2 offsuit"),
            ([Card(Rank.EIGHT, Suit.HEARTS), Card(Rank.THREE, Suit.CLUBS)], "8-3 offsuit"),
            ([Card(Rank.NINE, Suit.DIAMONDS), Card(Rank.TWO, Suit.CLUBS)], "9-2 offsuit"),
        ]

        for hole_cards, hand_name in worst_hands:
            game_state = State(n_players=3)
            game_state.hole_cards[self.player_index] = hole_cards
            # Simulate that opponent has bet big (we're facing a decision)
            game_state.bets_by_stage[GameStage.PRE_FLOP][1] = [3]

            private_state = self.get_private_state(game_state)
            model_input = self.get_model_input(private_state, self.actions)
            q_values = self.model.predict(model_input, verbose=0)[:, 0]

            action_strs = [f"fold" if a < 0 else f"bet ${a}" for a in self.actions]
            highest_q_idx = np.argmax(q_values)

            print(f"  {hand_name:20s} -> Highest Q: {action_strs[highest_q_idx]:10s}")

        # Test 3: Should never fold when can check for free
        print("\n[Test 3] Q(check) should always exceed Q(fold) when checking is free:")
        for _ in range(5):
            game_state = State(n_players=3)
            private_state = self.get_private_state(game_state)
            model_input = self.get_model_input(private_state, self.actions)
            q_values = self.model.predict(model_input, verbose=0)[:, 0]

            fold_idx = self.actions.index(-1)
            check_idx = self.actions.index(0)

            hole_cards = game_state.hole_cards[self.player_index]
            if q_values[fold_idx] > q_values[check_idx]:
                print(f"  {hole_cards} -> ❌ Q(fold) > Q(check) (BAD!)")
            else:
                print(f"  {hole_cards} -> ✓ Q(check) > Q(fold) (good)")

        # Test 4: Show Q-value trends across hand strengths
        print("\n[Test 4] Q-value comparison across different hand strengths:")
        print(f"  {'Hand Type':25s} {'Fold Q':>10s} {'Check Q':>10s} {'Bet $2 Q':>10s}")
        print("  " + "-"*55)

        test_hands = [
            ([Card(Rank.ACE, Suit.HEARTS), Card(Rank.ACE, Suit.SPADES)], "AA (best)"),
            ([Card(Rank.QUEEN, Suit.HEARTS), Card(Rank.QUEEN, Suit.SPADES)], "QQ (strong)"),
            ([Card(Rank.JACK, Suit.HEARTS), Card(Rank.TEN, Suit.HEARTS)], "JT suited (medium)"),
            ([Card(Rank.NINE, Suit.HEARTS), Card(Rank.FIVE, Suit.CLUBS)], "9-5 offsuit (weak)"),
            ([Card(Rank.SEVEN, Suit.HEARTS), Card(Rank.TWO, Suit.CLUBS)], "7-2 offsuit (worst)"),
        ]

        for hole_cards, hand_name in test_hands:
            game_state = State(n_players=3)
            game_state.hole_cards[self.player_index] = hole_cards
            private_state = self.get_private_state(game_state)
            model_input = self.get_model_input(private_state, self.actions)
            q_values = self.model.predict(model_input, verbose=0)[:, 0]

            fold_idx = self.actions.index(-1)
            check_idx = self.actions.index(0)
            bet2_idx = self.actions.index(2) if 2 in self.actions else 0

            print(f"  {hand_name:25s} {q_values[fold_idx]:10.2f} {q_values[check_idx]:10.2f} {q_values[bet2_idx]:10.2f}")

        print("\n" + "="*70)

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
        pot_size = game_state.total_bets()

        # Include opponent wealths (crucial for strategic play)
        opponent_wealths = [
            game_state.wealth[i]
            for i in range(len(game_state.wealth))
            if i != self.player_index
        ]

        # Include whether each opponent is still active (1) or has folded (0)
        # Critical for pot odds: 1v1 vs 1v2 requires different strategy
        opponent_active = [
            1 if not game_state.has_folded[i] else 0
            for i in range(len(game_state.has_folded))
            if i != self.player_index
        ]

        private_state = [
            game_stage,
            first_hole_card.rank,
            first_hole_card.suit,
            second_hole_card.rank,
            second_hole_card.suit,
            own_wealth,
            total_bet_by_self,
            pot_size,
        ] + public_cards + opponent_wealths + opponent_active

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

    def predicted_q(self, private_state, action):

        model_input = self.get_model_input(private_state, actions=[action])
        return self.model.predict(model_input)[0, 0]

    def get_model_input(self, private_state, actions):

        model_input = np.zeros((len(actions), self.n_inputs))
        private_state = np.expand_dims(np.array(private_state), 0)
        model_input[0 : len(actions), 0 : self.len_private_state] = np.repeat(
            private_state, len(actions), 0
        )
        model_input[:, -1] = actions

        return model_input

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
                #  softmax will assign them probability 0
                q_at_private_state[index] = -np.inf

        # Use softmax policy to sample actions probabilistically
        # This enables mixed strategies, which are essential for poker Nash equilibria
        action_probs = softmax_with_temperature(q_at_private_state, self.temperature)
        action_index = np.random.choice(len(self.actions), p=action_probs)
        return self.actions[action_index]
