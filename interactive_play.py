#!/usr/bin/env python3
"""
Interactive poker game where you play against 2 AI agents.

Usage:
    uv run python interactive_play.py              # Normal mode (hidden opponents' cards)
    uv run python interactive_play.py --full-info  # Full info mode (see opponents' cards)

The AI agents use the latest trained model from models/player_0_latest.h5
You are player 0, and you'll be prompted for actions each turn.
"""

import argparse
import sys
import numpy as np

from poker.state import State, GameStage
from poker.agent import Agent, softmax_with_temperature
from poker.config import TYPICAL_INITIAL_WEALTH, DEFAULT_ACTIONS, describe_action
from poker.cards import Card


def display_showdown(state, human_player_index, full_info=False):
    """
    Display the showdown result after a deal completes.

    Args:
        state: Current game State object (contains last_deal_* attributes)
        human_player_index: Index of the human player
        full_info: If True, show all hole cards (otherwise only show for active players)
    """
    if state.last_deal_winners is None:
        return  # No deal has completed yet

    print("\n" + "="*70)
    print("SHOWDOWN")
    print("="*70)

    # Show winner(s)
    winner_labels = []
    for winner_idx in state.last_deal_winners:
        label = "YOU" if winner_idx == human_player_index else f"Player {winner_idx}"
        winner_labels.append(label)

    winners_str = " and ".join(winner_labels)
    pot_str = f"${state.last_deal_pot:.0f}"

    if state.last_deal_won_by_fold:
        # Win by fold
        folded_labels = []
        for folded_idx in state.last_deal_folded_players:
            label = "You" if folded_idx == human_player_index else f"Player {folded_idx}"
            folded_labels.append(label)
        folded_str = ", ".join(folded_labels)

        print(f"\n{winners_str} wins {pot_str}!")
        print(f"Reason: {folded_str} folded\n")
    else:
        # Win by showdown
        print(f"\n{winners_str} wins {pot_str}!")
        print()

        # Show all players' hands (active and folded)
        print("Final hands:")
        for i in range(len(state.last_deal_hole_cards)):
            player_label = "YOU" if i == human_player_index else f"Player {i}"
            hole_cards_str = ", ".join(str(card) for card in state.last_deal_hole_cards[i])

            # Show hand info
            if i in state.last_deal_folded_players:
                print(f"  {player_label}: {hole_cards_str} [FOLDED]")
            else:
                hand_desc = state.last_deal_hand_descriptions[i]
                hand_str = state.last_deal_hand_strengths[i]
                winner_marker = " 🏆" if i in state.last_deal_winners else ""
                print(f"  {player_label}: {hole_cards_str}")
                print(f"    → {hand_desc} (strength: {hand_str:.1f}){winner_marker}")

        # Show community cards
        if state.last_deal_public_cards:
            public_cards_str = ", ".join(str(card) for card in state.last_deal_public_cards)
            print(f"\n  Community: {public_cards_str}")

    print("\n" + "="*70)


def display_game_state(state, human_player_index, full_info=False):
    """
    Display the current game state from the human player's perspective.

    Args:
        state: Current game State object
        human_player_index: Index of the human player (should be 0)
        full_info: If True, show opponents' hole cards (cheat mode)
    """
    print("\n" + "="*70)
    print(f"GAME STATE - Deal #{state.n_deals}")
    print("="*70)

    # Show game stage
    stage_names = {
        GameStage.PRE_FLOP: "PRE-FLOP",
        GameStage.FLOP: "FLOP",
        GameStage.TURN: "TURN",
        GameStage.RIVER: "RIVER"
    }
    print(f"Stage: {stage_names[state.game_stage]}")

    # Show pot
    pot = state.total_bets()
    print(f"Pot: ${pot:.0f}")

    # Show public cards
    if state.game_stage != GameStage.PRE_FLOP:
        public_cards_str = ", ".join(str(card) for card in state.public_cards)
        print(f"Community cards: {public_cards_str}")
    else:
        print(f"Community cards: (none yet)")

    print()

    # Show each player's status
    for i in range(state.n_players):
        player_label = "YOU" if i == human_player_index else f"AI {i}"
        wealth = state.wealth[i]
        total_bet = state.total_bet_by_player(i)
        folded = state.has_folded[i]
        is_current = (i == state.current_player)

        status = ""
        if folded:
            status = " [FOLDED]"
        elif is_current:
            status = " [ACTING NOW]"

        print(f"Player {i} ({player_label}){status}")
        print(f"  Wealth: ${wealth:.0f}")
        print(f"  Total bet this hand: ${total_bet:.0f}")

        # Show hole cards
        if i == human_player_index:
            # Always show human player's cards
            cards_str = ", ".join(str(card) for card in state.hole_cards[i])
            print(f"  Hole cards: {cards_str}")
        elif full_info:
            # Show opponent cards in full info mode
            cards_str = ", ".join(str(card) for card in state.hole_cards[i])
            print(f"  Hole cards: {cards_str} [CHEAT MODE]")
        else:
            # Hide opponent cards in normal mode
            print(f"  Hole cards: [hidden]")

        print()

    print("="*70)


def get_human_action(state):
    """
    Prompt the human player for an action.

    Args:
        state: Current game State object

    Returns:
        int: The chosen action
    """
    min_bet = state.minimum_legal_bet()
    max_bet = state.maximum_legal_bet()

    print(f"\nYour turn to act!")

    # Build action menu using DEFAULT_ACTIONS (what the AI was trained with)
    print(f"\nAvailable actions:")

    legal_default_actions = []
    for action in DEFAULT_ACTIONS:
        # Check if this action is legal
        if action < 0:  # Fold is always legal
            action_desc = describe_action(action, min_bet)
            print(f"  {int(action)}: {action_desc}")
            legal_default_actions.append(action)
        elif min_bet <= action <= max_bet:  # Check/Call/Bet/Raise must be in legal range
            action_desc = describe_action(action, min_bet)
            print(f"  {int(action)}: {action_desc}")
            legal_default_actions.append(action)

    # Get input - only allow actions from legal_default_actions
    while True:
        try:
            action_str = input("\nEnter action: ").strip()
            action = int(action_str)

            # Validate action is in legal_default_actions
            if action in legal_default_actions:
                return action
            else:
                # Show what actions are actually legal
                legal_values = ", ".join(str(int(a)) for a in legal_default_actions)
                print(f"Invalid action! Choose from: {legal_values}")
        except ValueError:
            print("Invalid input! Please enter a number.")
        except (EOFError, KeyboardInterrupt):
            print("\n\nExiting game...")
            sys.exit(0)


def play_interactive_game(model_path="models/player_0_latest.h5", full_info=False, initial_wealth=TYPICAL_INITIAL_WEALTH, max_deals=50):
    """
    Play an interactive game against AI agents.

    Args:
        model_path: Path to the trained model file
        full_info: If True, show opponents' hole cards
        initial_wealth: Starting wealth for each player
        max_deals: Maximum number of deals before game ends

    Returns:
        int: Winning player index
    """
    n_players = 3
    human_player_index = 0

    print("\n" + "="*70)
    print("INTERACTIVE POKER")
    print("="*70)
    print(f"You are Player {human_player_index}")
    mode_text = "FULL INFORMATION (you can see opponents' cards!)" if full_info else "NORMAL (opponents' cards hidden)"
    print(f"Mode: {mode_text}")
    print(f"Starting wealth: ${initial_wealth:.0f}")
    print(f"Actions: -1=Fold, 0=Check/Call, positive numbers=Bet/Raise")
    print("="*70)

    # Create AI agents
    print(f"\nLoading AI agents from {model_path}...")

    ai_agents = []
    for i in range(1, n_players):
        agent = Agent(
            player_index=i,
            n_players=n_players,
            learning_rate=0.0003,  # Doesn't matter for frozen play
            hidden_layers=(128, 128),  # Must match training config
            temperature=1.0
        )

        # Load the trained model
        if not agent.load_model(model_path):
            print(f"Warning: Could not load model from {model_path}")
            print(f"AI agents will use random initialization (they won't play well!)")

        ai_agents.append(agent)

    print(f"AI agents loaded successfully!\n")

    # Create game state
    state = State(
        n_players=n_players,
        initial_wealth=initial_wealth,
        initial_dealer=0,
        verbose=False
    )

    # Main game loop
    while not state.terminal:
        if state.n_deals > max_deals:
            print(f"\n\nGame ended after {max_deals} deals (max limit reached)")
            break

        current_player = state.current_player

        # Only display full game state when it's the human's turn
        if current_player == human_player_index:
            display_game_state(state, human_player_index, full_info)

            # Human player's turn
            action = get_human_action(state)

            # Show what we're doing
            action_desc = describe_action(action, state.minimum_legal_bet())
            print(f"\n>>> You {action_desc}")
        else:
            # AI agent's turn - just show a compact update
            agent_idx = current_player - 1  # AI agents are indexed 1, 2, ... but stored at 0, 1, ...
            agent = ai_agents[agent_idx]

            # In full-info mode, show AI's action probabilities for debugging
            if full_info:
                # Compute Q-values and probabilities (same logic as agent.get_action)
                private_state = agent.get_private_state(state)
                model_input = agent.get_model_input(private_state, agent.actions)
                q_values = agent.model.predict(model_input, verbose=0)[:, 0]

                # Apply legality constraints
                min_bet = state.minimum_legal_bet()
                max_bet = state.maximum_legal_bet()
                for index, act in enumerate(agent.actions):
                    if act >= 0 and ((act < min_bet) or (act > max_bet)):
                        q_values[index] = -np.inf

                # Compute softmax probabilities
                action_probs = softmax_with_temperature(q_values, agent.temperature)

                # Show probabilities for legal actions
                print(f"\n  [AI {current_player} Policy]")
                for index, act in enumerate(agent.actions):
                    if action_probs[index] > 0:  # Only show legal actions
                        act_desc = describe_action(act, min_bet)
                        print(f"    action={int(act)} ({act_desc}): {100*action_probs[index]:.1f}% (Q={q_values[index]:.2f})")

            # Get action from AI (no exploration, frozen policy)
            action = agent.get_action(state, proba_random_action=0.0)

            # Show compact AI action (before state update so we can see what they added)
            old_bet = state.total_bet_by_player(current_player)
            if action < 0:
                print(f">>> Player {current_player} (AI) folds")
            else:
                # Action is the amount they're adding this betting round
                print(f">>> Player {current_player} (AI) action={action} (total bet: ${old_bet:.0f} → ${old_bet + action:.0f})")

        # Track deal number before update
        deal_before_update = state.n_deals

        # Update game state
        state.update(action)

        # Check if a deal just completed (n_deals incremented)
        if state.n_deals > deal_before_update and not state.terminal:
            # Display showdown result
            display_showdown(state, human_player_index, full_info)

    # Game over - show results
    # Note: The last deal's showdown was already displayed via display_showdown()
    print("\n" + "="*70)
    print("GAME OVER")
    print("="*70)
    print(f"Total deals played: {state.n_deals}")
    print(f"\nFinal wealth:")
    for i in range(n_players):
        player_label = "YOU" if i == human_player_index else f"AI {i}"
        print(f"  Player {i} ({player_label}): ${state.wealth[i]:.0f}")

    # Determine winner
    winner = int(np.argmax(state.wealth))
    winner_label = "YOU" if winner == human_player_index else f"AI {winner}"

    print(f"\n{'='*70}")
    if winner == human_player_index:
        print(f"🎉 YOU WIN! 🎉")
    else:
        print(f"AI {winner} wins")
    print(f"{'='*70}\n")

    return winner


def main():
    parser = argparse.ArgumentParser(
        description="Play interactive poker against AI agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python interactive_play.py              # Normal mode
  uv run python interactive_play.py --full-info  # See opponents' cards
  uv run python interactive_play.py --wealth 100 # Start with $100 each
        """
    )

    parser.add_argument(
        "--full-info",
        action="store_true",
        help="Show opponents' hole cards (cheat mode)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="models/player_0_latest.h5",
        help="Path to trained model file (default: models/player_0_latest.h5)"
    )

    parser.add_argument(
        "--wealth",
        type=float,
        default=TYPICAL_INITIAL_WEALTH,
        help=f"Initial wealth for each player (default: {TYPICAL_INITIAL_WEALTH})"
    )

    parser.add_argument(
        "--max-deals",
        type=int,
        default=50,
        help="Maximum number of deals before game ends (default: 50)"
    )

    args = parser.parse_args()

    try:
        play_interactive_game(
            model_path=args.model,
            full_info=args.full_info,
            initial_wealth=args.wealth,
            max_deals=args.max_deals
        )
    except KeyboardInterrupt:
        print("\n\nGame interrupted by user. Goodbye!")
        sys.exit(0)


if __name__ == "__main__":
    main()
