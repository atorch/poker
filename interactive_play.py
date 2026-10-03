#!/usr/bin/env python3
"""
Interactive poker game where you play against 2 AI agents.

Usage:
    uv run python interactive_play.py              # Normal mode (hidden opponents' cards)
    uv run python interactive_play.py --full-info  # Full info mode (see opponents' cards and the AI's policy)
    uv run python interactive_play.py --model models/player_0_latest.h5   # the old Jan 2026 model
    uv run python interactive_play.py --model tag                         # a scripted bot

Both AI seats use the same agent: by default the TD agent trained against the scripted pool
(models/2026-10-03_td_pool/final.pt). --model takes anything poker.benchmark accepts: a .pt
model, an old .h5 / .weights.h5 model, or a scripted agent name.
You are player 0, and you'll be prompted for actions each turn.
"""

import argparse
import sys
import numpy as np

from poker.state import State, GameStage
from poker.actions import game_action
from poker.benchmark import agent_factory, display_name
from poker.config import TYPICAL_INITIAL_WEALTH, DEFAULT_ACTIONS, describe_action
from poker.cards import Card

DEFAULT_MODEL = "models/2026-10-03_td_pool/final.pt"


def describe_policy(seated_agent, state):
    """Lines showing the AI's action probabilities (and Q-values) for --full-info; none for scripted bots."""
    agent = seated_agent.agent
    agent.player_index = seated_agent.player_index
    min_bet = state.minimum_legal_bet()
    lines = []

    if hasattr(agent, "head_values"):
        # Note: poker.dqn.DQNAgent, with Q-values in big blinds and Q(fold) known exactly
        q_values = agent.head_values(state)
        probabilities = agent.head_probabilities(state)
        for head, q in enumerate(q_values):
            if np.isfinite(q):
                action = game_action(head, min_bet)
                lines.append(f"    {describe_action(action, min_bet)}: {100 * probabilities[head]:.1f}% (Q={q:+.2f} BB)")

    elif hasattr(agent, "get_private_state"):
        # Note: the old poker.agent.Agent (Q-values in chips, softmax policy). Imported here so that
        #  playing the new agents doesn't load TensorFlow
        from poker.agent import softmax_with_temperature

        private_state = agent.get_private_state(state)
        q_values = agent.q_values(agent.get_model_input(private_state, agent.actions))
        q_values[~agent.legal_action_mask(state)] = -np.inf
        action_probs = softmax_with_temperature(q_values, agent.temperature)
        for index, act in enumerate(agent.actions):
            if action_probs[index] > 0:
                lines.append(f"    action={int(act)} ({describe_action(act, min_bet)}): "
                             f"{100 * action_probs[index]:.1f}% (Q={q_values[index]:.2f})")
    return lines


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
    print("HAND OVER (everyone else folded)" if state.last_deal_won_by_fold else "SHOWDOWN")
    print("="*70)

    # Show winner(s)
    winner_labels = []
    for winner_idx in state.last_deal_winners:
        label = "YOU" if winner_idx == human_player_index else f"Player {winner_idx}"
        winner_labels.append(label)

    pot_str = f"${state.last_deal_pot:.0f}"
    if len(winner_labels) == 1:
        verb = "win" if winner_labels[0] == "YOU" else "wins"
        result_str = f"{winner_labels[0]} {verb} the {pot_str} pot!"
    else:
        result_str = f"{' and '.join(winner_labels)} split the {pot_str} pot"

    print(f"\n{result_str}")
    print_net_results(state, human_player_index)

    if state.last_deal_won_by_fold:
        folded_labels = []
        for folded_idx in state.last_deal_folded_players:
            label = "You" if folded_idx == human_player_index else f"Player {folded_idx}"
            folded_labels.append(label)
        print(f"Reason: {', '.join(folded_labels)} folded\n")
    else:
        print()

        # Show all players' hands (active and folded)
        print("Final hands:")
        for i in range(len(state.last_deal_hole_cards)):
            player_label = "YOU" if i == human_player_index else f"Player {i}"
            hole_cards_str = ", ".join(str(card) for card in state.last_deal_hole_cards[i])

            # Show hand info
            if i in state.last_deal_folded_players:
                # Note: folded cards stay hidden unless it's your own hand or full-info mode
                if full_info or i == human_player_index:
                    print(f"  {player_label}: {hole_cards_str} [FOLDED]")
                else:
                    print(f"  {player_label}: [FOLDED]")
            else:
                hand_desc = state.last_deal_hand_descriptions[i]
                winner_marker = " 🏆" if i in state.last_deal_winners else ""
                print(f"  {player_label}: {hole_cards_str}")
                print(f"    → {hand_desc}{winner_marker}")

        # Show community cards
        if state.last_deal_public_cards:
            public_cards_str = ", ".join(str(card) for card in state.last_deal_public_cards)
            print(f"\n  Community: {public_cards_str}")

    print("\n" + "="*70)


def print_net_results(state, human_player_index):
    """Print each player's net chip change for the deal that just finished."""
    pot = state.last_deal_pot
    winners = state.last_deal_winners
    for i, bet in enumerate(state.last_deal_bets_by_player):
        player_label = "You" if i == human_player_index else f"Player {i}"
        if i in winners:
            # Note: approximate for split pots with an odd chip (State gives it to one winner)
            net = (pot - sum(state.last_deal_bets_by_player[j] for j in winners)) / len(winners)
        else:
            net = -bet
        print(f"  {player_label}: {net:+.0f}")


def announce_new_stage(state):
    """Print the public cards when play moves to a new stage within the same deal."""
    stage_names = {GameStage.FLOP: "FLOP", GameStage.TURN: "TURN", GameStage.RIVER: "RIVER"}
    public_cards_str = ", ".join(str(card) for card in state.public_cards)
    print(f"\n--- {stage_names[state.game_stage]}: {public_cards_str} (pot ${state.total_bets():.0f}) ---")


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
        print(f"  Stack: ${wealth - total_bet:.0f} behind, ${total_bet:.0f} in the pot (started hand with ${wealth:.0f})")

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

    print(f"\nYour turn to act!")

    # Build action menu using DEFAULT_ACTIONS (what the AI was trained with)
    print(f"\nAvailable actions:")

    legal_default_actions = state.legal_actions(DEFAULT_ACTIONS)
    for action in legal_default_actions:
        print(f"  {int(action)}: {describe_action(action, min_bet)}")

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


def play_interactive_game(model_path=DEFAULT_MODEL, full_info=False, initial_wealth=TYPICAL_INITIAL_WEALTH, max_deals=50,
                          temperature=None):
    """
    Play an interactive game against AI agents.

    Args:
        model_path: Agent spec for both AI seats (a .pt or .h5 model path, or a scripted agent name)
        full_info: If True, show opponents' hole cards and the AI's policy
        initial_wealth: Starting wealth for each player
        max_deals: Maximum number of deals before game ends
        temperature: Softmax temperature for model agents (None: greedy for .pt models, 1.0 for old models)

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

    # Create AI agents (one shared model, seated at 1 and 2)
    print(f"\nLoading AI agents: {display_name(model_path)}...")
    make_ai = agent_factory(model_path, temperature=temperature)
    ai_agents = [make_ai(seat) for seat in range(1, n_players)]

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
                policy_lines = describe_policy(agent, state)
                if policy_lines:
                    print(f"\n  [AI {current_player} Policy]")
                    print("\n".join(policy_lines))

            # Get action from AI (no exploration, frozen policy)
            action = agent.get_action(state)

            # Show compact AI action (before state update so we can see what they added)
            old_bet = state.total_bet_by_player(current_player)
            action_desc = describe_action(action, state.minimum_legal_bet())
            print(f">>> Player {current_player} (AI): {action_desc} (in pot: ${old_bet:.0f} → ${old_bet + max(action, 0):.0f})")

        # Track deal number and stage before update
        deal_before_update = state.n_deals
        stage_before_update = state.game_stage

        # Update game state
        state.update(action)

        # Check if a deal just completed (n_deals incremented, or the game ended)
        if state.n_deals > deal_before_update or state.terminal:
            # Display showdown result
            display_showdown(state, human_player_index, full_info)
        elif state.game_stage != stage_before_update:
            announce_new_stage(state)

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
        default=DEFAULT_MODEL,
        help=f"Agent for both AI seats: a .pt or .h5 model path, or a scripted agent name (default: {DEFAULT_MODEL})"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Softmax temperature for model agents, in Q units (default: greedy for .pt models, 1.0 for old models)"
    )

    parser.add_argument(
        "--wealth",
        type=int,
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
            max_deals=args.max_deals,
            temperature=args.temperature,
        )
    except KeyboardInterrupt:
        print("\n\nGame interrupted by user. Goodbye!")
        sys.exit(0)


if __name__ == "__main__":
    main()
