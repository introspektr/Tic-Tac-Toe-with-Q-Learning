"""
eval.py — Evaluation functionality for the Tic-Tac-Toe Q-learning agent.

This module provides functions to evaluate a trained Q-learning agent's
performance by playing against a random opponent and tracking win/loss statistics.
It also allows a human player to play against the trained agent.
"""

import random
import time
from environment import TicTacToeEnv
from agent import QLearningAgent
from utils import print_board, POSITION_NAMES, visualize_q_values


def get_rule_based_action(env):
    """
    Determine the best move for a rule-based opponent using simple heuristics.
    
    Applies these rules in order:
    1. If the opponent can win this move, take that move.
    2. If the agent can be blocked from winning next move, block it.
    3. Otherwise, pick a random available action.
    
    Args:
        env (TicTacToeEnv): The game environment
        
    Returns:
        int: The selected action (board position 0-8)
    """
    board = env.board
    available = env.available_actions()
    player = 'O'  # The rule-based opponent always plays as O
    opponent = 'X'  # The agent always plays as X
    
    # Check if opponent can win this move
    for action in available:
        # Try the move
        board_copy = board.copy()
        board_copy[action] = player
        
        # Check for winning condition
        lines = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
            (0, 3, 6), (1, 4, 7), (2, 5, 8),  # columns
            (0, 4, 8), (2, 4, 6)              # diagonals
        ]
        for i, j, k in lines:
            if board_copy[i] == board_copy[j] == board_copy[k] == player:
                return action
    
    # Check if agent can be blocked from winning
    for action in available:
        # Try the move for the opponent to see if they would win
        board_copy = board.copy()
        board_copy[action] = opponent
        
        # Check for winning condition
        lines = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
            (0, 3, 6), (1, 4, 7), (2, 5, 8),  # columns
            (0, 4, 8), (2, 4, 6)              # diagonals
        ]
        for i, j, k in lines:
            if board_copy[i] == board_copy[j] == board_copy[k] == opponent:
                return action
    
    # If no winning or blocking move, choose randomly
    return random.choice(available)


def play_against_agent(agent_path='q_table.pkl', visualize=False, delay=0.5):
    """
    Allow a human player to play against the trained agent.
    
    The agent always plays as 'X' and goes first, while the human plays as 'O'.
    The game board is displayed after each move, and the human player inputs
    their moves via the console.
    
    Args:
        agent_path (str): Path to the saved Q-table file (default: 'q_table.pkl')
        visualize (bool): Whether to visualize Q-values for agent's moves (default: False)
        delay (float): Delay in seconds between moves for better readability (default: 0.5)
        
    Returns:
        str or None: The winner ('X' for agent, 'O' for human, None for draw)
    """
    print(f"Loading model from: {agent_path}")
    env = TicTacToeEnv()
    agent = QLearningAgent()
    
    try:
        agent.load(agent_path)
    except FileNotFoundError:
        print(f"Error: Model file '{agent_path}' not found.")
        return None
        
    agent.epsilon = 0  # Ensure greedy play
    
    print("\n" + "="*60)
    print("PLAYING AGAINST THE TRAINED AGENT")
    print("="*60)
    print("You will play as 'O'. The agent plays as 'X' and goes first.")
    print("Enter positions as numbers 0-8, corresponding to:")
    print("0 | 1 | 2")
    print("--+---+--")
    print("3 | 4 | 5")
    print("--+---+--")
    print("6 | 7 | 8")
    print("Or enter position names like 'top-left', 'center', 'bottom-right', etc.")
    print("Type 'quit' to exit the game.")
    print("="*60 + "\n")
    
    state = env.reset()
    done = False
    turn_count = 0
    
    print("Initial board:")
    print_board(state)
    
    while not done:
        turn_count += 1
        
        # Agent's move
        print("\nAgent (X) is thinking...")
        available = env.available_actions()
        
        if visualize:
            print("Agent's Q-values for available moves:")
            visualize_q_values(agent, state)
            
        if delay > 0:
            time.sleep(delay)
            
        action = agent.choose_action(state, available)
        next_state, reward, done, winner = env.step(action, player='X')
        state = next_state
        
        print(f"Agent chose position: {action} ({POSITION_NAMES[action]})")
        print("\nCurrent board:")
        print_board(state)
        
        if done:
            break
            
        # Human player's move
        while True:
            available = env.available_actions()
            
            # Display available positions
            available_positions = [f"{pos} ({POSITION_NAMES[pos]})" for pos in available]
            print(f"\nAvailable positions: {', '.join(available_positions)}")
            
            user_input = input("\nYour move (O): ").strip().lower()
            
            if user_input == 'quit':
                print("Game aborted.")
                return None
                
            try:
                # Check if input is a position name
                if user_input in [name.lower() for name in POSITION_NAMES.values()]:
                    from utils import name_to_index
                    action = name_to_index(user_input)
                else:
                    # Otherwise, assume it's a position number
                    action = int(user_input)
                    
                if action not in available:
                    print(f"Invalid move! Position {action} is already taken or out of range.")
                    continue
                    
                break
            except (ValueError, KeyError):
                print("Invalid input! Please enter a position number (0-8) or position name.")
                
        next_state, reward, done, winner = env.step(action, player='O')
        state = next_state
        
        print("\nCurrent board after your move:")
        print_board(state)
        
    # Game is done
    print("\n" + "="*60)
    print("GAME OVER")
    print("="*60)
    print("Final board:")
    print_board(state)
    
    if winner == 'X':
        print("The agent (X) won!")
    elif winner == 'O':
        print("You (O) won! Congratulations!")
    else:
        print("It's a draw!")
        
    print(f"Game length: {turn_count} moves")
    print("="*60)
    
    return winner


def evaluate(agent_path='q_table.pkl', num_games=1000, verbose=False, sample_games=5, visualize=False, delay=0, opponent_type='random'):
    """
    Evaluate a trained Q-learning agent against a specified opponent type.
    
    This function loads a trained agent from a file and evaluates its
    performance by playing a specified number of games against a random
    or rule-based opponent. It collects statistics on win/loss/draw rates and
    average game length, and can optionally display detailed information about sample games.
    
    Args:
        agent_path (str): Path to the saved Q-table file (default: 'q_table.pkl')
        num_games (int): Number of evaluation games to play (default: 1000)
        verbose (bool): Whether to print detailed information about sample games (default: False)
        sample_games (int): Number of sample games to show detailed info for if verbose=True (default: 5)
        visualize (bool): Whether to visualize Q-values during sample games (default: False)
        delay (float): Delay in seconds between moves for better readability during
                      visualization (default: 0)
        opponent_type (str): Type of opponent to play against - 'random', 'rule-based', or 'human' (default: 'random')
    
    Returns:
        dict: Statistics dictionary with keys 'wins', 'losses', 'draws', and 'avg_game_length'
        
    Notes:
        The agent always plays as 'X' and moves first, while the opponent plays as 'O'.
        When visualize=True, heatmaps of Q-values will be displayed for available actions
        in sample games.
    """
    # Special case for human opponent
    if opponent_type == 'human':
        result = play_against_agent(agent_path, visualize, delay)
        return {'human_game_result': result}
    
    print(f"Loading model from: {agent_path}")
    env = TicTacToeEnv()
    agent = QLearningAgent()
    agent.load(agent_path)
    agent.epsilon = 0  # Ensure greedy play
    
    print(f"Evaluating agent over {num_games} games against a {opponent_type} opponent...")
    if verbose:
        print(f"Showing details for {sample_games} sample games")
    
    stats = {'wins': 0, 'losses': 0, 'draws': 0, 'total_moves': 0}

    for game in range(num_games):
        state = env.reset()
        done = False
        turn_log = []

        while not done:
            # Agent's move
            available = env.available_actions()
            
            # Visualize Q-values before agent makes a move
            if visualize and verbose and game < sample_games:
                print("\n" + "="*60)
                print(f"GAME {game + 1}, TURN {len(turn_log) + 1}")
                print("="*60)
                print("Current board:")
                print_board(state)
                print("\nAgent (X) is calculating best move...")
                visualize_q_values(agent, state)
                if delay > 0:
                    time.sleep(delay)
            
            action = agent.choose_action(state, available)
            next_state, reward, done, winner = env.step(action, player='X')
            turn_log.append(('X', action, next_state))
            state = next_state

            if done:
                if winner == 'X':
                    stats['wins'] += 1
                elif winner == 'O':
                    stats['losses'] += 1
                else:
                    stats['draws'] += 1
                break

            # Opponent's move
            if opponent_type == 'rule-based':
                opp_action = get_rule_based_action(env)
            else:  # default to random
                opp_action = random.choice(env.available_actions())
                
            next_state, opp_reward, done, winner = env.step(opp_action, player='O')
            turn_log.append(('O', opp_action, next_state))
            state = next_state

            if done:
                if winner == 'X':
                    stats['wins'] += 1
                elif winner == 'O':
                    stats['losses'] += 1
                else:
                    stats['draws'] += 1
                break

            if delay > 0 and verbose and game < sample_games:
                time.sleep(delay)

        # Update total number of moves made in this game
        stats['total_moves'] += len(turn_log)
                
        # Verbose output for first few games
        if verbose and game < sample_games:
            print("\n" + "="*60)
            print(f"GAME {game + 1} SUMMARY")
            print("="*60)
            
            print("\nMove sequence:")
            for i, (player, action, resulting_state) in enumerate(turn_log):
                print(f"  Turn {i+1:2d}: Player {player} → {POSITION_NAMES[action]}")
                
            print("\nFinal board:")
            print_board(state)
            
            result = 'Draw' if winner is None else f'Player {winner} wins'
            print(f"\nResult: {result}")
            print(f"Game length: {len(turn_log)} moves")
            print("="*60 + "\n")

    # Calculate average game length
    avg_game_length = stats['total_moves'] / num_games if num_games > 0 else 0
    
    # Summary statistics in a clear tabular format
    print("\n" + "="*60)
    print(f"EVALUATION RESULTS ({num_games} games vs {opponent_type} opponent)")
    print("="*60)
    print(f"  Win rate:   {stats['wins']:5d} / {num_games} ({stats['wins'] / num_games:.2%})")
    print(f"  Loss rate:  {stats['losses']:5d} / {num_games} ({stats['losses'] / num_games:.2%})")
    print(f"  Draw rate:  {stats['draws']:5d} / {num_games} ({stats['draws'] / num_games:.2%})")
    print(f"  Avg game length: {avg_game_length:.2f} moves")
    print("="*60)

    # Add average game length to returned stats
    stats['avg_game_length'] = avg_game_length
    
    return stats 