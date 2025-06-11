import random
from game import TicTacToeEnv
from agent import QLearningAgent

POSITION_NAMES = {
    0: "top-left", 1: "top-middle", 2: "top-right",
    3: "middle-left", 4: "center", 5: "middle-right",
    6: "bottom-left", 7: "bottom-middle", 8: "bottom-right"
}

def print_board(state):
    """
    Print the board state in a readable format.
    """
    # state is a string with 9 characters (' ', 'X', or 'O')
    b = list(state)
    print(f"""
     {b[0]} | {b[1]} | {b[2]}
    ---+---+---
     {b[3]} | {b[4]} | {b[5]}
    ---+---+---
     {b[6]} | {b[7]} | {b[8]}
    """)

def evaluate(agent_path='q_table.pkl', num_games=1000, verbose=False, sample_games=5):
    """
    Evaluate a trained Q-learning agent against a random opponent.
    
    Args:
        agent_path: Path to the saved Q-table
        num_games: Number of evaluation games to play
        verbose: Whether to print detailed game information
        sample_games: Number of sample games to print if verbose is True
    
    Returns:
        Dictionary with win/loss/draw statistics
    """
    env = TicTacToeEnv()
    agent = QLearningAgent()
    agent.load(agent_path)
    agent.epsilon = 0  # Ensure greedy play

    stats = {'wins': 0, 'losses': 0, 'draws': 0}

    for game in range(num_games):
        state = env.reset()
        done = False
        turn_log = []

        while not done:
            # Agent's move
            available = env.available_actions()
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

            # Opponent's move (random)
            opp_action = random.choice(env.available_actions())
            state, _, done, winner = env.step(opp_action, player='O')
            turn_log.append(('O', opp_action, state))

            if done:
                if winner == 'X':
                    stats['wins'] += 1
                elif winner == 'O':
                    stats['losses'] += 1
                else:
                    stats['draws'] += 1
                break

        # Verbose output for first few games
        if verbose and game < sample_games:
            print(f"\n--- Game {game + 1} ---")
            for i, (player, action, resulting_state) in enumerate(turn_log):
                print(f"Turn {i + 1}: Player {player} moved to {POSITION_NAMES[action]}")
                print_board(resulting_state)
            print(f"Result: {'Draw' if winner is None else f'{winner} wins'}")

    # Summary statistics
    print("\nEvaluation Results:")
    print(f"Agent Wins:   {stats['wins']} ({stats['wins'] / num_games:.2%})")
    print(f"Agent Losses: {stats['losses']} ({stats['losses'] / num_games:.2%})")
    print(f"Draws:        {stats['draws']} ({stats['draws'] / num_games:.2%})")

    return stats

if __name__ == "__main__":
    evaluate(verbose=True, sample_games=3) 