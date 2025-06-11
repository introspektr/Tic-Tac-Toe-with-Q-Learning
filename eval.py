import random
import time
from environment import TicTacToeEnv
from agent import QLearningAgent
from utils import print_board, POSITION_NAMES, visualize_q_values

def evaluate(agent_path='q_table.pkl', num_games=1000, verbose=False, sample_games=5, visualize=False, delay=0):
    """
    Evaluate a trained Q-learning agent against a random opponent.
    
    Args:
        agent_path: Path to the saved Q-table
        num_games: Number of evaluation games to play
        verbose: Whether to print detailed game information
        sample_games: Number of sample games to print if verbose is True
        visualize: Whether to visualize Q-values for sample games
        delay: Delay in seconds between moves (for better readability)
    
    Returns:
        Dictionary with win/loss/draw statistics
    """
    print(f"Loading model from: {agent_path}")
    env = TicTacToeEnv()
    agent = QLearningAgent()
    agent.load(agent_path)
    agent.epsilon = 0  # Ensure greedy play
    
    print(f"Evaluating agent over {num_games} games against a random opponent...")
    if verbose:
        print(f"Showing details for {sample_games} sample games")
    
    stats = {'wins': 0, 'losses': 0, 'draws': 0}
    start_time = time.time()

    for game in range(num_games):
        state = env.reset()
        done = False
        turn_log = []

        while not done:
            # Agent's move
            available = env.available_actions()
            
            # Visualize Q-values before agent makes a move
            if visualize and verbose and game < sample_games:
                print("\n" + "="*50)
                print(f"GAME {game + 1}, TURN {len(turn_log) + 1}")
                print("="*50)
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

            if delay > 0 and verbose and game < sample_games:
                time.sleep(delay)

        # Verbose output for first few games
        if verbose and game < sample_games:
            print("\n" + "="*50)
            print(f"GAME {game + 1} SUMMARY")
            print("="*50)
            
            print("\nMove sequence:")
            for i, (player, action, resulting_state) in enumerate(turn_log):
                print(f"  Turn {i+1:2d}: Player {player} → {POSITION_NAMES[action]}")
                
            print("\nFinal board:")
            print_board(state)
            
            result = 'Draw' if winner is None else f'Player {winner} wins'
            print(f"\nResult: {result}")
            print("="*50 + "\n")

    elapsed_time = time.time() - start_time

    # Summary statistics in a clear tabular format
    print("\n" + "="*60)
    print(f"EVALUATION RESULTS ({num_games} games, {elapsed_time:.2f} seconds)")
    print("="*60)
    print(f"  Win rate:   {stats['wins']:5d} / {num_games} ({stats['wins'] / num_games:.2%})")
    print(f"  Loss rate:  {stats['losses']:5d} / {num_games} ({stats['losses'] / num_games:.2%})")
    print(f"  Draw rate:  {stats['draws']:5d} / {num_games} ({stats['draws'] / num_games:.2%})")
    print("="*60)

    return stats

if __name__ == "__main__":
    evaluate(verbose=True, sample_games=3, visualize=True, delay=0.5) 