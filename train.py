"""
train.py — Training functionality for the Tic-Tac-Toe Q-learning agent.

This module provides functions to train a Q-learning agent to play Tic-Tac-Toe,
either against a random opponent or through self-play, tracking performance metrics.
"""

import random
import time
import matplotlib.pyplot as plt
from environment import TicTacToeEnv
from agent import QLearningAgent

def train(
    num_episodes=50000,
    alpha=0.1,
    gamma=0.9,
    epsilon=1.0,
    epsilon_decay=0.999,
    min_epsilon=0.01,
    init_q=0.5,
    stats_interval=1000,
    save_as='q_table.pkl',
    opponent_type='random'
):
    """
    Train a Q-learning agent to play Tic-Tac-Toe.
    
    This function trains a Q-learning agent by playing a specified number of episodes
    against either a random opponent or against itself (self-play). During training,
    it collects statistics on win/loss/draw rates and displays progress periodically.
    Upon completion, it saves the trained agent to a file and generates a plot of
    performance metrics.
    
    Args:
        num_episodes (int): Number of training episodes (default: 50000)
        alpha (float): Learning rate (default: 0.1)
        gamma (float): Discount factor (default: 0.9)
        epsilon (float): Initial exploration rate (default: 1.0)
        epsilon_decay (float): Rate of decay for epsilon per episode (default: 0.999)
        min_epsilon (float): Minimum value for epsilon (default: 0.01)
        init_q (float): Initial Q-value for new state-action pairs (default: 0.5)
        stats_interval (int): Interval for printing statistics and recording data points (default: 1000)
        save_as (str): Filename to save the trained model (default: 'q_table.pkl')
        opponent_type (str): Type of opponent - 'random' or 'self' (default: 'random')
        
    Returns:
        tuple: (trained_agent, stats_dictionary)
            - trained_agent: The trained QLearningAgent instance
            - stats_dictionary: Dictionary with keys 'wins', 'losses', and 'draws'
            
    Notes:
        - The agent always plays as 'X' and moves first
        - When opponent_type='self', a second agent is created to play as 'O'
        - The function displays a plot of win/loss/draw rates and epsilon over time
        - For self-play, the opponent agent is saved with '_opponent' appended to the filename
    """
    print("=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Episodes:       {num_episodes}")
    print(f"Learning rate:  {alpha}")
    print(f"Discount:       {gamma}")
    print(f"Init epsilon:   {epsilon}")
    print(f"Epsilon decay:  {epsilon_decay}")
    print(f"Min epsilon:    {min_epsilon}")
    print(f"Init Q-value:   {init_q}")
    print(f"Opponent:       {opponent_type}")
    print(f"Output file:    {save_as}")
    print("=" * 60)
    
    env = TicTacToeEnv()
    agent = QLearningAgent(alpha, gamma, epsilon, epsilon_decay, min_epsilon, init_q)

    # For self-play, create a second instance of the agent to play as 'O'
    if opponent_type == 'self':
        # The opponent agent uses the same parameters but maintains its own Q-table
        opponent_agent = QLearningAgent(alpha, gamma, epsilon, epsilon_decay, min_epsilon, init_q)
        print("\nTraining agent against itself (self-play)...")
    else:
        opponent_agent = None
        print("\nTraining agent against random opponent...")
    
    start_time = time.time()
    
    stats = {'wins': 0, 'losses': 0, 'draws': 0}
    win_rates = []
    loss_rates = []
    draw_rates = []
    epsilons = []

    for ep in range(1, num_episodes + 1):
        state = env.reset()
        done = False

        # Agent always plays first as 'X'
        while not done:
            # Store state before agent's move
            pre_agent_state = state
            
            # Agent chooses and takes action
            available = env.available_actions()
            action = agent.choose_action(state, available)
            next_state, reward, done, winner = env.step(action, player='X')

            # Update Q-values based on agent's move
            next_available = [] if done else env.available_actions()
            agent.update(state, action, reward, next_state, next_available, done)
            state = next_state

            if done:
                if winner == 'X':
                    stats['wins'] += 1
                elif winner == 'O':
                    stats['losses'] += 1
                else:
                    stats['draws'] += 1
                break

            # Opponent's move - either self-play or random
            pre_opponent_state = state  # Store state before opponent's move
            
            if opponent_type == 'self':
                # Self-play: opponent agent chooses action
                available = env.available_actions()
                opp_action = opponent_agent.choose_action(state, available)
            else:
                # Random opponent
                opp_action = random.choice(env.available_actions())
                
            # Execute opponent's move
            next_state, opp_reward, done, winner = env.step(opp_action, player='O')
            
            # If self-play, update opponent agent's Q-values
            if opponent_type == 'self':
                next_available = [] if done else env.available_actions()
                opponent_agent.update(state, opp_action, opp_reward, next_state, next_available, done)
                
            # Update state
            state = next_state

            if done:
                # Update statistics
                if winner == 'X':
                    stats['wins'] += 1
                elif winner == 'O':
                    stats['losses'] += 1
                else:
                    stats['draws'] += 1
                
                # Update agent's Q-values based on opponent's terminal move
                # Convert opponent's reward to agent's perspective
                agent_reward = -opp_reward  # Opponent's win is agent's loss
                
                # Important: Update the agent's last action based on the new terminal state
                available_actions = []  # Terminal state has no available actions
                agent.update(pre_agent_state, action, agent_reward, state, available_actions, done)
                break

        # Decay exploration rates
        agent.decay_epsilon()
        if opponent_type == 'self':
            opponent_agent.decay_epsilon()

        if ep % stats_interval == 0:
            total = ep
            win_rates.append(stats['wins'] / total)
            loss_rates.append(stats['losses'] / total)
            draw_rates.append(stats['draws'] / total)
            epsilons.append(agent.get_epsilon())
            
            # More structured progress output
            print(f"Episode {ep:7d}/{num_episodes}: " +
                  f"Win {win_rates[-1]:.3f}, Loss {loss_rates[-1]:.3f}, " +
                  f"Draw {draw_rates[-1]:.3f}, Epsilon {epsilons[-1]:.4f}")

    elapsed_time = time.time() - start_time
    agent.save(save_as)
    
    # If self-play, save the opponent agent as well (optional)
    if opponent_type == 'self':
        opponent_save_file = save_as.replace('.pkl', '_opponent.pkl')
        opponent_agent.save(opponent_save_file)
        print(f"Opponent model saved as: {opponent_save_file}")

    # Plot progress
    x = [i * stats_interval for i in range(1, len(win_rates) + 1)]
    plt.figure(figsize=(10, 6))
    plt.plot(x, win_rates, label='Win Rate')
    plt.plot(x, loss_rates, label='Loss Rate')
    plt.plot(x, draw_rates, label='Draw Rate')
    plt.plot(x, epsilons, label='Epsilon', linestyle='--')
    plt.xlabel('Episodes')
    plt.ylabel('Rate / Value')
    plt.title(f'Training Performance Metrics (vs. {opponent_type})')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Final statistics
    print("\n" + "=" * 60)
    print(f"TRAINING COMPLETE ({elapsed_time:.2f} seconds)")
    print("=" * 60)
    print(f"  Win rate:   {stats['wins']:7d} / {num_episodes} ({stats['wins'] / num_episodes:.2%})")
    print(f"  Loss rate:  {stats['losses']:7d} / {num_episodes} ({stats['losses'] / num_episodes:.2%})")
    print(f"  Draw rate:  {stats['draws']:7d} / {num_episodes} ({stats['draws'] / num_episodes:.2%})")
    print(f"  Final ε:    {agent.get_epsilon():.6f}")
    print("=" * 60)
    print(f"Model saved as: {save_as}")
    print("=" * 60)

    return agent, stats 