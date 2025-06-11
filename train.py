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
    save_as='q_table.pkl'
):
    """
    Train a Q-learning agent to play Tic-Tac-Toe.
    
    Args:
        num_episodes: Number of training episodes
        alpha: Learning rate
        gamma: Discount factor
        epsilon: Initial exploration rate
        epsilon_decay: Rate of decay for epsilon
        min_epsilon: Minimum value for epsilon
        init_q: Initial Q-value
        stats_interval: Interval for printing statistics
        save_as: Filename to save the trained model
        
    Returns:
        Tuple of (trained_agent, stats_dictionary)
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
    print(f"Output file:    {save_as}")
    print("=" * 60)
    
    env = TicTacToeEnv()
    agent = QLearningAgent(alpha, gamma, epsilon, epsilon_decay, min_epsilon, init_q)

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

            # Opponent plays randomly
            pre_opponent_state = state  # Store state before opponent's move
            opp_action = random.choice(env.available_actions())
            state, opp_reward, done, winner = env.step(opp_action, player='O')

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
                # The agent gets a penalty when the opponent wins
                available_actions = []  # Terminal state has no available actions
                agent.update(pre_agent_state, action, agent_reward, state, available_actions, done)
                break

        agent.decay_epsilon()

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

    # Plot progress
    x = [i * stats_interval for i in range(1, len(win_rates) + 1)]
    plt.figure(figsize=(10, 6))
    plt.plot(x, win_rates, label='Win Rate')
    plt.plot(x, loss_rates, label='Loss Rate')
    plt.plot(x, draw_rates, label='Draw Rate')
    plt.plot(x, epsilons, label='Epsilon', linestyle='--')
    plt.xlabel('Episodes')
    plt.ylabel('Rate / Value')
    plt.title('Training Performance Metrics')
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

if __name__ == "__main__":
    trained_agent, final_stats = train() 