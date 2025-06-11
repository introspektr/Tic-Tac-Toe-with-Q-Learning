import random
import matplotlib.pyplot as plt
from game import TicTacToeEnv
from agent import QLearningAgent

def train(
    num_episodes=50000,
    alpha=0.1,
    gamma=0.9,
    epsilon=1.0,
    epsilon_decay=0.999,
    min_epsilon=0.01,
    init_q=0.5,
    stats_interval=1000
):
    env = TicTacToeEnv()
    agent = QLearningAgent(alpha, gamma, epsilon, epsilon_decay, min_epsilon, init_q)

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
            available = env.available_actions()
            action = agent.choose_action(state, available)
            next_state, reward, done, winner = env.step(action, player='X')

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

            # Opponent plays randomly; no Q-update here
            opp_action = random.choice(env.available_actions())
            state, opp_reward, done, winner = env.step(opp_action, player='O')

            if done:
                # assign terminal reward to agent
                final_reward = 1 if winner == 'X' else -1 if winner == 'O' else 0
                stats['wins' if winner == 'X' else 'losses' if winner == 'O' else 'draws'] += 1
                # No Q-update after opponent plays
                break

        agent.decay_epsilon()

        if ep % stats_interval == 0:
            total = ep
            win_rates.append(stats['wins'] / total)
            loss_rates.append(stats['losses'] / total)
            draw_rates.append(stats['draws'] / total)
            epsilons.append(agent.get_epsilon())
            print(f"Ep {ep}: Win {win_rates[-1]:.3f}, Loss {loss_rates[-1]:.3f}, "
                  f"Draw {draw_rates[-1]:.3f}, Epsilon {epsilons[-1]:.3f}")

    agent.save('q_table.pkl')

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

    return agent, stats

if __name__ == "__main__":
    trained_agent, final_stats = train()
    print("\nFinal Stats:")
    print(f"Wins: {final_stats['wins']}, "
          f"Losses: {final_stats['losses']}, "
          f"Draws: {final_stats['draws']}") 