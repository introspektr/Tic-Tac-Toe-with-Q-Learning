import random
from collections import defaultdict
import pickle

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01, init_q=0.5):
        """
        Q-learning agent for Tic-Tac-Toe.
        Parameters:
            alpha: Learning rate
            gamma: Discount factor
            epsilon: Exploration rate
            epsilon_decay: Rate of decay for epsilon per episode
            min_epsilon: Minimum value for epsilon
            init_q: Initial Q-value for unseen (state, action) pairs (optimistic initialization)
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.init_q = init_q

        # Use custom Q-value initialization
        self.q_table = defaultdict(lambda: self.init_q)

    def get_q(self, state, action):
        """
        Get the Q-value for a given state and action.
        """
        return self.q_table[(state, action)]

    def choose_action(self, state, available_actions):
        """
        Choose an action using epsilon-greedy strategy.
        """
        if random.random() < self.epsilon:
            return random.choice(available_actions)

        # Choose best action based on Q-values
        q_values = [self.get_q(state, a) for a in available_actions]
        max_q = max(q_values)
        best_actions = [a for a, q in zip(available_actions, q_values) if q == max_q]
        return random.choice(best_actions)

    def update(self, state, action, reward, next_state, next_available_actions, done):
        """
        Update the Q-value for the given transition.
        """
        current_q = self.get_q(state, action)

        if done:
            target = reward
        else:
            future_q = max([self.get_q(next_state, a) for a in next_available_actions]) if next_available_actions else 0
            target = reward + self.gamma * future_q

        self.q_table[(state, action)] = current_q + self.alpha * (target - current_q)

    def decay_epsilon(self):
        """
        Decay the exploration rate (epsilon) after an episode.
        """
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def get_epsilon(self):
        """
        Return the current exploration rate.
        """
        return self.epsilon

    def save(self, filepath):
        """
        Save the Q-table to a file using pickle.
        """
        with open(filepath, 'wb') as f:
            pickle.dump(dict(self.q_table), f)

    def load(self, filepath):
        """
        Load the Q-table from a file using pickle, with error handling.
        """
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.q_table = defaultdict(lambda: self.init_q, data)
        except (FileNotFoundError, pickle.UnpicklingError) as e:
            print(f"[Warning] Failed to load Q-table from {filepath}: {e}") 