"""
agent.py — Q-learning agent implementation for the Tic-Tac-Toe game.

This module implements a Q-learning agent that can learn to play Tic-Tac-Toe
through reinforcement learning, using a Q-table to track state-action values.
"""

import random
from collections import defaultdict
import pickle

class QLearningAgent:
    """
    A Q-learning agent implementation for Tic-Tac-Toe.
    
    This agent learns to play Tic-Tac-Toe using the Q-learning algorithm,
    a model-free reinforcement learning technique. It maintains a Q-table
    that maps state-action pairs to expected rewards (Q-values).
    
    Attributes:
        alpha (float): Learning rate - controls how much to update Q-values (0-1)
        gamma (float): Discount factor - controls the importance of future rewards (0-1)
        epsilon (float): Exploration rate - probability of taking a random action (0-1)
        epsilon_decay (float): Rate at which epsilon decreases after each episode
        min_epsilon (float): Minimum value for epsilon to ensure some exploration
        init_q (float): Initial Q-value for new state-action pairs
        q_table (defaultdict): Table mapping (state, action) pairs to Q-values
    """
    
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01, init_q=0.5):
        """
        Initialize a Q-learning agent.
        
        Args:
            alpha (float): Learning rate (default: 0.1)
            gamma (float): Discount factor (default: 0.9)
            epsilon (float): Initial exploration rate (default: 1.0)
            epsilon_decay (float): Rate of decay for epsilon per episode (default: 0.995)
            min_epsilon (float): Minimum value for epsilon (default: 0.01)
            init_q (float): Initial Q-value for unseen (state, action) pairs (default: 0.5)
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
        
        Args:
            state (str): The current state represented as a string
            action (int): The action (board position 0-8)
            
        Returns:
            float: The Q-value for the state-action pair
        """
        return self.q_table[(state, action)]

    def choose_action(self, state, available_actions):
        """
        Choose an action using epsilon-greedy strategy.
        
        With probability epsilon, chooses a random action (exploration),
        otherwise chooses the action with the highest Q-value (exploitation).
        
        Args:
            state (str): The current state represented as a string
            available_actions (list): List of valid actions (board positions)
            
        Returns:
            int: The chosen action (board position 0-8)
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
        Update the Q-value for the given transition using the Q-learning update rule.
        
        Q(s,a) = Q(s,a) + alpha * [reward + gamma * max_a'(Q(s',a')) - Q(s,a)]
        
        Args:
            state (str): The current state
            action (int): The action taken (board position 0-8)
            reward (float): The reward received
            next_state (str): The resulting state
            next_available_actions (list): List of valid actions in the next state
            done (bool): Whether the episode is done
        """
        current_q = self.get_q(state, action)
        
        # Calculate the target value
        if done:
            # If terminal state, there's no future reward to consider
            max_future_q = 0
        else:
            # Get max Q-value for the next state
            max_future_q = max([self.get_q(next_state, a) for a in next_available_actions]) if next_available_actions else 0
        
        # Calculate target: reward + discounted future value
        target = reward + self.gamma * max_future_q
        
        # Update the Q-value
        new_q = current_q + self.alpha * (target - current_q)
        
        # Store the updated Q-value
        self.q_table[(state, action)] = new_q

    def decay_epsilon(self):
        """
        Decay the exploration rate (epsilon) after an episode.
        
        This gradually reduces exploration over time, allowing the agent
        to exploit its learned knowledge more as training progresses.
        
        Returns:
            float: The new epsilon value
        """
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        return self.epsilon

    def get_epsilon(self):
        """
        Return the current exploration rate.
        
        Returns:
            float: The current epsilon value
        """
        return self.epsilon

    def save(self, filepath):
        """
        Save the Q-table to a file using pickle.
        
        Args:
            filepath (str): Path where the Q-table will be saved
        """
        with open(filepath, 'wb') as f:
            pickle.dump(dict(self.q_table), f)

    def load(self, filepath):
        """
        Load the Q-table from a file using pickle, with error handling.
        
        Args:
            filepath (str): Path to the saved Q-table file
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            pickle.UnpicklingError: If the file is corrupted or invalid
        """
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.q_table = defaultdict(lambda: self.init_q, data)
        except (FileNotFoundError, pickle.UnpicklingError) as e:
            print(f"[Warning] Failed to load Q-table from {filepath}: {e}") 