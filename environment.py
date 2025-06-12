# Cody Jackson, Jad Saad, Rafael Puente
# CS 441 Final Programming Project
# 6/13/2025
# Tic-Tac-Toe with Q-Learning

"""
environment.py — Tic-Tac-Toe environment for reinforcement learning.

This module implements the Tic-Tac-Toe game as a reinforcement learning environment,
providing methods to track game state, execute moves, and compute rewards.
"""

from typing import List, Tuple, Optional, Union

class TicTacToeEnv:
    """
    Tic-Tac-Toe environment for reinforcement learning.
    
    This environment implements the game of Tic-Tac-Toe as a reinforcement learning
    environment following a similar interface to OpenAI Gym. It maintains the game
    state and provides methods for taking actions and observing rewards.
    
    Attributes:
        board (List[str]): The game board represented as a list of 9 characters (' ', 'X', or 'O')
    """
    
    def __init__(self) -> None:
        """Initialize a new Tic-Tac-Toe environment with an empty board."""
        self.board: List[str] = []
        self.reset()

    def reset(self) -> str:
        """
        Reset the environment to the starting state (empty board).
        
        Returns:
            str: The initial state representation
        """
        self.board = [' ' for _ in range(9)]
        return self.get_state()

    def get_state(self) -> str:
        """
        Convert the board to a string representation for easier hashing in Q-table.
        
        Returns:
            str: The current board state as a 9-character string (e.g., 'XO X O   ')
        """
        return ''.join(self.board)

    def available_actions(self) -> List[int]:
        """
        Get a list of available (empty) positions on the board.
        
        Returns:
            List[int]: Indices of empty positions (0-8)
        """
        return [i for i, cell in enumerate(self.board) if cell == ' ']

    def step(self, action: int, player: str) -> Tuple[str, float, bool, Optional[str]]:
        """
        Execute a move by placing the player's mark at the specified position.
        
        Args:
            action (int): Board position to place the mark (0-8)
            player (str): The player making the move ('X' or 'O')
            
        Returns:
            Tuple[str, float, bool, Optional[str]]: (next_state, reward, done, winner)
                - next_state (str): The resulting board state
                - reward (float): Reward value (1 for win, -1 for loss, 0 otherwise)
                - done (bool): Whether the game is finished
                - winner (Optional[str]): The winning player ('X', 'O') or None
                
        Raises:
            ValueError: If the specified position is already occupied
        """
        if self.board[action] != ' ':
            raise ValueError(f"Invalid action: Cell {action} is already filled.")

        self.board[action] = player
        winner = self.check_winner()

        if winner == player:
            return self.get_state(), 1.0, True, winner
        elif winner is not None:
            return self.get_state(), -1.0, True, winner
        elif ' ' not in self.board:
            return self.get_state(), 0.0, True, None
        else:
            return self.get_state(), 0.0, False, None

    def check_winner(self) -> Optional[str]:
        """
        Check if there's a winner on the board.
        
        Checks all rows, columns, and diagonals for a winning condition.
        
        Returns:
            Optional[str]: The winning player ('X', 'O') or None if no winner
        """
        b = self.board
        lines = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
            (0, 3, 6), (1, 4, 7), (2, 5, 8),  # columns
            (0, 4, 8), (2, 4, 6)              # diagonals
        ]
        for i, j, k in lines:
            if b[i] == b[j] == b[k] and b[i] != ' ':
                return b[i]
        return None

    def render(self) -> None:
        """
        Print a human-readable representation of the board.
        
        Uses the utils.print_board function to display the current board state.
        """
        from utils import print_board
        print_board(self.get_state()) 