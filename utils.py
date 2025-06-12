# Cody Jackson, Jad Saad, Rafael Puente
# CS 441 Final Programming Project
# 6/13/2025
# Tic-Tac-Toe with Q-Learning

"""
utils.py — Helper functions for the Tic-Tac-Toe Q-learning project.

This module provides utility functions for displaying and manipulating
the Tic-Tac-Toe board, as well as visualizing the agent's Q-values.
"""

from typing import Dict, List, Optional, Tuple, Union
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

# Position mapping for human-readable output
POSITION_NAMES: Dict[int, str] = {
    0: "top-left", 1: "top-middle", 2: "top-right",
    3: "middle-left", 4: "center", 5: "middle-right",
    6: "bottom-left", 7: "bottom-middle", 8: "bottom-right"
}

def print_board(state: str) -> None:
    """
    Print a Tic-Tac-Toe board in a human-readable format.
    
    Args:
        state (str): A 9-character string representation of the board,
                    where each character is 'X', 'O', or ' ' (space)
                    
    Example:
        >>> print_board('XO X O   ')
             X | O |  
            ---+---+---
             X | O |  
            ---+---+---
               |   |  
    """
    b = list(state)
    print(f"""
     {b[0]} | {b[1]} | {b[2]}
    ---+---+---
     {b[3]} | {b[4]} | {b[5]}
    ---+---+---
     {b[6]} | {b[7]} | {b[8]}
    """)

def index_to_name(index: int) -> str:
    """
    Convert a board position index to its human-readable name.
    
    Args:
        index (int): Board position index (0-8)
        
    Returns:
        str: Human-readable position name (e.g., "top-left", "center")
        
    Raises:
        ValueError: If the index is out of bounds
        
    Example:
        >>> index_to_name(4)
        'center'
    """
    if index not in POSITION_NAMES:
        raise ValueError(f"Index {index} is out of bounds (must be 0-8)")
    return POSITION_NAMES[index]

def name_to_index(name: str) -> int:
    """
    Convert a human-readable position name to its board index.
    
    Args:
        name (str): Human-readable position name (e.g., "top-left", "center")
        
    Returns:
        int: Board position index (0-8)
        
    Raises:
        ValueError: If the name is not recognized
        
    Example:
        >>> name_to_index("center")
        4
    """
    for idx, pos_name in POSITION_NAMES.items():
        if pos_name.lower() == name.lower():
            return idx
    raise ValueError(f"Position name '{name}' not recognized")

def visualize_q_values(agent, state: str) -> None:
    """
    Create a heatmap visualization of Q-values for available actions.
    
    Generates a matplotlib figure showing the current board state and the
    Q-values for each available move using a color-coded heatmap.
    
    Args:
        agent: The Q-learning agent with a get_q method
        state (str): Current board state as a 9-character string
        
    Notes:
        - Occupied positions are shown with the player's mark ('X' or 'O')
        - Available positions show the Q-value with 4 decimal places
        - The color scale adapts to the range of Q-values present
    """
    # Create a heatmap of Q-values
    q_values = np.zeros(9)
    
    # Fill with Q-values for all positions
    for i in range(9):
        if state[i] == ' ':  # Only show Q-values for available actions
            q_values[i] = agent.get_q(state, i)
        else:
            q_values[i] = float('nan')  # NaN for occupied positions
    
    # Reshape to 3x3 grid
    q_values = q_values.reshape(3, 3)
    
    # Find non-NaN min and max for better color scaling
    non_nan_values = q_values[~np.isnan(q_values)]
    
    if len(non_nan_values) > 0:
        # Calculate a meaningful color range to highlight differences
        vmin, vmax = non_nan_values.min(), non_nan_values.max()
        
        # If all values are the same, create a small range around it
        if vmin == vmax:
            if vmin == 0:
                vmin, vmax = -0.1, 0.1
            else:
                # Create range of ±10% around the value
                margin = abs(vmin) * 0.1
                vmin -= margin
                vmax += margin
        
        # Ensure we include zero in the range for better intuition
        if vmin > 0:
            vmin = 0
        if vmax < 0:
            vmax = 0
    else:
        # Default range if no valid Q-values
        vmin, vmax = -1, 1
    
    plt.figure(figsize=(6, 6))
    heatmap = plt.imshow(q_values, cmap='coolwarm', interpolation='nearest', vmin=vmin, vmax=vmax)
    
    # Add text annotations with both raw values and normalized for visual clarity
    for i in range(3):
        for j in range(3):
            if not np.isnan(q_values[i, j]):
                # Determine text color based on background intensity
                val = q_values[i, j]
                # Show the raw Q-value with 4 decimal places
                plt.text(j, i, f'{val:.4f}', 
                         ha='center', va='center', 
                         color='black' if abs(val) < (vmax-vmin)*0.7 else 'white',
                         fontsize=12)
            else:
                pos_idx = i * 3 + j
                plt.text(j, i, state[pos_idx], 
                         ha='center', va='center', 
                         color='white', fontsize=14, fontweight='bold')
    
    plt.title('Q-Values for Available Actions')
    plt.colorbar(heatmap, label='Q-Value')
    plt.xticks([])
    plt.yticks([])
    plt.show() 