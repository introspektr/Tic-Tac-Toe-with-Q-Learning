"""
utils.py — Helper functions (board display, position mapping) for
Tic‑Tac‑Toe Q‑learning project.
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
    Pretty-print a Tic-Tac-Toe board given its state representation.
    
    Args:
        state: 9-character string ('X', 'O', or ' ')
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
    Return the human-readable name for a board index.
    
    Args:
        index: Board position index (0-8)
        
    Returns:
        Human-readable position name
        
    Raises:
        ValueError: If index is out of bounds
    """
    if index not in POSITION_NAMES:
        raise ValueError(f"Index {index} is out of bounds (must be 0-8)")
    return POSITION_NAMES[index]

def name_to_index(name: str) -> int:
    """
    Return the board index (0–8) corresponding to a human-readable name.
    
    Args:
        name: Human-readable position name
        
    Returns:
        Board position index
        
    Raises:
        ValueError: If name is not recognized
    """
    for idx, pos_name in POSITION_NAMES.items():
        if pos_name.lower() == name.lower():
            return idx
    raise ValueError(f"Position name '{name}' not recognized")

def visualize_q_values(agent, state: str) -> None:
    """
    Visualize the Q-values for available actions in a given state.
    
    Args:
        agent: QLearningAgent instance
        state: Current board state
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
    
    plt.figure(figsize=(6, 6))
    plt.imshow(q_values, cmap='coolwarm', interpolation='nearest')
    
    # Add text annotations
    for i in range(3):
        for j in range(3):
            if not np.isnan(q_values[i, j]):
                plt.text(j, i, f'{q_values[i, j]:.2f}', 
                         ha='center', va='center', 
                         color='black', fontsize=12)
            else:
                pos_idx = i * 3 + j
                plt.text(j, i, state[pos_idx], 
                         ha='center', va='center', 
                         color='white', fontsize=14, fontweight='bold')
    
    plt.title('Q-Values for Available Actions')
    plt.colorbar(label='Q-Value')
    plt.xticks([])
    plt.yticks([])
    plt.show() 