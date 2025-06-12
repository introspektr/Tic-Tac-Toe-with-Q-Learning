# Tic-Tac-Toe Q-Learning Agent

A reinforcement learning implementation of Tic-Tac-Toe using Q-learning, where an agent learns to play through self-play or against random opponents.

## Authors
- Cody Jackson
- Jad Saad
- Rafael Puente

*CS 441 Final Programming Project*

## Overview

This project implements a Q-learning agent that learns to play Tic-Tac-Toe through reinforcement learning. The agent maintains a Q-table that maps board states to action values, allowing it to learn the optimal strategy through repeated gameplay.

Key features:
- Training against random opponents or through self-play
- Interactive gameplay against the trained agent
- Rule-based opponent for evaluation
- Visualization of Q-values during gameplay
- Comprehensive statistics tracking

## Installation

### Clone the Repository

```bash
git clone https://github.com/introspektr/Tic-Tac-Toe-with-Q-Learning.git
cd Tic-Tac-Toe-with-Q-Learning
```

### Set Up the Environment

#### Use a Python Virtual Environment

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

The project has a command-line interface with two main modes: `train` and `eval`.

### Training the Agent

To train the agent with default parameters:

```bash
python main.py train
```

#### Training Options

```bash
# Train against random opponent with custom parameters
python main.py train --episodes 100000 --alpha 0.2 --gamma 0.9 --epsilon 1.0 --epsilon-decay 0.9995 --min-epsilon 0.01 --q-init 0.5 --save-as my_model.pkl

# Train using self-play
python main.py train --opponent self --episodes 50000
```

#### Key Training Parameters

- `--episodes`: Number of training episodes (default: 50000)
- `--alpha`: Learning rate (default: 0.1)
- `--gamma`: Discount factor (default: 0.9)
- `--epsilon`: Initial exploration rate (default: 1.0)
- `--epsilon-decay`: Rate at which epsilon decreases (default: 0.999)
- `--min-epsilon`: Minimum value for epsilon (default: 0.01)
- `--q-init`: Initial Q-value for new state-action pairs (default: 0.5)
- `--opponent`: Type of opponent, either 'random' or 'self' (default: 'random')
- `--stats-interval`: Interval for printing statistics (default: 1000)
- `--save-as`: Filename to save the trained model (default: 'q_table.pkl')

### Evaluating the Agent

To evaluate a trained agent:

```bash
python main.py eval --model q_table.pkl
```

#### Evaluation Options

```bash
# Evaluate against random opponent with verbose output
python main.py eval --model q_table.pkl --games 1000 --verbose

# Evaluate against rule-based opponent
python main.py eval --model q_table.pkl --opponent rule-based --games 500

# Play against the agent yourself
python main.py eval --model q_table.pkl --opponent human --visualize --delay 0.5
```

#### Key Evaluation Parameters

- `--model`: Path to the saved Q-table file (default: 'q_table.pkl')
- `--games`: Number of evaluation games to play (default: 1000)
- `--verbose`: Print detailed information about sample games
- `--samples`: Number of sample games to show when verbose is enabled (default: 5)
- `--visualize`: Visualize Q-values during sample games
- `--delay`: Delay in seconds between moves for better readability (default: 0)
- `--opponent`: Type of opponent to play against - 'random', 'rule-based', or 'human' (default: 'random')

### Human Play Instructions

When playing against the agent as a human player:
1. The agent always plays as 'X' and goes first
2. You play as 'O'
3. Enter your moves using position numbers (0-8) or position names:
   ```
   0 | 1 | 2
   --+---+--
   3 | 4 | 5
   --+---+--
   6 | 7 | 8
   ```
   Position names: 'top-left', 'top-middle', 'top-right', 'middle-left', 'center', 'middle-right', 'bottom-left', 'bottom-middle', 'bottom-right'
4. Type 'quit' to exit the game

## Example Workflows

### Quick Start: Train and Play

```bash
# Train an agent against a random opponent
python main.py train --episodes 20000 --save-as quick_model.pkl

# Play against the trained agent
python main.py eval --model quick_model.pkl --opponent human --visualize --delay 0.5
```

### Comprehensive Training and Evaluation

```bash
# Train using self-play for more challenging agent
python main.py train --opponent self --episodes 100000 --save-as self_play_model.pkl

# Evaluate against random opponent
python main.py eval --model self_play_model.pkl --games 1000

# Evaluate against rule-based opponent
python main.py eval --model self_play_model.pkl --opponent rule-based --games 1000

# Play against the agent yourself
python main.py eval --model self_play_model.pkl --opponent human --visualize
```

## Project Structure

- `main.py`: Command-line interface for the project
- `agent.py`: Implementation of the Q-learning agent
- `environment.py`: Tic-Tac-Toe game environment
- `train.py`: Training functionality
- `eval.py`: Evaluation functionality
- `utils.py`: Helper functions for board display and visualization
- `requirements.txt`: Dependencies list

## Requirements

- Python 3.6+
- NumPy >= 1.19.0
- Matplotlib >= 3.0.0 