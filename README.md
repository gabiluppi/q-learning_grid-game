# Q-Learning Grid Game

This project implements a Q-Learning algorithm for a grid-based navigation game with multiple agents. The goal is for agents to move from a starting point (red cell) to a goal (green cell) on a 10x12 grid, avoiding obstacles (black cells) and invalid areas (X cells), while navigating through free cells (white cells).

## Features
- **Grid**: 10x12 with cells: W (white, +1 reward), B (black, -100), G (green, +100), R (red, +1), X (invalid).
- **Actions**: Move north, south, east, west.
- **Action Selection**: 70% exploitation (best Q-value), 30% exploration (random).
- **Visualization**: Pygame-based interface showing agents movements.
- **Versions**:
    - **V1**: Initial implementation without additional penalties
    - **V2**: Adds a step penalty to encourage shorter paths.
    - **V3**: Replace the step penalty with a penalty for revisiting states, further reducing cycling.

## Project Structure
```
q-learning-grid-game/
├── config.py                    # Shared configuration parameters
├── v1/                         # Version 1 (no additional penalty)
│   ├── q_learning_multi_agent_v1.py
│   └── README.md
├── v2/                         # Version 2 (with step penalty)
│   ├── q_learning_multi_agent_v2.py
│   └── README.md
├── v3/                         # Version 3 (with revisiting penalty)
│   ├── q_learning_multi_agent_v3.py
│   └── README.md
├── utils/                      # Shared utilities
│   └── grid_utils.py
├── README.md                   # This file
├── pyproject.toml            # Dependencies
```

## Installation
For local execution:
1. Clone the repository: `git clone https://github.com/gabiluppi/q-learning_grid-game`
2. Install dependencies: `pip install -r requirements.txt`
3. Run a version:
    - V1: `python v1/q_learning_agent_v1.py`
    - V2: `python v2/q_learning_agent_v2.py`
    - V3: `python v3/q_learning_agent_v3.py`

## Requirements
- Python 3.13
- Pygame
- NumPy
See `requirements.txt` for details

## Usage
- Parameters are defined in `config.py` for all versions
- Each version runs a Q-learning simulation with three agents (orange, blue, magenta)
- The grid is visualized with agents moving from R to G
- After training, the optimal policy for each agent is printed
- V1 may show agents cycling due to lack of penalty
- V2 uses a step penalty to reduce cycling
- V3 uses a repeat penalty to reduce cycling