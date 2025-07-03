# Version 3: Multi-Agent Q-Learning

This directory contains Version 3 of the Q-learning grid game, with improvements over V2 to policy visualization.

## Features
- Three agents navigate from the red cell (R) to the green cell (G).
- Rewards: W (+1), B (-100), G (+100), R (+1), X (invalid).
- Replaces step penalty with revisiting penalty.
- Includes -5 penalty for repeated state visits per episode.
- Action selection: 70% exploitation, 30% exploration.
- Pygame visualization of agent movements and optimal policy.

## Improvements
- The optimal policy visualization with Pygame grid.

## Running
- `python q_learning_multi_agent_v3.py`