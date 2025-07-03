# Version 2: Multi-Agent Q-Learning

This directory contains Version 2 of the Q-learning grid game, with improvements over V1 to reduce cycling.

## Features
- Three agents navigates from the red cell (R) to the green cell (G)
- Rewards: W (+1), B (-100), G (+100), R (+1), X (invalid)
- Includes step penalty, default is -5 per step
- Action selection: 70% exploitation, 30% exploration
- Pygame visualization of agent movements.

## Improvements
- The -5 step penalty, reducing cycling but not fully eliminating

## Running
- `python q_learning_agent_v2.py`