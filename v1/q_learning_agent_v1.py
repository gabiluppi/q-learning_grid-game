import numpy as np
import random
import pygame
import asyncio
import platform
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import NUM_AGENTS, CELL_SIZE, GAMMA, ALPHA, EPSILON, EPISODES, MOVE_DELAY, FPS
from utils.grid_utils import grid, rows, cols, actions, action_map, rewards, colors, agent_colors, get_start_pos, is_valid_move


width, height = cols * CELL_SIZE, rows * CELL_SIZE
epsilon = EPSILON

q_tables = [np.zeros((rows, cols, len(actions))) for _ in range(NUM_AGENTS)]
start_pos = get_start_pos()

def choose_action(state, agent_idx):
    if random.random() < epsilon:
        return random.choice(actions)
    else:
        row, col = state
        return actions[np.argmax(q_tables[agent_idx][row, col])]

def setup():
    global screen
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("V1: Multi-Agent Q-Learning (No Step Penalty)")
    draw_grid([start_pos] * NUM_AGENTS)

def draw_grid(agent_positions):
    screen.fill((200, 200, 200))
    for i in range(rows):
        for j in range(cols):
            rect = pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, colors[grid[i][j]], rect)
            pygame.draw.rect(screen, (0, 0, 0), rect, 1)
    for idx, (row, col) in enumerate(agent_positions):
        agent_rect = pygame.Rect(col * CELL_SIZE + 10, row * CELL_SIZE + 10, 
                                CELL_SIZE - 20, CELL_SIZE - 20)
        pygame.draw.ellipse(screen, agent_colors[idx], agent_rect)
    pygame.display.flip()

async def update_loop():
    global epsilon
    for episode in range(EPISODES):
        agent_states = [start_pos for _ in range(NUM_AGENTS)]
        draw_grid(agent_states)
        await asyncio.sleep(MOVE_DELAY)
        while not all(grid[state[0]][state[1]] == 'G' for state in agent_states):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            for agent_idx in range(NUM_AGENTS):
                if grid[agent_states[agent_idx][0]][agent_states[agent_idx][1]] == 'G':
                    continue
                action = choose_action(agent_states[agent_idx], agent_idx)
                valid, next_state = is_valid_move(agent_states[agent_idx], action)
                reward = rewards[grid[next_state[0]][next_state[1]]]
                row, col = agent_states[agent_idx]
                next_row, next_col = next_state
                action_idx = actions.index(action)
                q_tables[agent_idx][row, col, action_idx] = \
                    (1 - ALPHA) * q_tables[agent_idx][row, col, action_idx] + \
                    ALPHA * (reward + GAMMA * np.max(q_tables[agent_idx][next_row, next_col]))
                agent_states[agent_idx] = next_state
            draw_grid(agent_states)
            await asyncio.sleep(MOVE_DELAY)
        epsilon = max(0.1, epsilon * 0.995)
    
    for agent_idx in range(NUM_AGENTS):
        print(f"\nOptimal Policy for Agent {agent_idx + 1} (V1):")
        policy = np.chararray((rows, cols), itemsize=5)
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 'B':
                    policy[i, j] = 'BLOCK'
                elif grid[i][j] == 'G':
                    policy[i, j] = 'GOAL'
                elif grid[i][j] == 'X':
                    policy[i, j] = 'NONE'
                else:
                    best_action = actions[np.argmax(q_tables[agent_idx][i, j])]
                    policy[i, j] = best_action.upper()[:5]
        for row in policy:
            print(' '.join(row.astype(str)))

async def main():
    setup()
    await update_loop()

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())