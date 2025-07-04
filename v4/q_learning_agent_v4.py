import numpy as np
import random
import pygame
import asyncio
import platform
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CELL_SIZE, GAMMA, ALPHA, EPSILON, EPISODES, MOVE_DELAY, FPS
from utils.grid_utils import grid, rows, cols, actions, action_map, rewards, colors, get_start_pos, is_valid_move

width, height = cols * CELL_SIZE, rows * CELL_SIZE
epsilon = EPSILON

q_table = np.zeros((rows, cols, len(actions)))
start_pos = get_start_pos()
agent_color = (255, 0, 0)  # Single agent color (red)

def choose_action(state):
    if random.random() < epsilon:
        return random.choice(actions)
    else:
        row, col = state
        return actions[np.argmax(q_table[row, col])]

def setup():
    global screen
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("V1: Single-Agent Q-Learning")
    draw_grid(start_pos)

def draw_grid(agent_position):
    screen.fill((200, 200, 200))
    for i in range(rows):
        for j in range(cols):
            rect = pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, colors[grid[i][j]], rect)
            pygame.draw.rect(screen, (0, 0, 0), rect, 1)
    row, col = agent_position
    agent_rect = pygame.Rect(col * CELL_SIZE + 10, row * CELL_SIZE + 10, 
                            CELL_SIZE - 20, CELL_SIZE - 20)
    pygame.draw.ellipse(screen, agent_color, agent_rect)
    pygame.display.flip()

def draw_policy_grid():
    screen.fill((200, 200, 200))
    for i in range(rows):
        for j in range(cols):
            rect = pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, colors[grid[i][j]], rect)
            pygame.draw.rect(screen, (0, 0, 0), rect, 1)
            if grid[i][j] in ['B', 'X', 'G']:
                continue
            best_action = actions[np.argmax(q_table[i, j])]
            center_x, center_y = j * CELL_SIZE + CELL_SIZE // 2, i * CELL_SIZE + CELL_SIZE // 2
            arrow_length = CELL_SIZE // 3
            arrow_head_size = 10
            if best_action == 'north':
                start = (center_x, center_y + arrow_length // 2)
                end = (center_x, center_y - arrow_length // 2)
                head = [(center_x, center_y - arrow_length // 2 - arrow_head_size),
                        (center_x - arrow_head_size // 2, center_y - arrow_length // 2),
                        (center_x + arrow_head_size // 2, center_y - arrow_length // 2)]
            elif best_action == 'south':
                start = (center_x, center_y - arrow_length // 2)
                end = (center_x, center_y + arrow_length // 2)
                head = [(center_x, center_y + arrow_length // 2 + arrow_head_size),
                        (center_x - arrow_head_size // 2, center_y + arrow_length // 2),
                        (center_x + arrow_head_size // 2, center_y + arrow_length // 2)]
            elif best_action == 'east':
                start = (center_x - arrow_length // 2, center_y)
                end = (center_x + arrow_length // 2, center_y)
                head = [(center_x + arrow_length // 2 + arrow_head_size, center_y),
                        (center_x + arrow_length // 2, center_y - arrow_head_size // 2),
                        (center_x + arrow_length // 2, center_y + arrow_head_size // 2)]
            else:  # west
                start = (center_x + arrow_length // 2, center_y)
                end = (center_x - arrow_length // 2, center_y)
                head = [(center_x - arrow_length // 2 - arrow_head_size, center_y),
                        (center_x - arrow_length // 2, center_y - arrow_head_size // 2),
                        (center_x - arrow_length // 2, center_y + arrow_head_size // 2)]
            pygame.draw.line(screen, agent_color, start, end, 3)
            pygame.draw.polygon(screen, agent_color, head)
    pygame.display.flip()

async def update_loop():
    global epsilon
    for episode in range(EPISODES):
        print(f'starting episode {episode}')
        agent_state = start_pos
        draw_grid(agent_state)
        await asyncio.sleep(MOVE_DELAY)
        while grid[agent_state[0]][agent_state[1]] != 'G':
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            action = choose_action(agent_state)
            valid, next_state = is_valid_move(agent_state, action)
            reward = rewards[grid[next_state[0]][next_state[1]]]
            row, col = agent_state
            next_row, next_col = next_state
            action_idx = actions.index(action)
            q_table[row, col, action_idx] = (reward + GAMMA * np.max(q_table[next_row, next_col]))
            agent_state = next_state
            draw_grid(agent_state)
            await asyncio.sleep(MOVE_DELAY)
       # epsilon = max(0.1, epsilon * 0.995)

    draw_policy_grid()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        await asyncio.sleep(0.1)

async def main():
    setup()
    await update_loop()

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())