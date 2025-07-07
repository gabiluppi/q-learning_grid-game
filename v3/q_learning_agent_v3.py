import numpy as np
import random
import pygame
import asyncio
import platform
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CELL_SIZE, GAMMA, ALPHA, EPSILON, EPISODES, FPS, MOVE_DELAY, REPEAT_PENALTY
from utils.grid_utils import grid, rows, cols, actions, action_map, rewards, colors, get_start_pos, is_valid_move

width, height = cols * CELL_SIZE, rows * CELL_SIZE
epsilon = EPSILON  # Inicia em 0.9 com decaimento

q_table = np.zeros((rows, cols, len(actions)))  # Tabela Q para um único agente
start_pos = get_start_pos()

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
    pygame.display.set_caption("V3: Single-Agent Q-Learning (With Repeat Penalty)")
    draw_grid(start_pos)

def draw_grid(agent_position):
    screen.fill((200, 200, 200))
    for i in range(rows):
        for j in range(cols):
            rect = pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, colors[grid[i][j]], rect)
            pygame.draw.rect(screen, (0, 0, 0), rect, 1)
    agent_rect = pygame.Rect(agent_position[1] * CELL_SIZE + 10, agent_position[0] * CELL_SIZE + 10, 
                            CELL_SIZE - 20, CELL_SIZE - 20)
    pygame.draw.ellipse(screen, (0, 0, 255), agent_rect)  # Agente azul
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
            pygame.draw.line(screen, (0, 0, 255), start, end, 3)  # Agente azul
            pygame.draw.polygon(screen, (0, 0, 255), head)
    pygame.display.flip()

async def update_loop():
    global epsilon
    success_count = 0
    steps_per_episode = []
    agent_state = start_pos
    # Track visits for this episode
    visit_counts = {(i, j): 0 for i in range(rows) for j in range(cols)}
    draw_grid(agent_state)
    await asyncio.sleep(MOVE_DELAY)
    for episode in range(EPISODES):
        agent_state = start_pos  # Reinicia posição
        draw_grid(agent_state)
        await asyncio.sleep(MOVE_DELAY)
        steps = 0
        # Reset visit counts for the episode
        visit_counts = {(i, j): 0 for i in range(rows) for j in range(cols)}
        while grid[agent_state[0]][agent_state[1]] != 'G':
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            action = choose_action(agent_state)
            valid, next_state = is_valid_move(agent_state, action)
            # Calculate reward with repeat penalty
            reward = rewards[grid[next_state[0]][next_state[1]]]
            visit_counts[next_state] += 1
            if visit_counts[next_state] > 1:
                reward += REPEAT_PENALTY  # Apply penalty for revisiting
            row, col = agent_state
            next_row, next_col = next_state
            action_idx = actions.index(action)
            q_table[row, col, action_idx] = (1 - ALPHA) * q_table[row, col, action_idx] + ALPHA * (reward + GAMMA * np.max(q_table[next_row, next_col]))
            agent_state = next_state
            draw_grid(agent_state)
            await asyncio.sleep(MOVE_DELAY)
            steps += 1
        steps_per_episode.append(steps)
        if grid[agent_state[0]][agent_state[1]] == 'G':
            success_count += 1
        print(f'Episode {episode} completed in {steps} steps')
        epsilon = max(0.1, epsilon * 0.995)  # Decaimento de epsilon

    convergence_mean = sum(steps_per_episode) / len(steps_per_episode) if steps_per_episode else 0
    success_rate = (success_count / EPISODES) * 100
    print(f'Convergência média: {convergence_mean} passos')
    print(f'Taxa de sucesso: {success_rate}%')

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