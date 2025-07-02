import numpy as np
import random
import pygame
import asyncio
import platform

# Grid definition
grid = [
    ['W', 'W', 'W', 'W', 'B', 'W', 'W', 'W', 'W', 'W', 'W', 'B'],
    ['W', 'B', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'B'],
    ['B', 'W', 'B', 'W', 'W', 'W', 'B', 'W', 'B', 'B', 'W', 'W'],
    ['W', 'B', 'W', 'W', 'W', 'W', 'W', 'W', 'B', 'W', 'W', 'W'],
    ['W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'G'],
    ['X', 'X', 'X', 'X', 'W', 'W', 'B', 'W', 'X', 'X', 'X', 'X'],
    ['X', 'X', 'X', 'X', 'W', 'W', 'W', 'W', 'X', 'X', 'X', 'X'],
    ['X', 'X', 'X', 'X', 'W', 'W', 'W', 'W', 'X', 'X', 'X', 'X'],
    ['X', 'X', 'X', 'X', 'W', 'W', 'B', 'W', 'X', 'X', 'X', 'X'],
    ['X', 'X', 'X', 'X', 'R', 'W', 'W', 'W', 'X', 'X', 'X', 'X']
]

# Parameters
num_agents = 3
rows, cols = len(grid), len(grid[0])
cell_size = 50
width, height = cols * cell_size, rows * cell_size
actions = ['north', 'south', 'east', 'west']
action_map = {'north': (-1, 0), 'south': (1, 0), 'east': (0, 1), 'west': (0, -1)}
rewards = {'R': 1, 'W': 1, 'B': -100, 'G': 100, 'X': -float('inf')}
colors = {'R': (255, 0, 0), 'W': (255, 255, 255), 'B': (0, 0, 0), 
          'G': (0, 255, 0), 'X': (128, 128, 128)}
agent_colors = [(255, 165, 0), (0, 165, 255), (255, 0, 255)]  # Orange, Blue, Magenta
gamma = 0.9
alpha = 0.1
epsilon = 0.3
episodes = 1000
FPS = 60
move_delay = 0.1

# Initialize Q-tables for each agent
q_tables = [np.zeros((rows, cols, len(actions))) for _ in range(num_agents)]

# Find start position
start_pos = None
for i in range(rows):
    for j in range(cols):
        if grid[i][j] == 'R':
            start_pos = (i, j)
            break
    if start_pos:
        break

def is_valid_move(state, action):
    row, col = state
    d_row, d_col = action_map[action]
    new_row, new_col = row + d_row, col + d_col
    if (0 <= new_row < rows and 0 <= new_col < cols and 
        grid[new_row][new_col] not in ['B', 'X']):
        return True, (new_row, new_col)
    return False, state

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
    pygame.display.set_caption("Multi-Agent Q-Learning Grid")
    draw_grid([start_pos] * num_agents)

def draw_grid(agent_positions):
    screen.fill((200, 200, 200))
    for i in range(rows):
        for j in range(cols):
            rect = pygame.Rect(j * cell_size, i * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, colors[grid[i][j]], rect)
            pygame.draw.rect(screen, (0, 0, 0), rect, 1)
    # Draw agents
    for idx, (row, col) in enumerate(agent_positions):
        agent_rect = pygame.Rect(col * cell_size + 10, row * cell_size + 10, 
                                cell_size - 20, cell_size - 20)
        pygame.draw.ellipse(screen, agent_colors[idx], agent_rect)
    pygame.display.flip()

async def update_loop():
    global epsilon
    for episode in range(episodes):
        agent_states = [start_pos for _ in range(num_agents)]
        draw_grid(agent_states)
        await asyncio.sleep(move_delay)
        # Continue until all agents reach the goal
        while not all(grid[state[0]][state[1]] == 'G' for state in agent_states):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            for agent_idx in range(num_agents):
                if grid[agent_states[agent_idx][0]][agent_states[agent_idx][1]] == 'G':
                    continue  # Skip agents that reached the goal
                action = choose_action(agent_states[agent_idx], agent_idx)
                valid, next_state = is_valid_move(agent_states[agent_idx], action)
                reward = rewards[grid[next_state[0]][next_state[1]]]
                row, col = agent_states[agent_idx]
                next_row, next_col = next_state
                action_idx = actions.index(action)
                q_tables[agent_idx][row, col, action_idx] = \
                    (1 - alpha) * q_tables[agent_idx][row, col, action_idx] + \
                    alpha * (reward + gamma * np.max(q_tables[agent_idx][next_row, next_col]))
                agent_states[agent_idx] = next_state
            draw_grid(agent_states)
            await asyncio.sleep(move_delay)
        epsilon = max(0.1, epsilon * 0.995)
    
    # Display final policy for each agent
    for agent_idx in range(num_agents):
        print(f"\nOptimal Policy for Agent {agent_idx + 1}:")
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