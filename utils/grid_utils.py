import numpy as np

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


rows, cols = len(grid), len(grid[0])
actions = ['north', 'south', 'east', 'west']
action_map = {
    'north': (-1, 0), 
    'south': (1, 0), 
    'east': (0, 1), 
    'west': (0, -1)
}
rewards = {
    'R': 1, 
    'W': 1, 
    'B': -100, 
    'G': 100, 
    'X': -float('inf')
}
colors = {
    'R': (255, 0, 0), 
    'W': (255, 255, 255), 
    'B': (0, 0, 0), 
    'G': (0, 255, 0), 
    'X': (128, 128, 128)
}
agent_colors = [
    (255, 165, 0), # Orange
    (0, 165, 255), # Blue
    (255, 0, 255)  # Magenta
]

def get_start_pos():
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 'R':
                return (i, j)
    raise ValueError("No start position (R) found in grid")

def is_valid_move(state, action):
    row, col = state
    d_row, d_col = action_map[action]
    new_row, new_col = row + d_row, col + d_col
    if (0 <= new_row < rows and 0 <= new_col < cols and 
        grid[new_row][new_col] not in ['B', 'X']):
        return True, (new_row, new_col)
    return False, state