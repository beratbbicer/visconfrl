import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random

def generate_maze(rows, cols):
    maze = np.zeros((rows, cols), dtype=bool)

    start = (0, 1)
    end = (rows - 1, cols - 2)
    maze[start] = True
    maze[end] = True

    def carve(x, y):
        dir = [(-2, 0), (2, 0), (0, -2), (0, 2)]
        np.random.shuffle(dir)

        for dx, dy in dir:
            nx, ny = x + dx, y + dy
            if (0 < nx < rows) and (0 < ny < cols) and not maze[nx, ny]:
                maze[nx-dx//2, ny-dy//2] = True
                maze[nx, ny] = True
                carve(nx, ny)

    carve(1, 1)

    return maze

def plot_maze(maze, filename):
    plt.figure(figsize=(10, 10))
    plt.imshow(maze, cmap='gray', interpolation='nearest')
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()

def extract_patch(maze, patch_size):
    rows, cols = maze.shape
    patch_rows, patch_cols = patch_size

    if patch_rows > rows or patch_cols > cols:
        raise ValueError("Patch size is larger than the maze itself.")

    top_left_x = random.randint(0, rows - patch_rows)
    top_left_y = random.randint(0, cols - patch_cols)

    patch = maze[top_left_x:top_left_x+patch_rows, top_left_y:top_left_y+patch_cols]

    return patch

def place_red_dot(patch):
    valid_positions = [(x, y) for x in range(patch.shape[0]) for y in range(patch.shape[1]) if patch[x, y]]
    red_dot_position = random.choice(valid_positions)
    print(valid_positions)
    print(red_dot_position)
    rgb_patch = np.stack((patch,) * 3, axis=-1)
    rgb_patch[red_dot_position] = [255, 0, 0]

    fig, ax = plt.subplots()
    ax.imshow(patch, cmap='gray', interpolation='nearest')
    ax.plot(red_dot_position[1], red_dot_position[0], 'ro', markersize=1 * fig.dpi / 8)
    ax.axis('off')
    plt.savefig("patch_reddot", bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    return rgb_patch

MAZE_SIZE = (21, 21)
PATHC_SIZE = (5, 5)

maze = generate_maze(*MAZE_SIZE)
plot_maze(maze, 'maze.png')

maze_patch = extract_patch(maze, PATHC_SIZE)
plot_maze(maze_patch, 'patch.png')

rgb_maze_patch = place_red_dot(maze_patch)

#print(maze)
print(maze_patch)