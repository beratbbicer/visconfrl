import pygame
from pygame.locals import *
import numpy as np
import gymnasium as gym
from collections import namedtuple
from dataclasses import dataclass, field

"""
TODO:
    - Fix reward function:
        Not even stable baselines can learn in this environment as it is
    - Cache maze view and _to_value() to speed up rendering
    - check out https://github.com/Farama-Foundation/Minigrid
    - Check if redundant observations are in replay buffer, consider deleting these
    - Integrate a visual_confidence simulator
    - Consider stabilizing the behavior w.r.t. random seed
    - Figure out how to mix multiple mazes, or whether to reinitialize the agent for each maze
    - Consider starting with a fixed maze design and record user inputs for each step
    - Consider swapping obstacle and empty space values (gapes become 1, obstacles become 0) for conv layers
    - reset uses seed, but confirmation required - especially when maze is being created

Done:
    1. Test the environment with random actions
    2. reset uses seed, but confirmation required - especially when maze is being created
    3. Increased termination reward so that if the goal is reached, it is likely to be positive always, negative otherwise.
        We might need to prune this so that the shortest path is incentivized.
    4. Train TD3 on this environment with a simple agent
        td3_adapted does this, performance to be determined
    5. Render and partial maze view are now aligned. Transposing the partial view in step() fixed it.
        Though I still don't understand why it was necessary.
    6. Convert observations to be a local view on the maze, rather than the entire maze.
    7. Changed reward so legal actions that reduce Manhattan dist bet. agent and target get a small positive reward
        Otherwise, a small negative reward is given.
    8. Played with reward function
        Now reward scales with new Manhattan distance to goal / manhattan distance at start, max value
    9. Fixed observation space
"""

VonNeumannMotion = namedtuple('VonNeumannMotion', 
                              ['north', 'south', 'west', 'east'], 
                              defaults=[[-1, 0], [1, 0], [0, -1], [0, 1]])
@dataclass
class Object:
    name: str
    value: int
    rgb: tuple
    impassable: bool
    positions: list = field(default_factory=list)

class RandomMaze(gym.Env):
    r"""Generate a random maze environment for gymnasium.

    Code based on https://github.com/zuoxingdong/mazelab
    """
    metadata = {"render_modes": ["none", "human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, window_size=1024, width=51, height=51, complexity=0.75, density=0.75, partial_view=False, view_kernel_size=1, render_mode=None):
        r"""Initialize the environment.
    
        Args:
        - :attr:`render_mode` - gymnasium render mode. Can be one of the following:
            - ``human`` - render to a PyGame window
            - ``rgb_array`` - render to a numpy array
        - :attr:`width` - the width of the square grid.
        - :attr:`height` - the height of the square grid.
        - :attr:`complexity` - a parameter that controls the complexity of the maze.
        - :attr:`density` - a parameter that controls the density of the maze.
        """
        
        self.window_size = window_size  # The size of the PyGame window

        assert width > 6 and height > 6, "Maze can be at least 7x."
        self._width = width
        self._height = height
        self._complexity = complexity
        self._density = density
        self._partial_view = partial_view
        self._view_kernel_size = view_kernel_size
        self._start_idx, self._goal_idx = [[1,1]], [[height-2, width-2]]

        self.maze = np.zeros((self._height, self._width), dtype=int)
        self.motions = VonNeumannMotion()

        self.observation_space = gym.spaces.Box(low=np.zeros((view_kernel_size*2+1,view_kernel_size*2+1)),\
                                                high=3+np.zeros((view_kernel_size*2+1,view_kernel_size*2+1)),\
                                                dtype=int)
        self.action_space = gym.spaces.Discrete(len(self.motions))

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

        # Instantiate objects:
        objects = self._make_objects()
        self._objects = namedtuple('Objects', map(lambda x: x.name, objects), defaults=objects)()  

    # Adjusted for readabilitysize based on: https://github.com/zuoxingdong/mazelab/blob/master/mazelab/generators/random_maze.py
    def _random_maze(self):
        r"""Generate a random maze array.

        It only contains two kind of objects, obstacle and free space. The numerical value for obstacle
        is ``1`` and for free space is ``0``.

        Code from https://en.wikipedia.org/wiki/Maze_generation_algorithm
        """

        # Only odd shapes
        shape = ((self._height // 2) * 2 + 1, (self._width // 2) * 2 + 1)

        # Adjust complexity and density relative to maze size
        complexity = int(self._complexity * (5 * (shape[0] + shape[1])))
        density = int(self._density* ((shape[0] // 2) * (shape[1] // 2)))

        # Build actual maze
        Z = np.zeros(shape, dtype=bool)

        # Fill borders
        Z[0, :] = Z[-1, :] = 1
        Z[:, 0] = Z[:, -1] = 1

        # Make aisles
        for i in range(density):
            x, y = (
                np.random.randint(0, shape[1] // 2 + 1) * 2,
                np.random.randint(0, shape[0] // 2 + 1) * 2,
            )

            Z[y, x] = 1

            for j in range(complexity):
                neighbours = []
                if x > 1:
                    neighbours.append((y, x - 2))
                if x < shape[1] - 2:
                    neighbours.append((y, x + 2))
                if y > 1:
                    neighbours.append((y - 2, x))
                if y < shape[0] - 2:
                    neighbours.append((y + 2, x))
                if len(neighbours):
                    y_, x_ = neighbours[np.random.randint(0, len(neighbours))]
                    if Z[y_, x_] == 0:
                        Z[y_, x_] = 1
                        Z[y_ + (y - y_) // 2, x_ + (x - x_) // 2] = 1
                        x, y = x_, y_

        return Z.astype(int)

    def generate_maze(self):
        r"""Generate a random maze.

        This is invoked to generate the maze and initialize maze objects.

        Generated maze is a ``np.ndarray`` of shape ``[self.height, self.width]``. 
            The value of each cell is either ``0`` or ``1``.
            ``0`` represents a free space, and ``1`` represents an obstacle.

        Maze objects are stored in ``self._objects``.
        """
        self.maze = self._random_maze()
        objects = self._make_objects()
        self._objects = namedtuple('Objects', map(lambda x: x.name, objects), defaults=objects)()  
    
    def _make_objects(self):
        r"""Initialize maze objects.

        This is invoked during initialization, and should only be re-invoked if the maze is regenerated.
        """
        free = Object('free', 0, (224, 224, 224), False, np.stack(np.where(self.maze == 0), axis=1))
        obstacle = Object('obstacle', 1, (160, 160, 160), True, np.stack(np.where(self.maze == 1), axis=1))
        agent = Object('agent', 2, (51, 153, 255), False, [])
        goal = Object('goal', 3, (51, 255, 51), False, [])
        return free, obstacle, agent, goal

    def _get_distance_to_goal(self, agent_pos):
        r"""Computes Manhattan distance between agent and goal."""
        return np.abs(np.asarray(self._objects.goal.positions[0]) - np.asarray(agent_pos)).sum()

    def _get_partial_view(self, maze):
        agent_pos = self._objects.agent.positions[0]
        h_min = max(0, agent_pos[0] - self._view_kernel_size)
        h_min_pad = self._view_kernel_size - (agent_pos[0] - h_min)

        h_max = min(self._height-1, agent_pos[0] + self._view_kernel_size)
        h_max_pad = self._view_kernel_size - (h_max - agent_pos[0])

        w_min = max(0, agent_pos[1] - self._view_kernel_size)
        w_min_pad = self._view_kernel_size - (agent_pos[1] - w_min)

        w_max = min(self._width-1, agent_pos[1] + self._view_kernel_size)
        w_max_pad = self._view_kernel_size - (w_max - agent_pos[1])

        state = maze[h_min:h_max+1,w_min:w_max+1]
        # Pad with obstacles if the view is not complete
        if state.shape != (self._view_kernel_size*2+1, self._view_kernel_size*2+1):
            state = np.pad(state, ((h_min_pad, h_max_pad), (w_min_pad, w_max_pad)), constant_values=1)
        dist = self._get_distance_to_goal(agent_pos)
        return state, dist

    def reset(self, seed=None, options=None):
        r"""Reset the gymnasium environment and re_goal_idxturns the initial state.

        Resets the agent and goal positions to the start and goal locations, and returns the initial state of the maze.
        """
        super().reset(seed=seed, options=options)
        self._objects.agent.positions = self._start_idx
        self._objects.goal.positions = self._goal_idx
        maze = self._to_value()

        if self._partial_view:
            state, dist = self._get_partial_view(maze)
            return state, {"distance": dist}
        else:
            return maze, {}
    
    def _to_value(self):
        r"""Fill an empty int array with maze content."""
        x = np.empty(self.maze.shape, dtype=int)
        return self._convert(x, 'value')
    
    def _convert(self, x, name):
        r"""Fills maze-shaped array with object values."""
        for obj in self._objects:
            pos = np.asarray(obj.positions)
            x[pos[:, 0], pos[:, 1]] = getattr(obj, name, None)
        return x

    def _to_impassable(self):
        x = np.empty(self.maze.shape, dtype=bool)
        return self._convert(x, 'impassable')

    def _is_valid(self, position):
        nonnegative = position[0] >= 0 and position[1] >= 0
        within_edge = position[0] < self.maze.shape[0] and position[1] < self.maze.shape[1]
        passable = not self._to_impassable()[position[0]][position[1]]
        return nonnegative and within_edge and passable

    def _is_goal(self, position):
        out = False
        for pos in self._objects.goal.positions:
            if position[0] == pos[0] and position[1] == pos[1]:
                out = True
                break
        return out

    def step(self, action):
        r"""Implements environment's `step(action)` function."""
        motion = self.motions[action]
        current_position = self._objects.agent.positions[0]
        new_position = [current_position[0] + motion[0], current_position[1] + motion[1]]
        valid = self._is_valid(new_position)

        if valid:
            self._objects.agent.positions = [new_position]
        
        if self._is_goal(new_position):
            # reward = +10
            reward = 1
            terminated = True
        elif not valid:
            # reward = -10
            reward = -1
            terminated = True
        else:
            '''
            pre_dist = self._get_distance_to_goal(current_position)
            post_dist = self._get_distance_to_goal(new_position)
            max_dist = self._get_distance_to_goal(self._start_idx[0])
            if pre_dist > post_dist:
                reward = post_dist / max_dist
            else:
                reward = -post_dist / max_dist
            '''
            
            # reward = -0.001
            reward = 1 - self._get_distance_to_goal(new_position) / self._get_distance_to_goal(self._start_idx[0])
            terminated = False

        maze = self._to_value()

        if self._partial_view:
            state, dist = self._get_partial_view(maze)
            return state.T, reward, terminated, False, {"distance": dist}
        else:
            return maze, reward, terminated, False, {}

    def _to_rgb(self):
        x = np.empty((*self.maze.shape, 3), dtype=np.uint8)
        return self._convert(x, 'rgb')

    def render(self):
        r"""Renders the environment for visualization purposes."""
        if self.render_mode is not None:
            return self._render_frame()
    
    def _render_frame(self):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))

        if self.clock is None:
            self.clock = pygame.time.Clock()

        img = self._to_rgb()
        pygame_img = pygame.image.frombuffer(img.tobytes(), img.shape[1::-1], "RGB")
        # pygame_img = pygame.transform.flip(pygame.transform.rotate(pygame_img, 90), True, False)
        # pygame_img = pygame.transform.flip(pygame_img, True, True)
        self.window.blit(pygame.transform.scale(pygame_img, (self.window_size, self.window_size)), (0, 0))
        pygame.event.pump()
        pygame.display.update()

        # We need to ensure that human-rendering occurs at the predefined framerate.
        # The following line will automatically add a delay to keep the framerate stable.
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        r"""Shuts down the enviroment."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

if __name__ == "__main__":
    from gymnasium.envs.registration import register

    register(
        id="RandomMaze-v1.0",
        entry_point=RandomMaze,
    )

    # For exporting the environment to the gym registry
    """

    setup(
        name="gym_examples",
        version="0.0.1",
        install_requires=["gymnasium==0.26.0", "pygame==2.1.0"],
    )

    # Use it with this afterwards: 
    import gym_examples
    env = gymnasium.make('gym_examples/GridWorld-v0')
    """

    env = gym.make("RandomMaze-v1.0", render_mode="human", partial_view=True, view_kernel_size=2)
    test = env.reset() # Set seed
    env.get_wrapper_attr('generate_maze')() # Generate maze & objects
    state = env.reset()[0] # Get initial state

    for _ in range(100):
        env.render()
        action = env.action_space.sample()
        print("Action taken:", action)
        next_state, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print("Environment is reset")
            next_state, info = env.reset()
        state = next_state
    env.close()