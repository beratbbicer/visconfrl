import pygame
from pygame.locals import *
import numpy as np
import gymnasium as gym
from collections import namedtuple
from dataclasses import dataclass, field
from pathlib import Path
import random, string, pickle, os

import torch
import torch.nn.functional as F

"""
TODO:
    - Cache maze view and _to_value() to speed up rendering
    - check out https://github.com/Farama-Foundation/Minigrid
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
    10. Current reward function seems appropriate at the moment: Inversely proportional to distance to goal.
    11. Fix generated mazes
    12. Explore from a random location during explorations. This is cheaty, so don't :)
    13. Reproducable mazes by I/O
"""

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
        self.motions = [[-1, 0], [1, 0], [0, -1], [0, 1]] # west, east, north, south
        
        # VonNeumannMotion = namedtuple('VonNeumannMotion', ['north', 'south', 'west', 'east'], defaults=[[-1, 0], [1, 0], [0, -1], [0, 1]])
        # self.motions = VonNeumannMotion()

        self._init_input_spaces()

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
    def _random_maze(self, width, height, complexity, density):
        r"""Generate a random maze array.

        It only contains two kind of objects, obstacle and free space. The numerical value for obstacle
        is ``1`` and for free space is ``0``.

        Code from https://en.wikipedia.org/wiki/Maze_generation_algorithm
        """

        # Only odd shapes
        shape = ((height // 2) * 2 + 1, (width // 2) * 2 + 1)

        # Adjust complexity and density relative to maze size
        complexity = int(complexity * (5 * (shape[0] + shape[1])))
        density = int(density * ((shape[0] // 2) * (shape[1] // 2)))

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

    def _init_input_spaces(self):
        self._view_kernel_size = min(self._view_kernel_size, self._width//2, self._height//2)
        self._start_idx, self._goal_idx = [[1,1]], [[self._height-2, self._width-2]]
        
        if self._partial_view:
            self.observation_space = gym.spaces.Box(low=np.zeros((self._view_kernel_size*2+1,self._view_kernel_size*2+1)),\
                                                    high=3+np.zeros((self._view_kernel_size*2+1,self._view_kernel_size*2+1)),\
                                                    dtype=int)
        else:
            self.observation_space = gym.spaces.Box(low=np.zeros((self._height,self._width)),\
                                                    high=3+np.zeros((self._height,self._width)),\
                                                    dtype=int)
            
        self.action_space = gym.spaces.Discrete(len(self.motions))

    def generate_maze(self, savepath):
        r"""Generate a random maze.

        This is invoked to generate the maze and initialize maze objects.

        Generated maze is a ``np.ndarray`` of shape ``[self.height, self.width]``. 
            The value of each cell is either ``0`` or ``1``.
            ``0`` represents a free space, and ``1`` represents an obstacle.

        Maze objects are stored in ``self._objects``.

        Args:
        - :attr:`savepath` - path to save the maze to. 
            If this path points to a file (pickle archive), the maze will be loaded from this path.
            Class arguments will also be adjusted accordingly. See ``self._init_input_spaces()`` for details.
            Otherwise, a new maze will be generated and dumped to this location.
        """

        if Path(savepath).exists() and Path(savepath).is_file():
            savepath = str(Path(savepath).resolve())
            fields = savepath.split(os.sep)[-1].split('.pkl')[0].split('_')
            for field in fields:
                if 'width' in field:
                    self._width = int(field.split('-')[-1])
                elif 'height' in field:
                    self._height = int(field.split('-')[-1])
                elif 'complexity' in field:
                    self._complexity = float(field.split('-')[-1])
                elif 'density' in field:
                    self._density = float(field.split('-')[-1])
                else:
                    pass

            self._init_input_spaces()

            with open(savepath, 'rb') as f:
                self.maze = pickle.load(f)
        else:
            name = "".join(random.SystemRandom().choice(string.ascii_uppercase+string.ascii_lowercase+string.digits) for _ in range(16))
            name = f"width-{self._width}_height-{self._height}_complexity-{self._complexity}_density-{self._density}_{name}.pkl"
            self.maze = self._random_maze(self._width, self._height, self._complexity, self._density)

            with open(Path(savepath).joinpath(name), 'wb') as f:
                pickle.dump(self.maze, f)

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

    def _get_distance_to_goal(self, goal_position, agent_pos):
        r"""Computes Manhattan distance between agent and goal."""
        # return np.abs(np.asarray(self._objects.goal.positions[0]) - np.asarray(agent_pos)).sum()
        return np.abs(np.asarray(goal_position) - np.asarray(agent_pos)).sum()

    def _get_partial_view(self, maze, goal_position, agent_pos, width, height, view_kernel_size):
        h_min = max(0, agent_pos[0] - view_kernel_size)
        h_min_pad = view_kernel_size - (agent_pos[0] - h_min)

        h_max = min(height-1, agent_pos[0] + view_kernel_size)
        h_max_pad = view_kernel_size - (h_max - agent_pos[0])

        w_min = max(0, agent_pos[1] - view_kernel_size)
        w_min_pad = view_kernel_size - (agent_pos[1] - w_min)

        w_max = min(width-1, agent_pos[1] + view_kernel_size)
        w_max_pad = view_kernel_size - (w_max - agent_pos[1])

        state = maze[h_min:h_max+1,w_min:w_max+1]
        # Pad with obstacles if the view is not complete
        if state.shape != (view_kernel_size*2+1, view_kernel_size*2+1):
            state = np.pad(state, ((h_min_pad, h_max_pad), (w_min_pad, w_max_pad)), constant_values=1)
        dist = self._get_distance_to_goal(goal_position, agent_pos)
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
            state, dist = self._get_partial_view(maze, self._objects.goal.positions[0], self._objects.agent.positions[0],\
                                                 self._width, self._height, self._view_kernel_size)
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

    def _is_valid(self, maze, position, flag=False):
        nonnegative = position[0] >= 0 and position[1] >= 0
        within_edge = position[0] < maze.shape[0] and position[1] < maze.shape[1]
        if flag == False:
            passable = not self._to_impassable()[position[0]][position[1]]
        else:
            passable = maze[position[0],position[1]] != 1
        return nonnegative and within_edge and passable

    def _is_goal(self, position):
        out = False
        for pos in self._objects.goal.positions:
            if position[0] == pos[0] and position[1] == pos[1]:
                out = True
                break
        return out

    def _get_reward(self, goal_position, first, second):
        return 1 - self._get_distance_to_goal(goal_position, second) / self._get_distance_to_goal(goal_position, first)

    def step(self, action):
        r"""Implements environment's `step(action)` function."""
        motion = self.motions[action]
        current_position = self._objects.agent.positions[0]
        new_position = [current_position[0] + motion[0], current_position[1] + motion[1]]
        valid = self._is_valid(self.maze, new_position)

        if valid:
            self._objects.agent.positions = [new_position]
        
        if self._is_goal(new_position):
            # reward = +10
            reward = 10
            terminated = True
        elif not valid:
            # reward = -10
            reward = -10
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
            # reward = 1 - self._get_distance_to_goal(new_position) / self._get_distance_to_goal(self._start_idx[0])
            reward = self._get_reward(self._objects.goal.positions[0], self._start_idx[0], new_position)
            terminated = False

        maze = self._to_value()

        if self._partial_view:
            state, dist = self._get_partial_view(maze, self._objects.goal.positions[0], self._objects.agent.positions[0], self._width, self._height, self._view_kernel_size)
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

    def _sample_random_position(self):
        while True:
            position = (np.random.randint(0, self.maze.shape[0]), np.random.randint(0, self.maze.shape[1]))
            if self._is_valid(self.maze, position):
                break
        return position

    def get_random_observation(self):
        r"""Generate a random observation from a non-obstructed location.

        This is used to generate observations for the replay buffer, to address the problem of termination bias: 
        The agent cannot explore the entire space, because it frequently terminates when it hits an obstacle.
        This is a severe problem that occurs during the initial stages, and whether it persists afterwards is yet to be determined.

        Note that the agent is first moved to the random location, and then moved back to its previous position.
        Consider this if there's a need to use this function for other purposes or for debugging.
            
        Returns:
        - :attr:`state` - the observation.
        - :attr:`action_logits` - random logits for the action space, sampled uniformly from [0,1] range.
        - :attr:`reward` - the reward received for the action.
        - :attr:`next_state` - the next observation.
        - :attr:`terminated` - whether the episode is terminated.
        """
        prev_agent_position = self._objects.agent.positions[0]
        new_position = self._sample_random_position()
        self._objects.agent.positions = [new_position]
        state, _ = self._get_partial_view(self._to_value(), self._objects.goal.positions[0], self._objects.agent.positions[0],\
                                          self._width, self._height, self._view_kernel_size)

        action_logits = np.random.rand(1,4)
        action = np.argmax(action_logits)
        next_state, reward, terminated, _, _ = self.step(action)
        self._objects.agent.positions = [prev_agent_position]
        return state, action_logits, reward, next_state, terminated

    def get_grid_view_collection(self, view_kernel_size=2, width=101, height=101, complexity=0.9, density=0.9, target=1000000, maze_count=100):
        r"""Generate a collection of grid views and regression targets for each action, for each position.
        
        Args:
        - :attr:`view_kernel_size` - the size of the grid view.
        - :attr:`width` - the width of the maze.
        - :attr:`height` - the height of the maze.
        - :attr:`complexity` - a parameter that controls the complexity of the maze.
        - :attr:`density` - a parameter that controls the density of the maze.
        - :attr:`target` - the number of grid views to generate, exluding goal placements. If this number is reached, or a maximum of `maze_count` mazes are created the function exits.
        - :attr:`maze_count` - the maximum number of mazes to generate. See `target` for more details.

        Returns:
        - :attr:`views` - a dictionary of grid views and regression targets for each position. Includes a separate view for each possible goal placements.
        - :attr:`mazes` - a list of mazes used to generate the views. Dumpped to save the overhead spent on generating the mazes.
        """
        views, mazes, count = {}, [], 0
        for i in range(maze_count):
            print(f'Maze {i+1:7d}/{maze_count:7d}, views: {count:7d}/{target:7d}')
            if count >= target:
                break

            maze = self._random_maze(width, height, complexity, density)
            mazes.append(maze)
            goal_position = (width-1,height-1)
            for i,j in np.ndindex(maze.shape):
                if (i,j) == goal_position:
                    continue
                
                if self._is_valid(maze, (i,j), True):
                    # Compute the grid view and legal actions for this position
                    state, _ = self._get_partial_view(maze, (goal_position), (i,j), width, height, view_kernel_size)
                    state[state == 2] = 0
                    state[view_kernel_size, view_kernel_size] = 2 # Place the agent in the center of the view
                    rewards = [1 if self._is_valid(maze, (i+self.motions[action][0], j+self.motions[action][1]), True) else 0 for action in range(len(self.motions))]
                    # rewards = F.softmax(torch.FloatTensor(rewards).view(1,-1), dim=-1).numpy().reshape(-1)
                    rewards = np.asarray(rewards) / np.sum(rewards)

                    key = state.tobytes()
                    if key in views:
                        continue

                    views[key]=[state, rewards]
                    count += 1

                    # Prevent bias towards fake goal views by generating exactly one view
                    while True:
                        action = np.random.randint(low=0, high=len(self.motions))
                        new_position = (i+self.motions[action][0], j+self.motions[action][1])
                        if self._is_valid(maze, new_position, True): #and (view_kernel_size <= i <= height - view_kernel_size) and (view_kernel_size <= j <= width - view_kernel_size):
                            new_state = state.copy()
                            new_state[new_state == 3] = 0
                            new_state[new_state == 2] = 0
                            new_state[view_kernel_size + self.motions[action][0], view_kernel_size + self.motions[action][1]] = 3
                            new_state[view_kernel_size, view_kernel_size] = 2

                            '''
                            # Fill it with obstacles since you are at the edge
                            # Move the goal to the new position, remember that agent is at the center of the view
                            new_state[new_state == 3] = 0
                            new_state[view_kernel_size + self.motions[action][0]+1:,:] = 1
                            new_state[:,view_kernel_size + self.motions[action][1]+1:] = 1
                            new_state[view_kernel_size + self.motions[action][0], view_kernel_size + self.motions[action][1]] = 3
                            '''
                            
                            new_rewards = [1 if i == action else 0 for i in range(len(self.motions))]
                            new_rewards = np.asarray(new_rewards) / np.sum(new_rewards)

                            new_key = new_state.tobytes()
                            if new_key not in views:
                                views[new_key]=[new_state, new_rewards]
                                break

        return views, mazes            

if __name__ == "__main__":
    """
    from gymnasium.envs.registration import register
    register(
        id="RandomMaze-v1.0",
        entry_point=RandomMaze,
    )

    # For exporting the environment to the gym registry
    '''
    setup(
        name="gym_examples",
        version="0.0.1",
        install_requires=["gymnasium==0.26.0", "pygame==2.1.0"],
    )

    # Use it with this afterwards: 
    import gym_examples
    env = gymnasium.make('gym_examples/GridWorld-v0')
    '''
    """
    # ========================================================================================
    '''
    import shutil
    try:
        shutil.rmtree(Path('./mazes'))
    except FileNotFoundError:
        pass
    
    Path('./mazes').mkdir(exist_ok=True)
    env = gym.make("RandomMaze-v1.0", render_mode="human", partial_view=True, view_kernel_size=2, width=51, height=51, complexity=0.9, density=0.9)
    env.get_wrapper_attr('generate_maze')('./mazes')
    env.get_wrapper_attr('generate_maze')('./mazes')
    env.get_wrapper_attr('generate_maze')('./mazes')
    env.close()
    env = gym.make("RandomMaze-v1.0", render_mode="human", partial_view=False, view_kernel_size=2, width=25, height=25, complexity=0.9, density=0.9)
    env.get_wrapper_attr('generate_maze')('./mazes')
    env.get_wrapper_attr('generate_maze')('./mazes')
    env.get_wrapper_attr('generate_maze')('./mazes')
    env.close()
    env = gym.make("RandomMaze-v1.0", render_mode="human", partial_view=False, view_kernel_size=2, width=7, height=7, complexity=0.9, density=0.9)
    env.get_wrapper_attr('generate_maze')('./mazes')
    env.get_wrapper_attr('generate_maze')('./mazes')
    env.get_wrapper_attr('generate_maze')('./mazes')
    '''
    # ========================================================================================
    '''
    env = RandomMaze(render_mode="human", partial_view=False, view_kernel_size=1)
    for path in Path('./mazes').glob('*.pkl'):
        path = str(path.resolve())
        env.generate_maze(path)

        state, _ = env.reset()
        for _ in range(10):
            env.render()
            action = env.action_space.sample()
            print("Action taken:", action)
            next_state, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                print("Environment is reset")
                next_state, info = env.reset()
            state = next_state
        # input("Press Enter to continue...")
    env.close()
    '''
    # ========================================================================================
    '''
    import pickle
    env = RandomMaze()
    view_kernel_size=2
    width=101
    height=101
    complexity=0.9
    density=0.9
    target=1000000
    maze_count=1000

    Path('./grid_views/').mkdir(parents=True, exist_ok=True)
    filepath = f'./grid_views/{view_kernel_size}_{width}_{height}_{complexity}_{density}_{target}_{maze_count}.pkl'
    
    if Path(filepath).exists() == False:
        views, mazes = env.get_grid_view_collection(view_kernel_size, width, height, complexity, density, target, maze_count)
        with open(f'./grid_views/{view_kernel_size}_{width}_{height}_{complexity}_{density}_{target}_{maze_count}.pkl', 'wb') as f:
            pickle.dump({
                'views': views,
                'mazes': mazes
            }, f)
    '''
    # ========================================================================================