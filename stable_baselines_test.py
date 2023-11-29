# huggingface lunar lander example
# https://huggingface.co/learn/deep-rl-course/unit1/hands-on

import random
fixed_seed = random.randint(0, 2**32 - 1)
random.seed(fixed_seed)
import numpy as np
np.random.seed(fixed_seed)
import torch
torch.manual_seed(fixed_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(fixed_seed)

import gymnasium as gym
from gymnasium.envs.registration import register
import stable_baselines3
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from random_maze1_3 import RandomMaze
import argparse

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class RandomMaze_StableBaselines(RandomMaze):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_space = gym.spaces.Box(low=np.zeros(((self._view_kernel_size*2+1)**2)),\
                                                high=3+np.zeros(((self._view_kernel_size*2+1)**2)),\
                                                shape=((self._view_kernel_size*2+1)**2,), dtype=int)
        _ = 1
        
    def reset(self, *args, **kwargs):
        state, info = super().reset(*args, **kwargs)
        return state.reshape(-1), info
    
    def step(self, action, *args, **kwargs):
        state, reward, terminated, truncated, info = super().step(action, *args, **kwargs)
        return state.reshape(-1), reward, terminated, truncated, info

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Env Arguments
    parser.add_argument('--window_size', default=256, type=int)
    parser.add_argument('--width', default=7, type=int)
    parser.add_argument('--height', default=7, type=int)
    parser.add_argument('--complexity', default=0.75, type=float)
    parser.add_argument('--density', default=0.75, type=float)
    parser.add_argument('--full_maze', action='store_true')
    parser.add_argument('--maze_view_kernel_size', default=1, type=int)
    args = parser.parse_args()
    
    env_name = "RandomMaze_StableBaselines-v1.0"
    register(
        id=env_name,
        entry_point=RandomMaze_StableBaselines,
    )

    env = gym.make(env_name, window_size=args.window_size, width=args.width, height=args.height,\
                   complexity=args.complexity, density=args.density, partial_view=not args.full_maze,\
                   view_kernel_size=args.maze_view_kernel_size)

    # Then we reset this environment
    env.reset(seed=fixed_seed) # Set seed
    env.get_wrapper_attr('generate_maze')() # Generate maze & objects
    #state = env.reset()[0] # Get initial state

    model = DQN(DQNPolicy, env, verbose=1, batch_size=100, learning_rate=1e-3, buffer_size=int(1e7), tau=0.005,\
                gamma=0.99, train_freq=1, gradient_steps=1, learning_starts=int(1e5), device='cuda:0')
    model.learn(total_timesteps=1e6)
    # Save the model
    model_name = "dqn-MazeSolver-v0"
    model.save(model_name)

    # Evaluate the agent
    eval_env = Monitor(env)
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
    env.close()
