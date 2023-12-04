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
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from random_maze1_3 import RandomMaze
import argparse, os
from pathlib import Path
import string, random

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
    
    def _init_input_spaces(self):
        self._view_kernel_size = min(self._view_kernel_size, self._width//2, self._height//2)
        self._start_idx, self._goal_idx = [[1,1]], [[self._height-2, self._width-2]]
        if self._partial_view:
            self.observation_space = gym.spaces.Box(low=np.zeros(((self._view_kernel_size*2+1)**2)),\
                                                high=3+np.zeros(((self._view_kernel_size*2+1)**2)),\
                                                shape=((self._view_kernel_size*2+1)**2,), dtype=int)
        else:
            self.observation_space = gym.spaces.Box(low=np.zeros(((self._height*2+1)*(self._width*2+1))),\
                                                high=3+np.zeros(((self._height*2+1)*(self._width*2+1))),\
                                                shape=(((self._height*2+1)*(self._width*2+1),)), dtype=int)
            
        self.action_space = gym.spaces.Discrete(len(self.motions))

    def reset(self, *args, **kwargs):
        state, info = super().reset(*args, **kwargs)
        return state.reshape(-1), info
    
    def step(self, action, *args, **kwargs):
        state, reward, terminated, truncated, info = super().step(action, *args, **kwargs)
        return state.reshape(-1), reward, terminated, truncated, info

def execute(args, mazepath):
    env = RandomMaze_StableBaselines(window_size=args.window_size, partial_view=not args.full_maze,\
                                     view_kernel_size=args.maze_view_kernel_size, render_mode="human" if args.render else None)
    eval_env = RandomMaze_StableBaselines(window_size=args.window_size, partial_view=not args.full_maze,\
                                     view_kernel_size=args.maze_view_kernel_size, render_mode="human" if args.render else None)
    
    env.generate_maze(mazepath)
    eval_env.generate_maze(mazepath)
    env.reset(seed=fixed_seed)
    eval_env.reset(seed=fixed_seed)
    eval_env = Monitor(eval_env)

    name = "".join(random.SystemRandom().choice(string.ascii_uppercase+string.ascii_lowercase+string.digits) for _ in range(8))
    modelpath = f'./experiments/sb3_dqn/{mazepath.split(os.sep)[-1].split(".pkl")[0].split("_")[-1]}/{name}'
    Path(modelpath).mkdir(parents=True, exist_ok=True)

    stop_train_callback = StopTrainingOnRewardThreshold(reward_threshold=10, verbose=1)
    eval_callback = EvalCallback(eval_env, callback_after_eval=stop_train_callback, best_model_save_path=modelpath,\
                                 log_path=modelpath, eval_freq=1000, n_eval_episodes=10, deterministic=True, render=False, verbose=0)
    model = DQN(DQNPolicy, env, verbose=0, batch_size=128, learning_rate=1e-3, buffer_size=int(1e7), tau=0.005,\
                gamma=0.99, train_freq=1, gradient_steps=1, learning_starts=int(1e5))
    model.learn(total_timesteps=1e7, callback=eval_callback)

    # Evaluate the agent manually
    '''
    eval_env = Monitor(env)
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
    with open(f'{model_name}.txt', 'w') as f:
        f.write(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
    '''
    
    env.close()

# This fails due to CUDA initilaization error: https://stackoverflow.com/a/40083505
def run_parallel_multiprocess(args):
    from multiprocessing import Process
    children = []
    for path in Path('./mazes').glob('*.pkl'):
        path = str(path.resolve())
        child = Process(target=execute, args=(args, path))
        child.start()
        children.append(child)

    for child in children:
        child.join()

def run_parallel_joblib(args):
    from joblib import Parallel, delayed
    paths = [str(p.resolve()) for p in Path('./mazes').glob('*.pkl')]
    Parallel(n_jobs=len(paths))(delayed(execute)(args, path) for path in paths)

def run_single(args):
    for path in Path('./mazes').glob('*.pkl'):
        path = str(path.resolve())
        execute(args, path)
        _ = 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--window_size', default=256, type=int)
    parser.add_argument('--full_maze', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--maze_view_kernel_size', default=1, type=int)
    args = parser.parse_args()
    
    run_single(args)