# https://github.com/openai/gym/blob/master/gym/envs/box2d/lunar_lander.py
# original implementation: https://github.com/sfujim/TD3

import random
fixed_seed = random.randint(0, 2**32 - 1)
random.seed(fixed_seed)
import numpy as np
np.random.seed(fixed_seed)
import torch
torch.manual_seed(fixed_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(fixed_seed)

import torch.nn as nn
import torch.nn.functional as F

import gymnasium as gym
import argparse, string, random
from pathlib import Path
import math
import matplotlib.pyplot as plt
import pickle

from random_maze1_3 import RandomMaze

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.layers = nn.Sequential(
            nn.Linear(state_dim[0]*state_dim[1],400),
            nn.ReLU(),
            nn.Linear(400,300),
            nn.ReLU(),
            nn.Linear(300,300),
            nn.ReLU(),
            nn.Linear(300,300),
            nn.ReLU(),
            nn.Linear(300,action_dim),
        )

    def forward(self, state):
        return self.layers(state.reshape(-1,self.state_dim[0]*self.state_dim[1]))

class ReplayBuffer:
    def __init__(self, max_size, full_maze):
        if max_size <= 0:
            self.max_size = math.inf
        else:
            self.max_size = max_size

        self.full_maze = full_maze
        self.buffer = []

    def add(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        if len(self.buffer) < self.max_size:
            self.buffer.append(experience)
        else:
            self.buffer.pop(0)
            self.buffer.append(experience)

    def sample(self, batch_size, double, device):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in indices])

        states = torch.from_numpy(np.stack(states,axis=0))
        actions = torch.vstack(actions)
        rewards = torch.FloatTensor(rewards).view(-1,1)
        next_states = torch.from_numpy(np.stack(next_states,axis=0))
        dones = torch.Tensor(dones).view(-1,1)
        
        # Convert to PyTorch tensors
        if double:
            states = states.double()
            actions = actions.long()
            rewards = rewards.double()
            next_states = next_states.double()
            dones = torch.Tensor(dones).view(-1,1)

        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        next_states = next_states.to(device)
        dones = dones.to(device)
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

def get_action(actor, state_tensor, action_low, action_high):
    raw_action = actor(state_tensor)
    (q_val, action) = torch.max(raw_action, dim=1)
    # scaled_action = action_low + ((scaled_action - (-1)) * (action_high - action_low)) / (1 - (-1))
    return q_val, action, raw_action

def update_critic(q_network, target_q_network, batch, gamma, q_network_optimizer, action_low, action_high):
    q_network_optimizer.zero_grad()
    state_batch, action_batch, reward_batch, next_state_batch, done_batch = batch

    if args.double:
        state_batch = state_batch.double()
        next_state_batch = next_state_batch.double()
    else:
        state_batch = state_batch.float()
        next_state_batch = next_state_batch.float()

    # Calculate Target Q Values
    with torch.no_grad():
        _, _, nextq = get_action(target_q_network, next_state_batch, action_low, action_high)
        targetq = reward_batch + gamma * (1 - done_batch) * nextq

    # Update the network with target Q values
    _, _, q = get_action(q_network, state_batch, action_low, action_high)
    critic_loss = F.smooth_l1_loss(q, targetq)
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(q_network.parameters(), max_norm=2)
    q_network_optimizer.step()

def soft_update_target_network(model, target_model, tau):
    for param, target_param in zip(model.parameters(), target_model.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

def initiate_experiment(args):
    from gymnasium.envs.registration import register
    env_name = "RandomMaze-v1.0"
    register(
        id=env_name,
        entry_point=RandomMaze,
    )

    if args.render:
        env = gym.make(env_name, window_size=args.window_size, width=args.width, height=args.height,\
                       complexity=args.complexity, density=args.density, partial_view=not args.full_maze,\
                       view_kernel_size=args.maze_view_kernel_size, render_mode="human")
    else:
        env = gym.make(env_name, window_size=args.window_size, width=args.width, height=args.height,\
                       complexity=args.complexity, density=args.density, partial_view=not args.full_maze,\
                       view_kernel_size=args.maze_view_kernel_size)
    # env.seed(args.fixed_seed)

    q_network = QNetwork(args.state_dim, args.action_dim)
    target_q_network = QNetwork(args.state_dim, args.action_dim)

    if args.double:
        q_network = q_network.double()
        target_q_network = target_q_network.double()

    q_network = q_network.to(args.device)
    target_q_network = target_q_network.to(args.device)
    target_q_network.load_state_dict(q_network.state_dict())
    q_network.train()
    target_q_network.eval()
    q_network_optimizer = torch.optim.Adam(q_network.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    replay_buffer = ReplayBuffer(args.buffer_size, args.full_maze)
    env.reset(seed=args.fixed_seed) # Set seed
    env.get_wrapper_attr('generate_maze')() # Generate maze & objects
    eval_avg_rewards_cache = []
 
    # Main training loop
    for episode in range(args.max_steps):
        # Epsilon Decay
        if episode > 0 and episode % args.epsilon_step == 0:
            args.epsilon = max(args.epsilon_min, args.epsilon * args.epsilon_decay)

        state = env.reset()[0] # Get initial state
        terminated = False

        while terminated == False:
            state_tensor = torch.from_numpy(state.astype(np.float32)).reshape(1,-1)
            if args.double:
                state_tensor = state_tensor.double()
            state_tensor = state_tensor.to(args.device)

            if args.render:
                env.render()

            rand = np.random.rand()
            with torch.no_grad():
                if rand < args.epsilon:
                    if args.double:
                        action = torch.DoubleTensor(np.random.randint(low=args.action_low, high=args.action_high+1, size = (1,1))).to(args.device)
                    else:
                        action = torch.FloatTensor(np.random.randint(low=args.action_low, high=args.action_high+1, size = (1,1))).to(args.device)
                else:
                    _, action, _ = get_action(q_network, state_tensor, args.action_low, args.action_high)

                next_state, reward, terminated, _, _  = env.step(int(action.item()))
                replay_buffer.add(state, action, reward, next_state, terminated)
            
            if replay_buffer.__len__() <= args.batch_size:
                continue            

            # Policy Smoothing - add policy noise to the action
            batch = replay_buffer.sample(args.batch_size, args.double, args.device)
            update_critic(q_network, target_q_network, batch, args.gamma, q_network_optimizer, args.action_low, args.action_high)
            soft_update_target_network(q_network, target_q_network, args.tau)
            state = next_state

        # ==================================================================================
        # Validation
        if episode > args.eval_start and episode % args.eval_steps == 0:
            reward_cache = []

            while len(reward_cache) < args.episode_avg:
                eval_state = env.reset()[0]
                terminated, total_reward, eval_steps = False, 0, 0
                                    
                while terminated == False and eval_steps < args.eval_episode_max_step:
                    with torch.no_grad():
                        eval_state_tensor = torch.from_numpy(eval_state.astype(np.float32)).reshape(1,-1)
                        if args.double:
                            eval_state_tensor = eval_state_tensor.double()
                        eval_state_tensor = eval_state_tensor.to(args.device)
                        
                        _, action, _ = get_action(q_network, eval_state_tensor, args.action_low, args.action_high)
                        eval_state, reward, terminated, _, _  = env.step(int(action.item()))
                        total_reward += reward
                        eval_steps += 1
                        
                reward_cache.append(total_reward)
        
            avg_reward = sum(reward_cache) / float(len(reward_cache))
            eval_avg_rewards_cache.append([int(episode / args.eval_steps),avg_reward])
            print(f"Episode: {episode:4d} -> Latest Eval Reward:{avg_reward: >7.3f}")
    
            if avg_reward > 10:
                print(f"Task solved in {episode:4d} episodes.")
                break

    env.close()

    # Save plots
    steps, rewards = zip(*eval_avg_rewards_cache)
    fig = plt.figure(figsize=(20, 8))
    x = np.arange(len(steps))
    plt.xticks(ticks=x, labels=steps, rotation=90)
    plt.plot(x, rewards, label='Rewards', color='red', linewidth=2, linestyle=':', marker='o')
    plt.xlabel('Steps (x5000)')
    plt.ylabel('Reward')
    plt.title('Avg. Reward Throughout Training')
    plt.legend()
    plt.tick_params(axis='x', which='major', labelsize=5)
    plt.tick_params(axis='y', which='major', labelsize=10)
    plt.tight_layout()
    # plt.show()

    with open(f'{args.writepath}.pkl', 'wb') as f:
        pickle.dump({
            "fig": fig,
            "steps": steps,
            "rewards": rewards,
        }, f)

    # Dump The Modelbuffer_size
    if args.dump:
        torch.save({
            'q_network': q_network.state_dict(),
            'target_q_network': target_q_network.state_dict(),
            'q_network_optimizer': q_network_optimizer.state_dict(),
            'replay_buffer': replay_buffer,
            'fixed_seed ': fixed_seed, 
            'args': args,
        }, f'{args.writepath}.pth')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # TD3 Arguments
    parser.add_argument('--gamma', type=float, default=0.99) # Future reward scaling (discount)
    parser.add_argument('--tau', type=float, default=1.0) # Target update rate
    parser.add_argument('--buffer_size', type=int, default=-1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_steps', type=int, default=1000000)
    parser.add_argument('--eval_steps', type=int, default=10000)
    parser.add_argument('--eval_start', default=10000, type=int)
    parser.add_argument('--episode_avg', default=10, type=int)
    parser.add_argument('--eval_episode_max_step', default=10000, type=int)
    parser.add_argument('--lr', type=float, default=2e-3)

    # DQL Arguments
    parser.add_argument('--epsilon', type=float, default=0.5)
    parser.add_argument('--epsilon_decay', type=float, default=0.5)
    parser.add_argument('--epsilon_min', type=float, default=0.1)
    parser.add_argument('--epsilon_step', type=float, default=500)

    # Env Arguments
    parser.add_argument('--window_size', default=256, type=int)
    parser.add_argument('--width', default=7, type=int)
    parser.add_argument('--height', default=7, type=int)
    parser.add_argument('--complexity', default=0.75, type=float)
    parser.add_argument('--density', default=0.75, type=float)
    parser.add_argument('--full_maze', action='store_true')
    parser.add_argument('--maze_view_kernel_size', default=1, type=int)

    # Misc Arguments
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--double', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--dump', action='store_true')
    parser.add_argument('--rootname',\
        default=f'maze_solver_dql_episodic_{"".join(random.SystemRandom().choice(string.ascii_uppercase+string.ascii_lowercase+string.digits) for _ in range(5))}',\
        type=str, help="Name for experiment root folder. Defaults to length-5 random string.")
    args = parser.parse_args()
    
    args.action_low, args.action_high, args.action_dim = 0, 3, 4
    if args.full_maze:
        args.state_dim = [args.height,args.width]
    else:
        args.state_dim = [args.maze_view_kernel_size * 2 + 1, args.maze_view_kernel_size * 2 + 1]

    args.device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    args.fixed_seed = fixed_seed

    settings = f"Configuration {args.rootname} ->\n\
        gamma:{args.gamma}, tau:{args.tau}\n\
        buffer_size:{args.buffer_size}, batch_size:{args.batch_size}\n\
        max_steps:{args.max_steps}, eval_steps:{args.eval_steps}, device:{args.device}, double:{args.double}, lr:{args.lr}\n\
        episode_avg:{args.episode_avg}, render:{args.render}, dump:{args.dump}, eval_start:{args.eval_start}, epsilon_step:{args.epsilon_step}\n\
        eval_episode_max_step:{args.eval_episode_max_step}, epsilon_decay:{args.epsilon_decay}, epsilon_min:{args.epsilon_min}, epsilon: {args.epsilon}\n\n\
        window_size:{args.window_size}, width:{args.width}, height:{args.height}, complexity:{args.complexity}, density:{args.density}\n\
        full_maze:{args.full_maze}, maze_view_kernel_size:{args.maze_view_kernel_size}, weight_decay:{args.weight_decay}\n"
    print(settings)

    args.device = torch.device(args.device)

    Path('experiments').mkdir(parents=True, exist_ok=True)
    args.writepath = f'experiments/{args.rootname}'

    initiate_experiment(args)

    """ 
    Optional partial maze view setting.
    Input is a grid of cells, either the full maze (height x width) or a partial view (view_kernel_size*2+1 x view_kernel_size*2+1)
    Output is still a 1-size action idx over 4-discrete values (0,1,2,3) -> (N,S,E,W). Look up maze definition for more.
    
    Render occurs post pure-exploration only
    """