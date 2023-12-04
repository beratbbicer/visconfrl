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

class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CriticNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.layers1 = nn.Sequential(
            nn.Linear(state_dim[0]*state_dim[1] + action_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 1),
        )

        self.layers2 = nn.Sequential(
            nn.Linear(state_dim[0]*state_dim[1] + action_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 1),
        )

    def forward(self, state, action):
        # Concatenate state and action
        x_q1 = torch.hstack((state.reshape(-1,self.state_dim[0]*self.state_dim[1]), action.reshape(-1,self.action_dim)))
        x_q2 = torch.clone(x_q1)
        
        q1 = self.layers1(x_q1)
        q2 = self.layers2(x_q2)
        return q1, q2
  
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(PolicyNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.layers = nn.Sequential(
            nn.Linear(state_dim[0]*state_dim[1],hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,action_dim)
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
    probs = F.softmax(raw_action, dim=1)
    (prob, action) = torch.max(probs, dim=1)
    # scaled_action = action_low + ((scaled_action - (-1)) * (action_high - action_low)) / (1 - (-1))
    return prob, action, probs

def update_critic(critic, target_critic, batch, gamma, critic_optimizer, target_actor, action_low, action_high):
    critic_optimizer.zero_grad()
    state_batch, action_batch, reward_batch, next_state_batch, done_batch = batch

    if args.double:
        state_batch = state_batch.double()()
        next_state_batch = next_state_batch.double()
    else:
        state_batch = state_batch.float()
        next_state_batch = next_state_batch.float()

    # Take the min q value between the two critics
    with torch.no_grad():
        _, _, next_action_probs = get_action(target_actor, next_state_batch, action_low, action_high)
        target_q1, target_q2 = target_critic(next_state_batch, next_action_probs)
        target_q = torch.min(target_q1, target_q2)
        target_q = reward_batch + gamma * (1 - done_batch) * target_q

    # Use the target q value to update both critics
    q1, q2 = critic(state_batch, action_batch)
    critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=2)
    critic_optimizer.step()

def update_actor(actor, critic, actor_optimizer, replay_buffer, batch_size, double, device, action_low, action_high):
    actor_optimizer.zero_grad()
    state = replay_buffer.sample(batch_size, double, device)[0]

    if args.double:
        state = state.double()
    else:
        state = state.float()

    # Take a deterministic policy step byt setting temperature to near-zero
    _, _, probs = get_action(actor, state, action_low, action_high)
    
    # Update the actor based on critics Q1 prediction
    actor_loss = -critic(state, probs)[0].mean()    
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=2)
    actor_optimizer.step()

def soft_update_target_network(model, target_model, tau):
    for param, target_param in zip(model.parameters(), target_model.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

def initiate_experiment(args):
    if args.render:
        env = RandomMaze(window_size=args.window_size, partial_view=not args.full_maze,\
                       view_kernel_size=args.maze_view_kernel_size, render_mode="human")
    else:
        env = RandomMaze(window_size=args.window_size, partial_view=not args.full_maze,\
                       view_kernel_size=args.maze_view_kernel_size) 
    state, _ = env.reset(seed=args.fixed_seed) # Set seed
    env.generate_maze(args.mazepath) # Generate maze & objects
    args.state_dim = [state.shape[0],state.shape[1]]

    actor = PolicyNetwork(args.state_dim, args.action_dim, args.hidden_dim)
    critic = CriticNetwork(args.state_dim, args.action_dim)
    target_actor = PolicyNetwork(args.state_dim, args.action_dim, args.hidden_dim)
    target_critic = CriticNetwork(args.state_dim, args.action_dim)

    if args.double:
        actor = actor.double()
        critic = critic.double()
        target_actor = target_actor.double()
        target_critic = target_critic.double()

    actor = actor.to(args.device)
    critic = critic.to(args.device)
    target_actor = target_actor.to(args.device)
    target_critic = target_critic.to(args.device)

    target_actor.load_state_dict(actor.state_dict())
    target_critic.load_state_dict(critic.state_dict())
    actor.train()
    critic.train()
    target_actor.eval()
    target_critic.eval()

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.actor_lr, weight_decay=args.weight_decay)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.critic_lr, weight_decay=args.weight_decay)
    
    replay_buffer = ReplayBuffer(args.buffer_size, args.full_maze)
    state = env.reset()[0] # Get initial state
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
                        # action = torch.DoubleTensor(np.random.rand(low=args.action_low, high=args.action_high+1, size = (1,1))).to(args.device)
                        action_probs = F.softmax(torch.rand(size=(1,args.action_dim)), dim=-1).double().to(args.device)
                    else:
                        # action = torch.FloatTensor(np.random.randint(low=args.action_low, high=args.action_high+1, size = (1,1))).to(args.device)
                        action_probs = F.softmax(torch.rand(size=(1,args.action_dim)), dim=-1).float().to(args.device)

                    action = torch.argmax(action_probs, dim=-1).item()
                else:
                    _, action, action_probs = get_action(actor, state_tensor, args.action_low, args.action_high)

                next_state, reward, terminated, _, _  = env.step(action)
                replay_buffer.add(state, action_probs, reward, next_state, terminated)
            
            if replay_buffer.__len__() <= args.batch_size:
                continue            

            # Policy Smoothing - add policy noise to the action
            batch = replay_buffer.sample(args.batch_size, args.double, args.device)
            update_critic(critic, target_critic, batch, args.gamma, critic_optimizer,\
                          target_actor, args.action_low, args.action_high)
            update_actor(actor, critic, actor_optimizer, replay_buffer, args.batch_size,\
                             args.double, args.device, args.action_low, args.action_high)
            soft_update_target_network(actor, target_actor, args.tau)
            soft_update_target_network(critic, target_critic, args.tau)
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
                        
                        _, action, _ = get_action(actor, state_tensor, args.action_low, args.action_high)
                        eval_state, reward, terminated, _, _  = env.step(action)
                        total_reward += reward
                        eval_steps += 1
                        
                reward_cache.append(total_reward)
        
            avg_reward = sum(reward_cache) / float(len(reward_cache))
            eval_avg_rewards_cache.append([int(episode / args.eval_steps),avg_reward])
            print(f"Episode: {episode:4d} -> Latest Eval Reward:{avg_reward: >7.3f}")
    
            if avg_reward > 10:
                print(f"Task solved in {episode:6d} episodes.")
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
            'actor': actor.state_dict(),
            'critic': critic.state_dict(),
            'target_actor': target_actor.state_dict(),
            'target_critic': target_critic.state_dict(),
            'actor_optimizer': actor_optimizer.state_dict(),
            'critic_optimizer': critic_optimizer.state_dict(),
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
    parser.add_argument('--actor_lr', type=float, default=1e-3)
    parser.add_argument('--critic_lr', type=float, default=1e-3)
    parser.add_argument('--hidden_dim', type=int, default=256)

    # DoubleQL Arguments
    parser.add_argument('--epsilon', type=float, default=0.5)
    parser.add_argument('--epsilon_decay', type=float, default=0.5)
    parser.add_argument('--epsilon_min', type=float, default=0.1)
    parser.add_argument('--epsilon_step', type=float, default=500)

    # Env Arguments
    parser.add_argument('--window_size', default=256, type=int)
    parser.add_argument('--full_maze', action='store_true')
    parser.add_argument('--maze_view_kernel_size', default=1, type=int)
    parser.add_argument('--mazepath', type=str, default='mazes/width-25_height-25_complexity-0.9_density-0.9_MtU6Wf2sgAp2Aq9m.pkl')

    # Misc Arguments
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--double', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--dump', action='store_true')
    parser.add_argument('--rootname',\
        default="".join(random.SystemRandom().choice(string.ascii_uppercase+string.ascii_lowercase+string.digits) for _ in range(8)),\
        type=str, help="Name for experiment root folder. Defaults to length-8 random string.")
    args = parser.parse_args()
    
    args.action_low, args.action_high, args.action_dim = 0, 3, 4
    args.device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    args.fixed_seed = fixed_seed

    settings = f"Configuration {args.rootname} ->\n\
        gamma:{args.gamma}, tau:{args.tau}\n\
        buffer_size:{args.buffer_size}, batch_size:{args.batch_size}\n\
        max_steps:{args.max_steps}, eval_steps:{args.eval_steps}, device:{args.device}, double:{args.double}, actor_lr:{args.actor_lr}, critic_lr:{args.critic_lr}\n\
        episode_avg:{args.episode_avg}, render:{args.render}, dump:{args.dump}, eval_start:{args.eval_start}, epsilon_step:{args.epsilon_step}\n\
        eval_episode_max_step:{args.eval_episode_max_step}, epsilon_decay:{args.epsilon_decay}, epsilon_min:{args.epsilon_min}, epsilon: {args.epsilon}\n\n\
        window_size:{args.window_size}, full_maze:{args.full_maze}, maze_view_kernel_size:{args.maze_view_kernel_size}, weight_decay:{args.weight_decay}\n\
        mazepath:{args.mazepath}\n"
    print(settings)

    args.device = torch.device(args.device)

    writepath = f'./experiments/doubleql/{args.mazepath.split("/")[-1].split(".pkl")[0].split("_")[-1]}/{args.rootname}'
    Path(writepath).mkdir(parents=True, exist_ok=True)
    args.writepath = writepath

    initiate_experiment(args)

    """ 
    Optional partial maze view setting.
    Input is a grid of cells, either the full maze (height x width) or a partial view (view_kernel_size*2+1 x view_kernel_size*2+1)
    Output is still a 1-size action idx over 4-discrete values (0,1,2,3) -> (N,S,E,W). Look up maze definition for more.
    
    Render occurs post pure-exploration only
    """