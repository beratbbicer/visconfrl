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
from pretrain import load_model
from random_maze1_3 import RandomMaze

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class QNetwork(nn.Module):
    def __init__(self, action_dim, path, freeze=True):
        super(QNetwork, self).__init__()
        self.action_dim = action_dim

        # ====================================================================================
        # Pretrained FE Layers
        # ====================================================================================
        model, checkpoint = load_model(path)
        self.state_dim = model.state_dim
        self.hidden_dim = model.hidden_dim
        self.backbone = model
        self.layers = model.layers
    
        if freeze:
            for param in self.layers.parameters():
                param.requires_grad = False
        # ====================================================================================
        # Attach an LSTM in the middle
        # ====================================================================================
        self.rnn = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim, num_layers=1,\
                           batch_first=True, bidirectional=False)
        # ====================================================================================
        # Output Layer
        # ====================================================================================
        self.output_layer = model.output_layer

    def initialize_lstm_states(self, batch_size, device, tensor_type):
        num_directions = 2 if self.rnn.bidirectional else 1
        h0 = torch.randn(self.rnn.num_layers * num_directions, batch_size, self.rnn.hidden_size)
        c0 = torch.randn(self.rnn.num_layers * num_directions, batch_size, self.rnn.hidden_size)

        if tensor_type == torch.DoubleTensor:
            h0, c0 = h0.double().to(device), c0.double().to(device)
        else:
            h0, c0 = h0.float().to(device), c0.float().to(device)

        nn.init.constant_(h0, 0.01)
        nn.init.constant_(c0, 0.01)

        '''
        if self.h0 is None:
            self.h0, self.c0 = nn.Parameter(h0), nn.Parameter(c0)
        else:
            self.h0.data, self.c0.data = h0, c0
        '''
        return h0, c0

    def forward(self, state, h0, c0):
        if len(state.size()) == 2:
            b,h,w = state.unsqueeze(0).size()
        elif len(state.size()) == 3:
            b,h,w = state.size()
        else:
            b,_,h,w = state.size()
        # ================================================================
        # Pretrained FE Layers
        out = state.view(b, 1, h, w)
        for layer in self.layers:
            out = layer(out)
        out = F.adaptive_avg_pool2d(out, output_size=(1,1)).view(b, 1, -1)
        # ================================================================
        # LSTM
        # Input and output tensors have batch_size as the first dimension!
        '''
        if h0 is not None and c0 is not None:
            out, (h0_out, c0_out) = self.rnn(out, (h0, c0))
        else:
            out, (h0_out, c0_out) = self.rnn(out, (self.h0, self.c0))
            self.h0.data, self.c0.data = h0_out, c0_out
        '''
        out, (h0_out, c0_out) = self.rnn(out, (h0, c0))
        # ================================================================
        # Output Layer
        out = F.softmax(self.output_layer(out.view(b,-1)))
        return out, h0_out, c0_out

class ReplayBuffer:
    def __init__(self, max_size):
        if max_size <= 0:
            self.max_size = math.inf
        else:
            self.max_size = max_size

        self.buffer = []

    def add(self, state, action, reward, next_state, done, h0, c0, h0_next, c0_next):
        experience = (state, action.cpu().numpy(), reward, next_state, done,\
                      h0.cpu().numpy(), c0.cpu().numpy(), h0_next.cpu().numpy(), c0_next.cpu().numpy())
        if len(self.buffer) < self.max_size:
            self.buffer.append(experience)
        else:
            self.buffer.pop(0)
            self.buffer.append(experience)

    def sample(self, batch_size, double, device):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones, h0, c0, h0_next, c0_next = zip(*[self.buffer[i] for i in indices])

        states = torch.from_numpy(np.stack(states, axis=0))
        actions = torch.from_numpy(np.stack(actions, axis=0)).squeeze(1)
        rewards = torch.FloatTensor(rewards).view(-1, 1)
        next_states = torch.from_numpy(np.stack(next_states, axis=0))
        dones = torch.Tensor(dones).view(-1, 1)
        h0 = torch.from_numpy(np.stack(h0, axis=0)).squeeze(1).permute(1,0,2)
        c0 = torch.from_numpy(np.stack(c0, axis=0)).squeeze(1).permute(1,0,2)
        h0_next = torch.from_numpy(np.stack(h0_next, axis=0)).squeeze(1).permute(1,0,2)
        c0_next = torch.from_numpy(np.stack(c0_next, axis=0)).squeeze(1).permute(1,0,2)

        # Convert to PyTorch tensors
        if double:
            states = states.double()
            actions = actions.long()
            rewards = rewards.double()
            next_states = next_states.double()
            dones = torch.Tensor(dones).view(-1, 1)
            h0 = h0.double()
            c0 = c0.double()
            h0_next = h0_next.double()
            c0_next = c0_next.double()

        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        next_states = next_states.to(device)
        dones = dones.to(device)
        h0 = h0.to(device)
        c0 = c0.to(device)
        h0_next = h0_next.to(device)
        c0_next = c0_next.to(device)
        return states, actions, rewards, next_states, dones, h0, c0, h0_next, c0_next

    def __len__(self):
        return len(self.buffer)

def initiate_experiment(args):
    if args.render:
        env = RandomMaze(
            window_size=args.window_size,
            partial_view=not args.full_maze,
            view_kernel_size=args.maze_view_kernel_size,
            render_mode="human",
        )
    else:
        env = RandomMaze(
            window_size=args.window_size,
            partial_view=not args.full_maze,
            view_kernel_size=args.maze_view_kernel_size,
        )

    state, _ = env.reset(seed=args.fixed_seed)  # Set seed
    env.generate_maze(args.mazepath)  # Generate maze & objects
    args.state_dim = [state.shape[0], state.shape[1]]

    q_network = QNetwork(args.action_dim, args.weightspath, not args.unfreeze_policy)
    target_q_network = QNetwork(args.action_dim, args.weightspath, not args.unfreeze_policy)

    if args.double:
        q_network = q_network.double()
        target_q_network = target_q_network.double()

    q_network = q_network.to(args.device)
    target_q_network = target_q_network.to(args.device)
    target_q_network.load_state_dict(q_network.state_dict())
    q_network.train()
    target_q_network.eval()
    optimizer = torch.optim.Adam(q_network.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    replay_buffer = ReplayBuffer(args.buffer_size)
    state = env.reset()[0]  # Get initial state
    eval_avg_rewards_cache = []
    epsilon, epsilon_policy = args.epsilon, 0.1

    # Main training loop
    for episode in range(args.max_steps):
        # Epsilon Decay
        if episode > 0 and episode % args.epsilon_step == 0:
            epsilon = max(args.epsilon_min, epsilon * args.epsilon_decay)

        state = env.reset()[0] # Get initial state
        terminated = False

        if args.double:
            hs, cs = q_network.initialize_lstm_states(1, args.device, torch.DoubleTensor)
        else:
            hs, cs = q_network.initialize_lstm_states(1, args.device, torch.FloatTensor)

        while terminated == False:
            state_tensor = torch.from_numpy(state.astype(np.float32))
            if args.double:
                state_tensor = state_tensor.double()
            state_tensor = state_tensor.to(args.device)

            if args.render:
                env.render()

            with torch.no_grad():
                q_network.eval()
                q_value_logits, hs_out, cs_out = q_network(state_tensor, hs, cs)
                q_values = q_value_logits + torch.rand_like(q_value_logits) * epsilon
                action = torch.argmax(q_values)
                next_state, reward, terminated, _, _  = env.step(action)
                replay_buffer.add(state, q_values, reward, next_state, terminated, hs, cs, hs_out, cs_out)
                state, hs, cs = next_state, hs_out, cs_out
                q_network.train()
            
            if replay_buffer.__len__() <= args.batch_size:
                continue            

            batch = replay_buffer.sample(args.batch_size, args.double, args.device)
            optimizer.zero_grad()
            state_batch, _, reward_batch, next_state_batch, done_batch,\
                h0_batch, c0_batch, next_h0_batch, next_c0_batch = batch

            if args.double:
                state_batch = state_batch.double()
                next_state_batch = next_state_batch.double()
            else:
                state_batch = state_batch.float()
                next_state_batch = next_state_batch.float()

            with torch.no_grad():
                next_q_value_logits, next_hs_out, next_cs_out = target_q_network(next_state_batch, next_h0_batch, next_c0_batch)
                next_q_values = next_q_value_logits + torch.rand_like(next_q_value_logits) * epsilon_policy
                target_q_values = reward_batch + args.gamma * (1 - done_batch) * next_q_values

            q_values, _, _ = q_network(state_batch, h0_batch, c0_batch)
            loss = F.l1_loss(q_values, target_q_values)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q_network.parameters(), max_norm=2)
            optimizer.step()

            for param, target_param in zip(q_network.parameters(), target_q_network.parameters()):
                target_param.data.copy_(param.data)

        if args.render:
            env.render()

        # ==================================================================================
        # Validation
        if episode > args.eval_start and episode % args.eval_steps == 0:
            q_network.eval()
            reward_cache = []

            while len(reward_cache) < args.episode_avg:
                eval_state = env.reset()[0]
                if args.double:
                    hs_eval, cs_eval = q_network.initialize_lstm_states(1, args.device, torch.DoubleTensor)
                else:
                    hs_eval, cs_eval = q_network.initialize_lstm_states(1, args.device, torch.FloatTensor)

                terminated, total_reward, eval_steps = False, 0, 0
                                    
                while terminated == False and eval_steps < args.eval_episode_max_step:
                    with torch.no_grad():
                        eval_state_tensor = torch.from_numpy(eval_state.astype(np.float32))
                        if args.double:
                            eval_state_tensor = eval_state_tensor.double()
                        eval_state_tensor = eval_state_tensor.to(args.device)

                        q_values_eval, hs_eval, cs_eval = q_network(eval_state_tensor, hs_eval, cs_eval)
                        action = torch.argmax(q_values_eval)
                        eval_state, reward, terminated, _, _  = env.step(action)
                        total_reward += reward
                        eval_steps += 1
                        
                reward_cache.append(total_reward)

            q_network.train()
            avg_reward = sum(reward_cache) / float(len(reward_cache))
            eval_avg_rewards_cache.append([int(episode / args.eval_steps),avg_reward])
            print(f"Episode: {episode:4d} -> Latest Eval Reward:{avg_reward: >7.3f}")

            if avg_reward >= 1:
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
            'optimizer': optimizer.state_dict(),
            'replay_buffer': replay_buffer,
            'fixed_seed ': fixed_seed, 
            'args': args,
        }, f'{args.writepath}.pth')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weightspath', default='./download/pretrain/2_101_101_0.9_0.9_13000_600_split/lNY48Pwj/e60.pth', type=str, help='Path to pretrained weights.')
    parser.add_argument('--unfreeze_critic', action='store_true')
    parser.add_argument('--unfreeze_policy', action='store_true')
    
    # TD3 Arguments
    parser.add_argument('--gamma', type=float, default=0.99) # Future reward scaling (discount)
    parser.add_argument('--tau', type=float, default=1.0) # Target update rate
    parser.add_argument('--buffer_size', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_steps', type=int, default=1000000)
    parser.add_argument('--eval_steps', type=int, default=5000)
    parser.add_argument('--eval_start', default=10000, type=int)
    parser.add_argument('--episode_avg', default=10, type=int)
    parser.add_argument('--eval_episode_max_step', default=10000, type=int)
    parser.add_argument('--lr', type=float, default=2e-3)

    # DQL Arguments
    parser.add_argument('--epsilon', type=float, default=0.5)
    parser.add_argument('--epsilon_decay', type=float, default=0.9)
    parser.add_argument('--epsilon_min', type=float, default=0.1)
    parser.add_argument('--epsilon_step', type=float, default=1000)

    # Env Arguments
    parser.add_argument('--window_size', default=256, type=int)
    parser.add_argument("--mazepath", type=str, default="./mazes/width-51_height-51_complexity-0.9_density-0.9_bbnYDPhsjerrENoS.pkl")

    parser.add_argument('--full_maze', action='store_true')
    parser.add_argument('--maze_view_kernel_size', default=1, type=int)

    # Misc Arguments
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--double', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--dump', action='store_true')
    parser.add_argument("--rootname", type=str, help="Name for experiment root folder. Defaults to length-8 random string.",\
                        default="".join(random.SystemRandom().choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(8)))
    args = parser.parse_args()
    
    args.device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    args.fixed_seed = fixed_seed
    args.maze_view_kernel_size = 2

    settings = f"Configuration {args.rootname} ->\n\
        gamma:{args.gamma}, tau:{args.tau}\n\
        buffer_size:{args.buffer_size}, batch_size:{args.batch_size}\n\
        max_steps:{args.max_steps}, eval_steps:{args.eval_steps}, device:{args.device}, double:{args.double}, lr:{args.lr}\n\
        episode_avg:{args.episode_avg}, render:{args.render}, dump:{args.dump}, eval_start:{args.eval_start}, epsilon_step:{args.epsilon_step}\n\
        eval_episode_max_step:{args.eval_episode_max_step}, epsilon_decay:{args.epsilon_decay}, epsilon_min:{args.epsilon_min}, epsilon: {args.epsilon}\n\n\
        window_size:{args.window_size}, full_maze:{args.full_maze}, maze_view_kernel_size:{args.maze_view_kernel_size}, weight_decay:{args.weight_decay}\n"
    print(settings)

    args.action_low, args.action_high, args.action_dim = 0, 3, 4
    args.device = torch.device(args.device)

    writepath = f'./experiments/dql_conv_pretrained_lstm/{args.mazepath.split("/")[-1].split(".pkl")[0].split("_")[-1]}/{args.rootname}'
    Path(writepath).mkdir(parents=True, exist_ok=True)
    args.writepath = writepath

    initiate_experiment(args)

    """ 
    Optional partial maze view setting.
    Input is a grid of cells, either the full maze (height x width) or a partial view (view_kernel_size*2+1 x view_kernel_size*2+1)
    Output is still a 1-size action idx over 4-discrete values (0,1,2,3) -> (N,S,E,W). Look up maze definition for more.
    
    Render occurs post pure-exploration only
    """