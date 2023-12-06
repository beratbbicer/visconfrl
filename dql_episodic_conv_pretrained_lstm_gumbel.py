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
import pretrain
from random_maze1_3 import RandomMaze

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class QNetwork(nn.Module):
    def __init__(self, action_dim, path, freeze=True):
        super(QNetwork, self).__init__()
        self.action_dim = action_dim
        self.freeze = freeze

        # ====================================================================================
        # Pretrained FE Layers
        # ====================================================================================
        model, checkpoint = pretrain.load_model(path)
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
        linear = nn.Linear(self.hidden_dim, self.action_dim)
        nn.init.kaiming_uniform_(linear.weight, nonlinearity='linear')
        self.output_layer = linear

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
        out = self.output_layer(out.view(b,-1))
        return out, h0_out, c0_out

class GumbelQNetwork(nn.Module):
    def __init__(self, action_dim, weightspath, freeze, init_temperature=1.0, min_temperature=0.1, max_temperature=1.0, anneal_rate=1e-5):
        super(GumbelQNetwork, self).__init__()
        self.model = QNetwork(action_dim, weightspath, freeze)
        self.temperature = nn.Parameter(torch.ones(1) * init_temperature)
        self.min_temperature = nn.Parameter(torch.tensor(min_temperature), requires_grad=False)
        self.max_temperature = nn.Parameter(torch.tensor(max_temperature), requires_grad=False)
        self.anneal_rate = anneal_rate

    def gumbel_softmax_sample(self, logits, temperature=1.0):
        u = torch.rand_like(logits)
        gumbel_sample = -torch.log(-torch.log(u + 1e-20) + 1e-20)
        # gumbel_softmax_sample = F.softmax((logits + gumbel_sample) * temperature, dim=-1)
        gumbel_softmax_sample = logits + gumbel_sample * temperature
        return gumbel_softmax_sample

    def forward(self, state, hs, cs, temperature=None):
        q_values, hs_out, cs_out = self.model(state, hs, cs)
        gumbel_q_values = self.gumbel_softmax_sample(q_values, temperature=temperature if temperature is not None else self.temperature)
        action = torch.argmax(gumbel_q_values)

        # action = F.gumbel_softmax(logits, tau=temperature if temperature is not None else self.temperature, hard=True)
        return gumbel_q_values, action, hs_out, cs_out

    def anneal_temperature(self):
        self.temperature.data = torch.clip(self.temperature * np.exp(-self.anneal_rate), self.min_temperature, self.max_temperature)

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

def evaluate(q_network, env, episode_avg, device, eval_episode_max_step, double, render=False, one_temperature=True):
    reward_cache = []
    while len(reward_cache) < episode_avg:
        eval_state = env.reset()[0]
        if double:
            hs_eval, cs_eval = q_network.model.initialize_lstm_states(1, device, torch.DoubleTensor)
        else:
            hs_eval, cs_eval = q_network.model.initialize_lstm_states(1, device, torch.FloatTensor)

        terminated, total_reward, eval_steps = False, 0, 0
                            
        while terminated == False and eval_steps < eval_episode_max_step:
            with torch.no_grad():
                if render:
                    env.render()

                eval_state_tensor = torch.from_numpy(eval_state.astype(np.float32))
                if double:
                    eval_state_tensor = eval_state_tensor.double()
                eval_state_tensor = eval_state_tensor.to(device)

                '''
                if one_temperature:
                    q_values_eval, action, hs_eval, cs_eval = q_network(eval_state_tensor, hs_eval, cs_eval)
                else:
                    q_values_eval, action, hs_eval, cs_eval = q_network(eval_state_tensor, hs_eval, cs_eval, temperature=1e-8)
                '''
                q_values_eval, action, hs_eval, cs_eval = q_network(eval_state_tensor, hs_eval, cs_eval, temperature=1e-8)
                eval_state, reward, terminated, _, _  = env.step(action.item())
                total_reward += reward
                eval_steps += 1

        if render:
            env.render()  

        reward_cache.append(total_reward)

    avg_reward = sum(reward_cache) / float(len(reward_cache))
    return avg_reward

def run_experiment(args):
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

    # q_network = QNetwork(args.action_dim, args.weightspath, not args.unfreeze_policy)
    # target_q_network = QNetwork(args.action_dim, argtemperature_anneal_rates.weightspath, not args.unfreeze_policy)
    q_network = GumbelQNetwork(args.action_dim, args.weightspath, not args.unfreeze_policy,\
                               args.init_temperature, args.min_temperature, args.max_temperature, args.temperature_anneal_rate)
    target_q_network = GumbelQNetwork(args.action_dim, args.weightspath, not args.unfreeze_policy,\
                                      args.init_temperature, args.min_temperature, args.max_temperature, args.temperature_anneal_rate)

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

    # Main training loop
    for episode in range(args.max_steps):
        # Temperature Decay
        if episode > 0 and episode % args.temperature_anneal_step == 0:
            # anneal_amount = np.exp(-q_network.anneal_rate)
            # rate = torch.max(q_network.temperature * np.exp(-q_network.anneal_rate),torch.tensor(q_network.min_temperature))
            q_network.anneal_temperature()
            target_q_network.anneal_temperature()

        state = env.reset()[0] # Get initial state
        terminated = False

        if args.double:
            hs, cs = q_network.model.initialize_lstm_states(1, args.device, torch.DoubleTensor)
        else:
            hs, cs = q_network.model.initialize_lstm_states(1, args.device, torch.FloatTensor)

        while terminated == False:
            state_tensor = torch.from_numpy(state.astype(np.float32))
            if args.double:
                state_tensor = state_tensor.double()
            state_tensor = state_tensor.to(args.device)

            if args.render:
                env.render()

            with torch.no_grad():
                q_network.eval()
                q_values, action, hs_out, cs_out = q_network(state_tensor, hs, cs)
                next_state, reward, terminated, _, _  = env.step(action.item())
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
                if args.one_temperature:
                    next_q_values, _, next_hs_out, next_cs_out = target_q_network(next_state_batch, next_h0_batch, next_c0_batch)
                else:
                    next_q_values, _, next_hs_out, next_cs_out = target_q_network(next_state_batch, next_h0_batch, next_c0_batch, args.policy_temperature)

                target_q_values = reward_batch + args.gamma * (1 - done_batch) * next_q_values

            if args.one_temperature:
                q_values, _, _, _ = q_network(state_batch, h0_batch, c0_batch)
            else:
                q_values, _, _, _ = q_network(state_batch, h0_batch, c0_batch, 1e-8)

            loss = F.l1_loss(q_values, target_q_values)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q_network.parameters(), max_norm=4)
            optimizer.step()

            for param, target_param in zip(q_network.parameters(), target_q_network.parameters()):
                target_param.data.copy_(param.data)

        if args.render:
            env.render()

        # ==================================================================================
        # Validation
        if episode > args.eval_start and episode % args.eval_steps == 0:
            q_network.eval()
            avg_reward = evaluate(q_network, env, args.episode_avg, args.device, args.eval_episode_max_step, args.double, args.render, args.one_temperature)
            q_network.train()
            eval_avg_rewards_cache.append([episode // args.eval_steps, avg_reward])
            print(f"[{args.rootname}] Episode: {episode:4d} -> Latest Eval Reward:{avg_reward: >7.4f}")
            
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

    with open(f'{args.writepath}/figure.pkl', 'wb') as f:
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
        }, f'{args.writepath}/weights.pth')

def initialize_experiment():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weightspath', default='./download/pretrain/2_101_101_0.9_0.9_13000_600_split/Ag58NjWa/e99.pth', type=str, help='Path to pretrained weights.')
    parser.add_argument('--unfreeze_critic', action='store_true')
    parser.add_argument('--unfreeze_policy', action='store_true')
    
    # TD3 Arguments
    parser.add_argument('--gamma', type=float, default=0.95) # Future reward scaling (discount)
    parser.add_argument('--buffer_size', type=int, default=-1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_steps', type=int, default=100000000)
    parser.add_argument('--eval_steps', type=int, default=1000)
    parser.add_argument('--eval_start', default=1000, type=int)
    parser.add_argument('--episode_avg', default=10, type=int)
    parser.add_argument('--eval_episode_max_step', default=1000, type=int)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--one_temperature', action='store_true')

    # DQL Gumbel-Softmax Arguments
    parser.add_argument("--init_temperature", type=float, default=1.0)
    parser.add_argument("--min_temperature", type=float, default=0.1)
    parser.add_argument("--max_temperature", type=float, default=1.0)
    parser.add_argument("--policy_temperature", type=float, default=0.1)
    parser.add_argument("--temperature_anneal_rate", type=float, default=1e-6)
    parser.add_argument("--temperature_anneal_step", type=float, default=100)

    # Env Arguments
    parser.add_argument('--window_size', default=256, type=int)
    parser.add_argument("--mazepath", type=str, default="./mazes_7/width-7_height-7_complexity-0.9_density-0.9_HaZkfFSrmPtakGDH.pkl")
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
        gamma:{args.gamma}, buffer_size:{args.buffer_size}, batch_size:{args.batch_size}, max_steps:{args.max_steps},\n\
        eval_steps:{args.eval_steps}, device:{args.device}, double:{args.double}, lr:{args.lr}, episode_avg:{args.episode_avg}\n\
        render:{args.render}, dump:{args.dump}, eval_start:{args.eval_start}, eval_episode_max_step:{args.eval_episode_max_step}\n\
        init_temperature:{args.init_temperature}, min_temperature:{args.min_temperature}, max_temperature:{args.max_temperature}, policy_temperature:{args.policy_temperature}\n\
        temperature_anneal_rate: {args.temperature_anneal_rate}, temperature_anneal_step: {args.temperature_anneal_step}, one_temperature:{args.one_temperature}\n\n\
        window_size:{args.window_size}, full_maze:{args.full_maze}, maze_view_kernel_size:{args.maze_view_kernel_size}, weight_decay:{args.weight_decay}\n"
    print(settings)

    args.action_low, args.action_high, args.action_dim = 0, 3, 4
    args.device = torch.device(args.device)

    writepath = f'./experiments/dql_conv_pretrained_lstm_gumbel/{args.mazepath.split("/")[-1].split(".pkl")[0].split("_")[-1]}/{args.rootname}'
    # writepath = f'./experiments/dummy/{args.mazepath.split("/")[-1].split(".pkl")[0].split("_")[-1]}/{args.rootname}'
    Path(writepath).mkdir(parents=True, exist_ok=True)
    args.writepath = writepath
    
    args.render, args.dump = False, True
    run_experiment(args)

def load_model(path, device):
    checkpoint = torch.load(path)
    q_network = QNetwork(checkpoint['args'].action_dim, checkpoint['args'].weightspath, not checkpoint['args'].unfreeze_policy)
    q_network.load_state_dict(checkpoint['q_network'])
    q_network = q_network.to(device)
    q_network.eval()
    return q_network, checkpoint

def evaluate_agent():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weightspath', default='./experiments/dql_conv_pretrained_lstm/HaZkfFSrmPtakGDH/8ZQ5LpxO/weights.pth', type=str, help='Path to pretrained weights.')
    parser.add_argument('--window_size', default=256, type=int)
    parser.add_argument("--eval_episodes", type=int, default=100)
    parser.add_argument('--eval_episode_max_step', default=1000, type=int)
    parser.add_argument('--device', default=0, type=int)
    args = parser.parse_args()
    args.device = torch.device(args.device)
    
    q_network, checkpoint = load_model(args.weightspath, args.device)
    # q_network = q_network.float().to("cpu")

    env = RandomMaze(window_size=args.window_size, partial_view=not checkpoint['args'].full_maze,\
                     view_kernel_size=checkpoint['args'].maze_view_kernel_size, render_mode="human")

    _, _ = env.reset(seed=checkpoint['args'].fixed_seed) # Set seed
    env.generate_maze(checkpoint['args'].mazepath) # Generate maze & objects
    q_network.eval()
    avg_reward = evaluate(q_network, env, args.eval_episodes, args.device, args.eval_episode_max_step, False, True)
    q_network.train()
    print(f"Avg Reward:{avg_reward: >7.4f}")

if __name__ == "__main__":
    initialize_experiment()
    # evaluate_agent()

    '''
    In this setup, I use the previous Q-Network that predicts future action Q-Values 
        but employ Gumbel Softmax reparameterization when dealing with random noise.

    Since this ain't a policy network and Q-Network directly predicts future Q-Values,
        the main advantage of applying Gumbel Softmax trick is to make the gradient flow "continuous"
        despite the fact that it is indeed continous in the first place.

    Also let the temperature be a model parameter and allow the network to change it as it wishes.
        It can increase or decrease, whenever the model likes, except manual annealing that counteracts large values. 
    '''