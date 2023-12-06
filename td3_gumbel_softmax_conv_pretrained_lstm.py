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
import pretrain
import gc

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def initialize_bias_linear(layer):
    if layer.bias is not None:
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(layer.bias, -bound, bound)

class CriticNetwork(nn.Module):
    def __init__(self, action_dim, path, freeze=True):
        super(CriticNetwork, self).__init__()
        self.action_dim = action_dim
        self.freeze = freeze

        # ====================================================================================
        # Pretrained FE Layers 1
        # ====================================================================================
        self.state_dim, self.hidden_dim, self.layers1, _ = self.load_model(path)
        linear = nn.Linear(self.hidden_dim // 4, 1)
        nn.init.kaiming_uniform_(linear.weight, nonlinearity='linear')
        initialize_bias_linear(linear)

        self.output_layer1 = nn.Sequential(
            nn.Linear(self.hidden_dim + self.action_dim, self.hidden_dim // 4),
            nn.LeakyReLU(negative_slope=0.01),
            linear
        )
        # ====================================================================================
        # Pretrained FE Layers 2
        # ====================================================================================
        _, _, self.layers2, _ = self.load_model(path)

        linear = nn.Linear(self.hidden_dim // 4, 1)
        nn.init.kaiming_uniform_(linear.weight, nonlinearity='linear')
        initialize_bias_linear(linear)

        self.output_layer2 = nn.Sequential(
            nn.Linear(self.hidden_dim + self.action_dim, self.hidden_dim // 4),
            nn.LeakyReLU(negative_slope=0.01),
            linear
        )
        # ====================================================================================
        # LSTM Layer
        # ====================================================================================
        self.rnn1 = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
        self.rnn2 = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
    
    def load_model(self, path):
        model, _ = pretrain.load_model(path)
        layers = model.layers
        if self.freeze:
            for param in layers.parameters():
                param.requires_grad = False
        return model.state_dim, model.hidden_dim, layers, model.output_layer

    def forward_pass(self, state, layers, rnn, h0, c0):
        if len(state.size()) == 2:
            b,h,w = state.unsqueeze(0).size()
        elif len(state.size()) == 3:
            b,h,w = state.size()
        else:
            b,_,h,w = state.size()
        out = state.view(b, 1, h, w)

        for layer in layers:
            out = layer(out)
        out = F.adaptive_avg_pool2d(out, output_size=(1,1)).view(b, 1, -1)

        # ================================================================
        # LSTM
        # Input and output tensors have batch_size as the first dimension!
        out, (_, _) = rnn(out, (h0, c0))
        out = out.view(b,-1)
        return out

    def forward(self, state, action, h0, c0):
        # Concatenate state and action
        x_q1 = state
        x_q2 = torch.clone(state)
        
        q1 = self.forward_pass(x_q1, self.layers1, self.rnn1, h0, c0)
        q2 = self.forward_pass(x_q2, self.layers2, self.rnn2, h0, c0)

        q1 = torch.concatenate((q1, action), dim=1)
        q2 = torch.concatenate((q2, action), dim=1)
        q1 = self.output_layer1(q1)
        q2 = self.output_layer2(q2)
        return q1, q2

class PolicyNetwork(nn.Module):
    def __init__(self, action_dim, path, freeze=True):
        super(PolicyNetwork, self).__init__()
        self.action_dim = action_dim

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
        self.rnn = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
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
        out = torch.tanh(self.output_layer(out.view(b,-1)))
        return out, h0_out, c0_out

class AnnealedGumbelSoftmaxPolicy(nn.Module):
    def __init__(self, action_dim, weightspath, freeze, init_temperature=1.0, min_temperature=0.1, max_temperature=1.0, anneal_rate=1e-5):
        super(AnnealedGumbelSoftmaxPolicy, self).__init__()
        self.model = PolicyNetwork(action_dim, weightspath, freeze)
        self.temperature = nn.Parameter(torch.ones(1) * init_temperature)
        self.min_temperature = nn.Parameter(torch.tensor(min_temperature), requires_grad=False)
        self.max_temperature = nn.Parameter(torch.tensor(max_temperature), requires_grad=False)
        self.anneal_rate = anneal_rate

    def gumbel_softmax_sample(self, logits, temperature=1.0):
        u = torch.rand_like(logits)
        gumbel_sample = -torch.log(-torch.log(u + 1e-20) + 1e-20)
        gumbel_softmax_sample = F.softmax(logits + gumbel_sample * temperature, dim=-1)
        return gumbel_softmax_sample

    def forward(self, state, h0, c0, temperature=None):
        logits, h0_out, c0_out = self.model(state, h0, c0)
        gumbel_probs = self.gumbel_softmax_sample(logits, temperature=temperature if temperature is not None else self.temperature)
        _, action = torch.max(gumbel_probs, dim=-1)

        # action = F.gumbel_softmax(logits, tau=temperature if temperature is not None else self.temperature, hard=True)
        return gumbel_probs, action, h0_out, c0_out

    def anneal_temperature(self):
        self.temperature.data = torch.clip(self.temperature * np.exp(-self.anneal_rate), self.min_temperature, self.max_temperature)

class ReplayBuffer:
    def __init__(self, max_size, full_maze):
        if max_size <= 0:
            self.max_size = math.inf
        else:
            self.max_size = max_size

        self.full_maze = full_maze
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

def get_action(actor, state_tensor, h0, c0, temperature=None):
    probs, selected_action, h0_out, c0_out = actor(state_tensor, h0, c0, temperature)
    return probs, selected_action, h0_out, c0_out

def update_critic(args, critic, target_critic, batch, gamma, critic_optimizer, target_actor, temperature=None, one_temperature=False):
    critic_optimizer.zero_grad()
    state_batch, action_batch, reward_batch, next_state_batch, done_batch, h0_batch, c0_batch, next_h0_batch, next_c0_batch = batch

    if args.double:
        state_batch = state_batch.double()
        next_state_batch = next_state_batch.double()
    else:
        state_batch = state_batch.float()
        next_state_batch = next_state_batch.float()

    # Take the min q value between the two critics
    with torch.no_grad():
        # Accept t+1 state and t+1 state tensors as input to the target actor to determine t+2 action probs
        if one_temperature:
            next_action_probs, _, h0_out_2steps, c0_out_2steps = get_action(target_actor, next_state_batch, next_h0_batch, next_c0_batch)
        else:
            next_action_probs, _, h0_out_2steps, c0_out_2steps = get_action(target_actor, next_state_batch, next_h0_batch, next_c0_batch, temperature)

        # Use the t+2 action probs and t+1 state, as well as t+1 state tensors to determine t+2 q values
        # target_q1, target_q2 = target_critic(next_state_batch, next_action_probs, next_h0_batch, next_c0_batch)
        target_q1, target_q2 = target_critic(next_state_batch, next_action_probs, h0_out_2steps, c0_out_2steps)

        target_q = torch.min(target_q1, target_q2)
        target_q = reward_batch + gamma * (1 - done_batch) * target_q

    # Use the target q value to update both critics
    q1, q2 = critic(state_batch, action_batch, h0_batch, c0_batch)
    critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=2)
    critic_optimizer.step()

def update_actor(args, actor, critic, actor_optimizer, replay_buffer, batch_size, double, device, one_temperature=False):
    actor_optimizer.zero_grad()
    state, _, _, _, _, h0_batch, c0_batch, _, _ = replay_buffer.sample(batch_size, double, device)

    if args.double:
        state = state.double()
    else:
        state = state.float()

    # Take a deterministic policy step byt setting temperature to near-zero
    if one_temperature:
        probs, _, _, _ = get_action(actor, state, h0_batch, c0_batch)
    else:
        probs, _, _, _ = get_action(actor, state, h0_batch, c0_batch, temperature=1e-8)

    # Update the actor based on critics Q1 prediction
    actor_loss = -critic(state, probs, h0_batch, c0_batch)[0].mean()
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=2)
    actor_optimizer.step()

def soft_update_target_network(model, target_model, tau):
    for param, target_param in zip(model.parameters(), target_model.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

def evaluate(actor, env, episode_avg, device, eval_episode_max_step, double, render=False, one_temperature=True):
    reward_cache = []
    while len(reward_cache) < episode_avg:
        eval_state = env.reset()[0]
        terminated, total_reward, eval_steps = False, 0, 0

        if double:
            h0_actor_eval, c0_actor_eval = actor.model.initialize_lstm_states(1, device, torch.DoubleTensor)
        else:
            h0_actor_eval, c0_actor_eval = actor.model.initialize_lstm_states(1, device, torch.FloatTensor)

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
                    _, action, h0_out_eval, c0_out_eval = get_action(actor, eval_state_tensor, h0_actor_eval, c0_actor_eval)
                else:
                    _, action, h0_out_eval, c0_out_eval = get_action(actor, eval_state_tensor, h0_actor_eval, c0_actor_eval, temperature=1e-8)
                '''
                _, action, h0_out_eval, c0_out_eval = get_action(actor, eval_state_tensor, h0_actor_eval, c0_actor_eval, temperature=1e-8)
                eval_next_state, reward, terminated, _, _ = env.step(int(action.item()))
                eval_state, h0_actor_eval, c0_actor_eval = eval_next_state, h0_out_eval, c0_out_eval
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

    actor = AnnealedGumbelSoftmaxPolicy(args.action_dim, args.weightspath, not args.unfreeze_policy,\
                               args.init_temperature, args.min_temperature, args.max_temperature, args.temperature_anneal_rate)
    critic = CriticNetwork(args.action_dim, args.weightspath, not args.unfreeze_critic)
    target_actor = AnnealedGumbelSoftmaxPolicy(args.action_dim, args.weightspath, not args.unfreeze_policy,\
                               args.init_temperature, args.min_temperature, args.max_temperature, args.temperature_anneal_rate)
    target_critic = CriticNetwork(args.action_dim, args.weightspath, not args.unfreeze_critic)

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
    state = env.reset()[0]  # Get initial state
    eval_avg_rewards_cache = []

    if args.double:
        h0_actor, c0_actor = actor.model.initialize_lstm_states(1, args.device, torch.DoubleTensor)
        # h0_target, c0_target = target_actor.model.initialize_lstm_states(args.batch_size, args.device, torch.DoubleTensor)
    else:
        h0_actor, c0_actor = actor.model.initialize_lstm_states(1, args.device, torch.FloatTensor)
        # h0_target, c0_target = target_actor.model.initialize_lstm_states(args.batch_size, args.device, torch.FloatTensor)

    # Expected buffer content:
    # state_batch, action_batch, reward_batch, next_state_batch, done_batch, h0_batch, c0_batch, next_h0_batch, next_c0_batch
    # ---- Ignored: h0_target_batch, c0_target_batch, next_h0_target_batch, next_c0_target_batch

    # Main training loop
    for step in range(args.max_steps):
        state_tensor = torch.from_numpy(state.astype(np.float32))
        if args.double:
            state_tensor = state_tensor.double()
        state_tensor = state_tensor.to(args.device)

        if step < args.exploration_steps:
            # Pure exploratory policy - random actions
            with torch.no_grad():
                if args.double:
                    # action = torch.DoubleTensor(np.random.rand(low=args.action_low, high=args.action_high+1, size = (1,1))).to(args.device)
                    action_probs = F.softmax(torch.rand(size=(1, args.action_dim)), dim=-1).double().to(args.device)
                else:
                    # action = torch.FloatTensor(np.random.randint(low=args.action_low, high=args.action_high+1, size = (1,1))).to(args.device)
                    action_probs = F.softmax(torch.rand(size=(1, args.action_dim)), dim=-1).float().to(args.device)
                action = torch.argmax(action_probs, dim=-1).item()
                next_state, reward, terminated, _, _ = env.step(action)

                # Push zero state tensors as we haven't seen the policy in action yet - this is just to fill the buffer and get the policy started.
                # With limited-size buffer these will eventually go away, replaced by meaningful state tensors.
                replay_buffer.add(state, action_probs, reward, next_state, terminated,\
                                  torch.zeros_like(h0_actor), torch.zeros_like(c0_actor), torch.zeros_like(h0_actor), torch.zeros_like(c0_actor))
                                  # torch.zeros_like(h0_target), torch.zeros_like(c0_target), torch.zeros_like(h0_target), torch.zeros_like(c0_target))

            if terminated:
                state = env.reset()[0]
            else:
                state = next_state

            gc.collect()
        elif step == args.exploration_steps:
            state = env.reset()[0]

            if args.double:
                h0_actor, c0_actor = actor.model.initialize_lstm_states(1, args.device, torch.DoubleTensor)
            else:
                h0_actor, c0_actor = actor.model.initialize_lstm_states(1, args.device, torch.FloatTensor)
        else:                
            if args.render:
                env.render()

            # Test the smallest value of policy temperature after annealing, ignoring its threshold.
            # rate = np.exp(-args.temperature_anneal_rate) ** (args.max_steps / args.temperature_anneal_step)

            # Anneal policy temperature
            if step % args.temperature_anneal_step == 0:
                actor.anneal_temperature()
                target_actor.anneal_temperature()

            # Off-Policy Exploration - add exploration noise to the action
            with torch.no_grad():
                actor.eval()
                action_probs, action, h0_out, c0_out = get_action(actor, state_tensor, h0_actor, c0_actor)
                next_state, reward, terminated, _, _ = env.step(action.item())
                replay_buffer.add(state, action_probs, reward, next_state, terminated, h0_actor, c0_actor, h0_out, c0_out)
                h0_actor, c0_actor = h0_out, c0_out
                actor.train()

            # Policy Smoothing - add policy noise to the action
            sample_size = min(replay_buffer.__len__(), args.batch_size)
            batch = replay_buffer.sample(sample_size, args.double, args.device)
            update_critic(args, critic, target_critic, batch, args.gamma, critic_optimizer, target_actor, args.policy_temperature, args.one_temperature)

            if step % args.policy_freq == 0:
                update_actor(args, actor,critic, actor_optimizer, replay_buffer, sample_size, args.double, args.device, args.one_temperature)
                soft_update_target_network(actor, target_actor, args.tau)
                soft_update_target_network(critic, target_critic, args.tau)

            if terminated:
                state = env.reset()[0]

                if args.double:
                    h0_actor, c0_actor = actor.model.initialize_lstm_states(1, args.device, torch.DoubleTensor)
                else:
                    h0_actor, c0_actor = actor.model.initialize_lstm_states(1, args.device, torch.FloatTensor)
            else:
                state = next_state

            # ==================================================================================
            # Validation
            if step % args.eval_steps == 0:
                actor.eval()
                avg_reward = evaluate(actor, env, args.episode_avg, args.device, args.eval_episode_max_step, args.double, args.render, args.one_temperature)
                actor.train()
                eval_avg_rewards_cache.append([int(step / args.eval_steps), avg_reward])
                print(f"[{args.rootname}] Step: {step:7d} -> Latest Eval Reward:{avg_reward: >8.4f}")

                if avg_reward >= 1:
                    print(f"Task solved in {step:7d} steps.")
                    break
            
            gc.collect()
    env.close()

    # Save plots
    steps, rewards = zip(*eval_avg_rewards_cache)
    fig = plt.figure(figsize=(20, 8))
    x = np.arange(len(steps))
    plt.xticks(ticks=x, labels=steps, rotation=90)
    plt.plot(x, rewards, label="Rewards", color="red", linewidth=2, linestyle=":", marker="o")
    plt.xlabel("Steps (x5000)")
    plt.ylabel("Reward")
    plt.title("Avg. Reward Throughout Training")
    plt.legend()
    plt.tick_params(axis="x", which="major", labelsize=5)
    plt.tick_params(axis="y", which="major", labelsize=10)
    plt.tight_layout()
    # plt.show()

    with open(f"{args.writepath}.pkl", "wb") as f:
        pickle.dump({
                "fig": fig,
                "steps": steps,
                "rewards": rewards,
            }, f)

    # Dump The Modelbuffer_size
    if args.dump:
        torch.save(
            {
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "target_actor": target_actor.state_dict(),
                "target_critic": target_critic.state_dict(),
                "actor_optimizer": actor_optimizer.state_dict(),
                "critic_optimizer": critic_optimizer.state_dict(),
                "replay_buffer": replay_buffer,
                "fixed_seed ": fixed_seed,
                "args": args,
            }, f"{args.writepath}.pth")

def initialize_experiment():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weightspath', default='./download/pretrain/2_101_101_0.9_0.9_13000_600_split/Ag58NjWa/e99.pth', type=str, help='Path to pretrained weights.')
    parser.add_argument('--unfreeze_critic', action='store_true')
    parser.add_argument('--unfreeze_policy', action='store_true')

    # TD3 Arguments
    parser.add_argument("--gamma", type=float, default=0.99)  # Future reward scaling (discount)
    parser.add_argument("--tau", type=float, default=0.005)  # Target update rate
    parser.add_argument("--policy_freq", type=int, default=2)
    parser.add_argument("--buffer_size", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_steps", type=int, default=1000000)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--exploration_steps", type=int, default=128)
    parser.add_argument("--episode_avg", default=10, type=int) 
    parser.add_argument("--eval_episode_max_step", default=100, type=int)
    parser.add_argument("--actor_lr", type=float, default=1e-4)
    parser.add_argument("--critic_lr", type=float, default=1e-4)
    
    parser.add_argument("--init_temperature", type=float, default=1.0)
    parser.add_argument("--min_temperature", type=float, default=0.1)
    parser.add_argument("--max_temperature", type=float, default=1.0)
    parser.add_argument("--policy_temperature", type=float, default=0.1)
    parser.add_argument("--temperature_anneal_rate", type=float, default=1e-6)
    parser.add_argument("--temperature_anneal_step", type=float, default=100)
    parser.add_argument('--one_temperature', action='store_true')

    # Env Arguments
    parser.add_argument("--window_size", default=256, type=int)
    parser.add_argument("--full_maze", action="store_true")
    parser.add_argument("--mazepath", type=str, default="./mazes_51/width-51_height-51_complexity-0.9_density-0.9_bbnYDPhsjerrENoS.pkl")

    # Misc Arguments
    parser.add_argument("--weight_decay", default=1e-5, type=float)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--double", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--dump", action="store_true")
    parser.add_argument("--rootname", type=str, help="Name for experiment root folder. Defaults to length-8 random string.",\
                        default="".join(random.SystemRandom().choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(8)))
    args = parser.parse_args()

    args.device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    args.fixed_seed = fixed_seed
    args.maze_view_kernel_size = 2

    settings = f"Configuration {args.rootname} ->\n\
        gamma:{args.gamma}, tau:{args.tau}, policy_freq:{args.policy_freq}, buffer_size:{args.buffer_size}, batch_size:{args.batch_size}\n\
        max_steps:{args.max_steps}, eval_steps:{args.eval_steps}, device:{args.device}, double:{args.double}, actor_lr:{args.actor_lr}\n\
        critic_lr:{args.critic_lr}, episode_avg:{args.episode_avg}, render:{args.render}, dump:{args.dump}, weight_decay:{args.weight_decay}\n\
        exploration_steps:{args.exploration_steps}, eval_episode_max_step:{args.eval_episode_max_step}, policy_temperature:{args.policy_temperature}\n\
        init_temperature:{args.init_temperature}, min_temperature:{args.min_temperature}, max_temperature:{args.max_temperature}, one_temperature:{args.one_temperature}\n\
        anneal_rate:{args.temperature_anneal_rate}, anneal_step:{args.temperature_anneal_step}, unfreeze_critic:{args.unfreeze_critic}, unfreeze_policy:{args.unfreeze_policy}\n\n\
        window_size:{args.window_size}, full_maze:{args.full_maze}, maze_view_kernel_size:{args.maze_view_kernel_size}\n\
        mazepath:{args.mazepath}\n"
    print(settings)

    args.action_low, args.action_high, args.action_dim = 0, 3, 4
    args.device = torch.device(args.device)

    # args.one_temperature = True
        
    writepath = f'./experiments/td3_conv_pretrained_lstm/{args.mazepath.split("/")[-1].split(".pkl")[0].split("_")[-1]}/{args.rootname}'
    # writepath = f'./experiments/dummy/{args.mazepath.split("/")[-1].split(".pkl")[0].split("_")[-1]}/{args.rootname}'
    Path(writepath).mkdir(parents=True, exist_ok=True)
    args.writepath = writepath

    run_experiment(args)

    """ 
    Based on pretrained2, but with an LSTM attached:
        I think right now, the policy cannot learn from the future and past observations and is instead only learning from the current observation.
        This is all good but then the agent gets stuck at the same places over and over again.
        So I thought why not attach an LSTM on top of the pretrained conv layers and see if that helps.
        The idea is that the LSTM will learn to predict the next observation based on the current observation and the action taken.

        Of course this approach is a bit buffer-heavy since we need to store the LSTM states for each step, hopefully for now w/o the gradients attached.
        Otherwise we're fucked cause memory's gonna overflow soo quickly.

        So the idea is this: 
            During exploration and evaluation both, you don't need to compute gradients anyway.
            You can simply use the previous LSTM state outputs w/o doing a buffer lookup: These state tensors (hidden/cell state) will be saved and only reset on env.reset()
            When env.reset() is invoked, you're gonna manually reset the state tensors with preferably constant initialization so the agent knows it's a reset.
            The most important thing to remember here is to dump the detached state tensors to the lookup and to keep the buffer size limited.
            Otherwise, you're gonna observe some noisy tensors from the earlier episodes. 
            Keep in mind that the output of the LSTM will be used to produce the action.

            During gradient updates, when the buffer samples from the observations, part of these observations now consist of state tensors.
            So, LSTM will use these as its state tensors and produce a new output. These state tensors will de dumped to the buffer as well, as part of the obdservation.    

        As I am making things up on the fly, I am pushing zero-tensors to the buffer during pure-explorative policy.
            With a limited-size buffer, these will eventually go away, replaced by meaningful state tensors.
            LSTMs when they see these zero tensors, they will simply ignore them. This is the default behavior after all.

        In critic update:
            Accept t+1 state and t+1 state tensors as input to the target actor to determine t+2 action probs
            Use the t+2 action probs and t+1 state, as well as t+1 state tensors to determine t+2 q values
            This logic is similar in HW3 as well so I think we are safe. 


    """

    # python td3_gumbel_softmax_conv.py --mazepath ./mazes_51/width-51_height-51_complexity-0.9_density-0.9_bbnYDPhsjerrENoS.pkl --dump --device 1 --maze_view_kernel_size

def load_model(path, device):
    checkpoint = torch.load(path)
    q_network = AnnealedGumbelSoftmaxPolicy(checkpoint['args'].action_dim, checkpoint['args'].weightspath, not checkpoint['args'].unfreeze_policy,\
                                            checkpoint['args'].init_temperature, checkpoint['args'].min_temperature, checkpoint['args'].max_temperature,\
                                            checkpoint['args'].temperature_anneal_rate)
    q_network.load_state_dict(checkpoint['actor'])
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
    
    actor, checkpoint = load_model(args.weightspath, args.device)
    # q_network = q_network.float().to("cpu")

    env = RandomMaze(window_size=args.window_size, partial_view=not checkpoint['args'].full_maze,\
                     view_kernel_size=checkpoint['args'].maze_view_kernel_size, render_mode="human")

    _, _ = env.reset(seed=checkpoint['args'].fixed_seed) # Set seed
    env.generate_maze(checkpoint['args'].mazepath) # Generate maze & objects
    actor.eval()
    avg_reward = evaluate(actor, env, args.eval_episodes, args.device, args.eval_episode_max_step, False, True, checkpoint['args'].one_temperature)
    print(f"Avg Reward:{avg_reward: >7.4f}")

if __name__ == "__main__":
    initialize_experiment()
    # evaluate_agent()