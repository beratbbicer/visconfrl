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
import glob

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

class AnnealedGumbelSoftmaxPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, init_temperature=1.0, min_temperature=0.1, anneal_rate=1e-5):
        super(AnnealedGumbelSoftmaxPolicy, self).__init__()
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim)
        self.temperature = nn.Parameter(torch.ones(1) * init_temperature)
        self.min_temperature = min_temperature
        self.anneal_rate = anneal_rate

    def gumbel_softmax_sample(self, logits, temperature=1.0):
        u = torch.rand_like(logits)
        gumbel_sample = -torch.log(-torch.log(u + 1e-20) + 1e-20)
        gumbel_softmax_sample = F.softmax((logits + gumbel_sample) / temperature, dim=-1)
        return gumbel_softmax_sample

    def forward(self, state, temperature=None):
        logits = self.policy_net(state)
        gumbel_probs = self.gumbel_softmax_sample(logits, temperature=temperature if temperature is not None else self.temperature)
        _, action = torch.max(gumbel_probs, dim=-1)

        # action = F.gumbel_softmax(logits, tau=temperature if temperature is not None else self.temperature, hard=True)
        return gumbel_probs, action

    def anneal_temperature(self):
        self.temperature.data = torch.max(self.temperature * np.exp(-self.anneal_rate), torch.tensor(self.min_temperature))

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

def get_action(actor, state_tensor, action_low, action_high, temperature=None):
    probs, selected_action = actor(state_tensor, temperature)
    return probs, selected_action

def update_critic(critic, target_critic, batch, gamma, critic_optimizer, target_actor, action_low, action_high, temperature=None):
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
        next_action_probs, _ = get_action(target_actor, next_state_batch, action_low, action_high, temperature)
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
    probs, _ = get_action(actor, state, action_low, action_high, temperature=1e-8)

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

    actor = AnnealedGumbelSoftmaxPolicy(args.state_dim, args.action_dim, args.hidden_dim, args.init_temperature, args.min_exporation_temperature, args.temperature_anneal_rate)
    critic = CriticNetwork(args.state_dim, args.action_dim)
    target_actor = AnnealedGumbelSoftmaxPolicy(args.state_dim, args.action_dim, args.hidden_dim, args.init_temperature, args.min_exporation_temperature, args.temperature_anneal_rate)
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
    for step in range(args.max_steps):
        state_tensor = torch.from_numpy(state.astype(np.float32)).reshape(1,-1)
        if args.double:
            state_tensor = state_tensor.double()
        state_tensor = state_tensor.to(args.device)

        if step < args.exploration_steps:
            # Pure exploratory policy - random actions
            with torch.no_grad():
                if args.double:
                    # action = torch.DoubleTensor(np.random.rand(low=args.action_low, high=args.action_high+1, size = (1,1))).to(args.device)
                    action_probs = F.softmax(torch.rand(size=(1,args.action_dim)), dim=-1).double().to(args.device)
                else:
                    # action = torch.FloatTensor(np.random.randint(low=args.action_low, high=args.action_high+1, size = (1,1))).to(args.device)
                    action_probs = F.softmax(torch.rand(size=(1,args.action_dim)), dim=-1).float().to(args.device)
                action = torch.argmax(action_probs, dim=-1).item()
                next_state, reward, terminated, _, _  = env.step(action)
                replay_buffer.add(state, action_probs, reward, next_state, terminated)
            
            if terminated:
                state = env.reset()[0]
            else:
                state = next_state
        else:
            if args.render:
                env.render()

            # Test the smallest value of policy temperature after annealing, ignoring its threshold.
            rate = np.exp(-args.temperature_anneal_rate) ** (args.max_steps / args.temperature_anneal_step) 

            # Anneal policy temperature
            if step % args.temperature_anneal_step == 0:
                actor.anneal_temperature()
                target_actor.anneal_temperature()

            # Off-Policy Exploration - add exploration noise to the action
            with torch.no_grad():
                action_probs, action = get_action(actor, state_tensor, args.action_low, args.action_high)
                next_state, reward, terminated, _, _  = env.step(action.item())
                replay_buffer.add(state, action_probs, reward, next_state, terminated)

            # Policy Smoothing - add policy noise to the action
            batch = replay_buffer.sample(args.batch_size, args.double, args.device)
            update_critic(critic, target_critic, batch, args.gamma, critic_optimizer, target_actor,\
                          args.action_low, args.action_high, args.policy_temperature)

            if step % args.policy_freq == 0:
                update_actor(actor, critic, actor_optimizer, replay_buffer, args.batch_size,\
                             args.double, args.device, args.action_low, args.action_high)
                soft_update_target_network(actor, target_actor, args.tau)
                soft_update_target_network(critic, target_critic, args.tau)

            if terminated:
                state = env.reset()[0]
            else:
                state = next_state

            # ==================================================================================
            # Validation
            if step % args.eval_steps == 0:
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
                            
                            # During evaluation, make the policy deterministic:
                            #   Setting temperature as close to 0 as possible, but not exactly 0 because that would cause the Gumbel-Softmax to crash
                            #   Making the noise scaling zero. 
                            _, action = get_action(actor, eval_state_tensor, args.action_low, args.action_high, temperature=1e-8)
                            eval_state, reward, terminated, _, _  = env.step(int(action.item()))
                            total_reward += reward
                            eval_steps += 1
                            
                    reward_cache.append(total_reward)
            
                avg_reward = sum(reward_cache) / float(len(reward_cache))
                eval_avg_rewards_cache.append([int(step / args.eval_steps),avg_reward])
                print(f"Step: {step:7d} -> Latest Eval Reward:{avg_reward: >8.3f}")
        
                if avg_reward > 10:
                    print(f"Task solved in {step:7d} steps.")
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
    parser.add_argument('--tau', type=float, default=0.005) # Target update rate
    parser.add_argument('--policy_freq', type=int, default=2)
    parser.add_argument('--buffer_size', type=int, default=-1)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--max_steps', type=int, default=1000000)
    parser.add_argument('--eval_steps', type=int, default=10000)
    parser.add_argument('--exploration_steps', type=int, default=10000)
    parser.add_argument('--episode_avg', default=10, type=int)
    parser.add_argument('--eval_episode_max_step', default=10000, type=int)
    parser.add_argument('--actor_lr', type=float, default=1e-3)
    parser.add_argument('--critic_lr', type=float, default=1e-3)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--init_temperature', type=float, default=1.0)
    parser.add_argument('--min_exporation_temperature', type=float, default=0.2)
    parser.add_argument('--policy_temperature', type=float, default=0.1)
    parser.add_argument('--temperature_anneal_rate', type=float, default=1.25e-5)
    parser.add_argument('--temperature_anneal_step', type=float, default=1)

    # Env Arguments
    parser.add_argument('--window_size', default=256, type=int)
    parser.add_argument('--full_maze', action='store_true')
    parser.add_argument('--maze_view_kernel_size', default=1, type=int)
    parser.add_argument('--mazepath', type=str, default='./mazes')

    # Misc Arguments
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--double', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--dump', action='store_true')
    parser.add_argument('--rootname',\
        default="".join(random.SystemRandom().choice(string.ascii_uppercase+string.ascii_lowercase+string.digits) for _ in range(8)),\
        type=str, help="Name for experiment root folder. Defaults to length-8 random string.")
    args = parser.parse_args()

    args.device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    args.fixed_seed = fixed_seed

    settings = f"Configuration maze_solver_td3_{args.rootname} ->\n\
        gamma:{args.gamma}, tau:{args.tau}, policy_freq:{args.policy_freq}, buffer_size:{args.buffer_size}, batch_size:{args.batch_size}\n\
        max_steps:{args.max_steps}, eval_steps:{args.eval_steps}, device:{args.device}, double:{args.double}, actor_lr:{args.actor_lr}\n\
        critic_lr:{args.critic_lr}, episode_avg:{args.episode_avg}, render:{args.render}, dump:{args.dump}, weight_decay:{args.weight_decay}\n\
        exploration_steps:{args.exploration_steps}, eval_episode_max_step:{args.eval_episode_max_step}, hidden_dim:{args.hidden_dim}\n\
        init_temperature:{args.init_temperature}, min_exporation_temperature:{args.min_exporation_temperature}, policy_temperature:{args.policy_temperature}\n\
        anneal_rate:{args.temperature_anneal_rate}, anneal_step:{args.temperature_anneal_step}\n\n\
        window_size:{args.window_size}, full_maze:{args.full_maze}, maze_view_kernel_size:{args.maze_view_kernel_size}\n\
        mazepath:{args.mazepath}\n"
    print(settings)

    args.action_low, args.action_high, args.action_dim = 0, 3, 4
    args.device = torch.device(args.device)

    writepath = f'./experiments/td3/{args.mazepath.split("/")[-1].split(".pkl")[0].split("_")[-1]}/{args.rootname}'
    Path(writepath).mkdir(parents=True, exist_ok=True)
    args.writepath = writepath

    initiate_experiment(args)

    """ 
    Adapting TD3 to for discrete action spaces using Gumbel-Softmax:
    - Change the actor to output raw logits
    - Use Gumbel-Softmax to sample actions from the logits
    - Apply temperature annealing to the Gumbel-Softmax so policy slowly becomes more deterministic
    - Modify the policy to output discrete actions:
        Output raw logits from the policy network
        Use Gumbel-Softmax to sample actions from the logits
        Eliminate policy and exploration noise from the action selection process since temperature parameter handles exploration already.
            Although, one consideration here is that in TD3 policy and exploration noise are independent sources of noise
            Removing them both and relying on temperature might not be the best idea since technically noise sources are identical.
    
            My initial solution is to use min_temperature parameter explicitly during policy updates so both noise sources are independent.
            Based on this, update_critic gets min_temperature as an argument and passes it to get_action() function.
            update_actor uses this implicitly by setting it to near-zero, so the policy is updated deterministically
            Requires further investigation!!!

        Convert get_action() into a glorified interface that simply returns the output of Gumbel-Softmax (i.e. the action)

    - During evaluation, make the policy deterministic:
        Setting temperature as close to 0 as possible, but not exactly 0 because that would cause the Gumbel-Softmax to crash, similar to update_actor
        Make the noise scaling zero, but since noise is eliminated from the action selection process this is not necessary.

    From ChatGPT:
        Traditional Softmax:
            Suppose you have a discrete action space with three actions [A, B, C], and you obtain probabilities from your policy network as [0.3, 0.6, 0.1]. 
            In a traditional softmax approach, you might choose the action with the highest probability, which is B in this case.

        Gumbel-Softmax:
            Now, let's consider the Gumbel-Softmax approach during training. You obtain logits from your policy network as [1.0, 2.0, 0.5]. 
            You add Gumbel noise to these logits and apply the softmax operation to obtain differentiable probabilities. This allows backpropagation during training.
            Now, the Gumbel-Softmax sampling process might give you [0.2, 0.7, 0.1] as the sampled probabilities. 
            During training, you use these probabilities for action selection.
        
        Advantage:
            Differentiability:
                The Gumbel-Softmax trick allows for the entire process (sampling from logits) to be differentiable. 
                This is crucial during training because you can backpropagate gradients through the sampling operation, enabling end-to-end training of your neural network.

            Exploration during Training:
                The Gumbel-Softmax introduces stochasticity during training, allowing the model to explore different actions even when the policy indicates 
                    a strong preference for a particular action. This exploration is beneficial for learning a more robust policy.

            Temperature Annealing:
                The temperature parameter in Gumbel-Softmax can be annealed during training. Higher temperature values lead to more exploration, 
                    while lower values make the sampling process approach the argmax operation. This annealing allows you to control the level of 
                    exploration during different phases of training.

        During Evaluation:
            During evaluation, when you need a concrete action, you can still use argmax on the probabilities obtained from the Gumbel-Softmax. 
            This allows you to make a deterministic decision based on the learned policy while still benefiting from the advantages mentioned above during training.

    In essence, additive noise may be disconnected from the policy network, making the gradients nonsensical. This isn't the case for Gumbel-softmax.
    
    - Change exploration to randomly compute probabilities instead of a single action.
    - Added a fixed policy_temperature and an annealed 'exploration_temperature', mimicing original TD3 behavior.
    """

    # python td3_gumbel_softmax.py --mazepath ./mazes/width-7_height-7_complexity-0.9_density-0.9_HaZkfFSrmPtakGDH.pkl
    # python td3_gumbel_softmax.py --device 1 --dump --mazepath width-25_height-25_complexity-0.9_density-0.9_MtU6Wf2sgAp2Aq9m.pkl --maze_view_kernel_size