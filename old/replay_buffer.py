import numpy as np
import torch
import torch.nn.functional as F

"""
November 30, 2023 
Changelog:
    - Separate buffer to be used by all agents
    - Buffer keys the entries as (byte form of state, selected_action) tuples, input is still (state, action, reward, next_state, done)
    - Removed the buffer size limit
    - Updating same-key observations instead of ignoring them
"""

def populate_with_random_observation(args, env, replay_buffer, tries=1):
    # Populate with random observations from entire maze
    flag = False
    for _ in range(tries):
        if flag == True:
            break

        new_state, new_action_logits, new_reward, new_next_state, new_terminated = env.get_random_observation()
        
        if args.double:
            new_action_probs = F.softmax(torch.from_numpy(new_action_logits), dim=-1).double().to(args.device)
        else:
            new_action_probs = F.softmax(torch.from_numpy(new_action_logits), dim=-1).float().to(args.device)

        flag = replay_buffer.add(new_state, new_action_probs, new_reward, new_next_state, new_terminated)

class ReplayBuffer:
    r"""Experience buffer based on history of observations.
    
    Observations are keyed dictionaries (k,v) = (state, action), (state,action,reward,next_state,done).
    
    Note that keys are byte forms of numpy arrays to allow hashing. Actions are converted to numpy arrays before hashing.
    
    This buffer has no size limit as it's expected to hold the entire maze in memory."""
    def __init__(self):
        self.buffer = {}

    def add(self, state, action, reward, next_state, done):
        r"""Add an observation to the buffer.
        
        Args:
            state (np.array): The current state
            action (torch.tensor): The action taken
            reward (float): The reward received
            next_state (np.array): The next state
            done (bool): Whether the episode is done

        Returns:
            flag (bool): Whether a new observation is added to the buffer. 
                         Note that replacing an existing one returns False.
        """
        key = (state.tobytes(), torch.argmax(action).item())
        if key in self.buffer:
            self.buffer[key] = [state, action.cpu().numpy(), reward, next_state, done]
            return False
        else:
            self.buffer[key] = [state, action.cpu().numpy(), reward, next_state, done]
            return True

    def sample(self, batch_size, double, device):
        r"""Sample a batch of observations from the buffer.
        
        Args:
            batch_size (int): The size of the batch to sample
            double (bool): Whether to use double precision
            device (torch.device): The device to move the tensors to.

        Returns:
            states (torch.tensor): The current states
            actions (torch.tensor): The actions taken
            rewards (torch.tensor): The rewards received
            next_states (torch.tensor): The next states
            dones (torch.tensor): Whether the episode is done
        """
        indices = np.random.choice(len(self.buffer.keys()), min(len(self.buffer), batch_size), replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[key] for key in [list(self.buffer.keys())[i] for i in indices]])

        states = torch.from_numpy(np.stack(states, axis=0))
        actions = torch.from_numpy(np.stack(actions, axis=0)).squeeze(1)
        rewards = torch.FloatTensor(rewards).view(-1, 1)
        next_states = torch.from_numpy(np.stack(next_states, axis=0))
        dones = torch.Tensor(dones).view(-1, 1)

        # Convert to PyTorch tensors
        if double:
            states = states.double()
            actions = actions.long()
            rewards = rewards.double()
            next_states = next_states.double()
            dones = torch.Tensor(dones).view(-1, 1)

        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        next_states = next_states.to(device)
        dones = dones.to(device)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)