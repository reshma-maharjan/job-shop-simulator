import numpy as np
import random


import numpy as np
import random


class ReplayBuffer:
    """
    Replay buffer specifically designed for Job Shop Scheduling problems.
    Handles storage and sampling of experiences with flattened observations.
    """
    def __init__(self, buffer_size, batch_size, n_steps=1):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.n_steps = n_steps  # Number of steps for n-step returns
        
        # Initialize buffers
        self.actions = []
        self.cur_obs = []
        self.next_obs = []
        self.rewards = []
        self.terminated = []
        self.truncated = []
        
        # Initialize sample buffers
        self.sampled_actions = []
        self.sampled_cur_obs = []
        self.sampled_next_obs = []
        self.sampled_rewards = []
        self.sampled_terminated = []
        self.sampled_trunc = []
        
        # Position tracker
        self.position = 0
        self.buffer_filled = False
        
    def save_experience(self, action, cur_obs, next_obs, reward, terminated, truncated):
        """
        Save an experience to the buffer
        
        Args:
            action: The action taken (job index)
            cur_obs: The current observation (flattened)
            next_obs: The next observation (flattened)
            reward: The reward received
            terminated: Whether the episode is terminated
            truncated: Whether the episode is truncated
        """
        if len(self.actions) < self.buffer_size:
            # Buffer not yet filled
            self.actions.append(action)
            self.cur_obs.append(cur_obs)
            self.next_obs.append(next_obs)
            self.rewards.append(reward)
            self.terminated.append(terminated)
            self.truncated.append(truncated)
        else:
            # Buffer full, replace old experiences
            self.actions[self.position] = action
            self.cur_obs[self.position] = cur_obs
            self.next_obs[self.position] = next_obs
            self.rewards[self.position] = reward
            self.terminated[self.position] = terminated
            self.truncated[self.position] = truncated
            
            # Update position
            self.position = (self.position + 1) % self.buffer_size
            self.buffer_filled = True
            
    def sample(self):
        """
        Sample a batch of experiences from the buffer
        
        Returns:
            Batch of experiences stored in class attributes
        """
        # Determine valid indices for sampling
        max_idx = len(self.actions) if not self.buffer_filled else self.buffer_size
        
        if max_idx <= self.batch_size:
            # Not enough samples, use all available indices
            indices = np.arange(max_idx)
        else:
            # Sample random indices
            indices = np.random.choice(max_idx, self.batch_size, replace=False)
        
        # Clear previous samples
        self.sampled_actions = []
        self.sampled_cur_obs = []
        self.sampled_next_obs = []
        self.sampled_rewards = []
        self.sampled_terminated = []
        self.sampled_trunc = []
        
        # Sample new experiences
        for idx in indices:
            self.sampled_actions.append(self.actions[idx])
            self.sampled_cur_obs.append(self.cur_obs[idx])
            self.sampled_next_obs.append(self.next_obs[idx])
            self.sampled_rewards.append(self.rewards[idx])
            self.sampled_terminated.append(self.terminated[idx])
            self.sampled_trunc.append(self.truncated[idx])
            
    def sample_n_seq(self):
        """
        Sample a batch of n-step sequences from the buffer
        
        Returns:
            Batch of n-step sequences stored in class attributes
        """
        if self.n_steps <= 1:
            self.sample()
            return
        
        # Ensure we have enough samples
        max_idx = len(self.actions) - self.n_steps + 1
        
        if max_idx <= 0:
            # Not enough samples for n-step
            if len(self.actions) > 0:
                print(f"Warning: Not enough samples for {self.n_steps}-step returns. Using 1-step returns instead.")
                self.sample()
            return
        
        # Determine valid indices for sampling (need n consecutive experiences)
        if self.buffer_filled:
            # Need to be careful with buffer wrapping
            valid_indices = []
            for i in range(self.buffer_size):
                # Check if we can get n consecutive samples starting from i
                is_valid = True
                if (self.position > i and self.position < i + self.n_steps):
                    # Would wrap around buffer boundary
                    is_valid = False
                valid_indices.append(i) if is_valid else None
                
            if len(valid_indices) < self.batch_size:
                # Not enough valid indices
                indices = np.random.choice(valid_indices, len(valid_indices), replace=False)
            else:
                indices = np.random.choice(valid_indices, self.batch_size, replace=False)
        else:
            # No wrapping concerns
            if max_idx <= self.batch_size:
                indices = np.arange(max_idx)
            else:
                indices = np.random.choice(max_idx, self.batch_size, replace=False)
        
        # Clear previous samples
        self.sampled_actions = []
        self.sampled_cur_obs = []
        self.sampled_next_obs = []
        self.sampled_rewards = []
        self.sampled_terminated = []
        self.sampled_trunc = []
        
        # Sample new n-step sequences
        for idx in indices:
            # Store initial action and observation
            actions_seq = [self.actions[idx]]
            cur_obs_seq = [self.cur_obs[idx]]
            
            # Store rewards, terminated flags, and truncated flags for all n steps
            rewards_seq = []
            terminated_seq = []
            truncated_seq = []
            
            for i in range(self.n_steps):
                step_idx = (idx + i) % self.buffer_size
                rewards_seq.append(self.rewards[step_idx])
                terminated_seq.append(self.terminated[step_idx])
                truncated_seq.append(self.truncated[step_idx])
                
                # Store the last next_obs
                if i == self.n_steps - 1:
                    next_obs_seq = [self.next_obs[step_idx]]
            
            # Add the sampled sequence to the batch
            self.sampled_actions.append(actions_seq)
            self.sampled_cur_obs.append(cur_obs_seq)
            self.sampled_next_obs.append(next_obs_seq)
            self.sampled_rewards.append(rewards_seq)
            self.sampled_terminated.append(terminated_seq)
            self.sampled_trunc.append(truncated_seq)


if __name__ == '__main__':
    pass
