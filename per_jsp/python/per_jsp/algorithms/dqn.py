import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple
from collections import deque
import random
import logging
from per_jsp.python.per_jsp.algorithms.base import BaseScheduler
from per_jsp.python.per_jsp.environment.job_shop_environment import JobShopEnvironment, Action

logger = logging.getLogger(__name__)

class DQNNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    m.bias.data.zero_()
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class DQNScheduler(BaseScheduler):
    def __init__(
        self,
        learning_rate: float = 0.001,
        discount_factor: float = 0.99,
        exploration_rate: float = 1.0,
        episodes: int = 1000,
        hidden_size: int = 128,
        buffer_size: int = 10000,
        batch_size: int = 32,
        target_update: int = 10
    ):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.episodes = episodes
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.target_update = target_update
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = None
        self.target_net = None
        self.optimizer = None
        self.memory = ReplayBuffer(buffer_size)
        
        self.best_time = float('inf')
        self.best_schedule = []

    def _get_state_size(self, env: JobShopEnvironment) -> int:
        """Calculate the state vector size."""
        num_jobs = len(env.jobs)
        max_operations = max(len(job.operations) for job in env.jobs)
        
        return (
            num_jobs * max_operations +  # job_progress
            env.num_machines +          # machine_availability
            num_jobs +                 # next_operation_for_job
            num_jobs +                 # completed_jobs
            num_jobs                   # job_start_times
        )

    def _get_state_vector(self, env: JobShopEnvironment) -> np.ndarray:
        """Convert environment state to vector representation."""
        # Get job progress features
        job_progress = env.current_state.job_progress.flatten()
        
        # Get machine availability
        machine_availability = env.current_state.machine_availability
        
        # Get next operation for each job
        next_operations = env.current_state.next_operation_for_job
        
        # Get completed jobs status
        completed_jobs = env.current_state.completed_jobs
        
        # Get job start times (normalized)
        job_start_times = env.current_state.job_start_times.copy()
        job_start_times[job_start_times == -1] = 0  # Replace -1 with 0 for non-started jobs
        
        # Concatenate all features
        state = np.concatenate([
            job_progress,
            machine_availability,
            next_operations,
            completed_jobs.astype(np.float32),
            job_start_times
        ])
        
        return self._normalize_state(state)

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize state values to improve training stability."""
        state = np.array(state, dtype=np.float32)
        if state.std() > 0:
            return (state - state.mean()) / (state.std() + 1e-8)
        return state

    def _initialize_networks(self, env: JobShopEnvironment) -> None:
        """Initialize neural networks."""
        state_size = self._get_state_size(env)
        action_size = env.num_machines * len(env.jobs)
        
        self.policy_net = DQNNetwork(state_size, action_size, self.hidden_size).to(self.device)
        self.target_net = DQNNetwork(state_size, action_size, self.hidden_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

    def _select_action(self, env: JobShopEnvironment, state: np.ndarray) -> Action:
        """Select action using epsilon-greedy strategy."""
        possible_actions = env.get_possible_actions()
        
        if not possible_actions:
            return None 
            
        if random.random() < self.exploration_rate:
            return random.choice(possible_actions)
            
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            
        # Store actions and their corresponding Q-values in a list of tuples
        action_values = []
        for action in possible_actions:
            # Create unique index for each job-machine combination
            action_idx = action.job * env.num_machines + action.machine
            action_values.append((action, q_values[0][action_idx].item()))
            
        # Return the action with the highest Q-value
        return max(action_values, key=lambda x: x[1])[0]

    def _optimize_model(self) -> None:
        """Perform one step of optimization."""
        if len(self.memory) < self.batch_size:
            return
            
        transitions = self.memory.sample(self.batch_size)
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)
        
        state_batch = torch.FloatTensor(batch_state).to(self.device)
        action_batch = torch.LongTensor(batch_action).to(self.device)
        reward_batch = torch.FloatTensor(batch_reward).to(self.device)
        next_state_batch = torch.FloatTensor(batch_next_state).to(self.device)
        done_batch = torch.FloatTensor(batch_done).to(self.device)
        
        # Compute current Q values
        current_q_values = self.policy_net(state_batch)
        current_q_values = current_q_values.gather(1, action_batch.unsqueeze(1))
        
        # Compute next Q values
        next_q_values = self.target_net(next_state_batch).max(1)[0].detach()
        expected_q_values = reward_batch + (1 - done_batch) * self.discount_factor * next_q_values
        
        # Compute loss and optimize
        loss = F.smooth_l1_loss(current_q_values, expected_q_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

    def _run_episode(self, env: JobShopEnvironment, max_steps: int = 1000) -> List[Action]:
        """Run a single episode."""
        env.reset()
        episode_actions = []
        state = self._get_state_vector(env)
        
        step = 0
        while not env.is_done() and step < max_steps:
            action = self._select_action(env, state)
            if action is None:
                break
                
            # Convert action to index for replay buffer
            action_idx = action.job * env.num_machines + action.machine
            
            # Get state before taking action
            prev_total_time = env.total_time
            
            # Take action
            env.step(action)
            episode_actions.append(action)
            
            # Get new state
            next_state = self._get_state_vector(env)
            
            # Calculate reward components
            time_penalty = -(env.total_time - prev_total_time)
            utilization_bonus = np.mean(env.get_machine_utilization()) * 100
            completion_bonus = 1000 if env.is_done() else 0
            
            # Combined reward
            reward = time_penalty + utilization_bonus + completion_bonus
            
            # Store transition
            self.memory.push(
                state, action_idx, reward, next_state, env.is_done()
            )
            
            # Move to next state
            state = next_state
            
            # Optimize model
            self._optimize_model()
            
            step += 1
            
        return episode_actions

    def solve(self, env: JobShopEnvironment, max_steps: int = 1000) -> Tuple[List[Action], int]:
        """Solve using DQN."""
        # Initialize networks if not already done
        if self.policy_net is None:
            self._initialize_networks(env)
            
        logger.info(f"Starting DQN training for {self.episodes} episodes...")
        
        for episode in range(self.episodes):
            # Run episode
            episode_actions = self._run_episode(env, max_steps)
            
            # Evaluate episode
            env.reset()
            for action in episode_actions:
                env.step(action)
                
            # Track best solution
            if env.total_time < self.best_time:
                self.best_time = env.total_time
                self.best_schedule = episode_actions.copy()
                logger.info(f"New best makespan: {self.best_time}")
                
            # Update target network
            if episode % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
                
            # Decay exploration rate
            self.exploration_rate = max(0.01, self.exploration_rate * 0.995)
            
            if (episode + 1) % 10 == 0:
                logger.info(f"Episode {episode + 1}/{self.episodes}, "
                          f"Best makespan: {self.best_time}")
        
        # Final run with best actions
        env.reset()
        for action in self.best_schedule:
            env.step(action)
            
        return self.best_schedule, env.total_time