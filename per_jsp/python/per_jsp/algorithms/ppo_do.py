import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import namedtuple
import logging
from typing import List, Tuple, Optional
import random
from dataclasses import dataclass
from per_jsp.python.per_jsp.algorithms.base import BaseScheduler
from per_jsp.python.per_jsp.environment.job_shop_environment import JobShopEnvironment, Action
from per_jsp.python.per_jsp.environment.job_shop_taillard_generator import TaillardJobShopGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done', 'action_log_prob'])

class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # Increased network capacity with layer normalization instead of batch normalization
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2),
            nn.ReLU()
        )
        
        # Separate actor and critic networks for better specialization
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.ReLU(),
            nn.Linear(hidden_dim//4, action_dim),
            nn.Softmax(dim=-1)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.ReLU(),
            nn.Linear(hidden_dim//4, 1)
        )
        
        # Improved initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(state)
        action_probs = self.actor(features)
        value = self.critic(features)
        return action_probs, value

class PPOMemory:
    def __init__(self, batch_size=64):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.advantages = None  # Will be numpy array
        self.returns = None     # Will be numpy array
        self.batch_size = batch_size
        
    def store(self, state, action, reward, value, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
        self.advantages = None  # Reset numpy arrays to None
        self.returns = None
        
    def compute_advantages(self, gamma: float, gae_lambda: float, last_value: float):
        # Convert lists to numpy arrays
        rewards = np.array(self.rewards, dtype=np.float32)
        values = np.array(self.values + [last_value], dtype=np.float32)
        dones = np.array(self.dones + [0], dtype=np.float32)
        
        # Initialize advantages array
        self.advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0
        
        # Compute GAE
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
            self.advantages[t] = gae
            
        # Compute returns
        self.returns = self.advantages + np.array(self.values, dtype=np.float32)
        
        # Normalize advantages
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)
        
    def get_batch(self):
        batch_size = min(self.batch_size, len(self.states))
        indices = np.random.permutation(len(self.states))
        
        for start_idx in range(0, len(self.states), batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            yield (
                np.array(self.states)[batch_indices],
                np.array(self.actions)[batch_indices],
                np.array(self.advantages)[batch_indices],
                np.array(self.returns)[batch_indices],
                np.array(self.log_probs)[batch_indices]
            )

class PPOScheduler(BaseScheduler):
    def __init__(self, 
                 learning_rate: float = 3e-4,  # Adjusted learning rate
                 epochs: int = 10,
                 batch_size: int = 64,
                 clip_range: float = 0.2,
                 value_loss_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5,
                 episodes: int = 500,  # Increased episodes
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 hidden_dim: int = 256):  # Increased hidden dimension
        
        super().__init__()
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.clip_range = clip_range
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.episodes = episodes
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.hidden_dim = hidden_dim
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_makespan = float('inf')
        self.best_actions = None
        self.memory = PPOMemory(batch_size)
        
        # Add learning rate scheduler
        self.lr_scheduler = None
        
        # Add experience replay buffer
        self.replay_buffer = []
        self.replay_buffer_size = 10000
        
    def _extract_state_features(self, env: JobShopEnvironment) -> np.ndarray:
        """Enhanced state features extraction."""
        features = []
        
        # 1. Job completion status (normalized)
        features.extend(env.current_state.completed_jobs.astype(float))
        
        # 2. Machine utilization
        machine_util = env.get_machine_utilization()
        features.extend(machine_util)
        
        # 3. Job progress (normalized)
        max_ops = max(len(job.operations) for job in env.jobs)
        for job_idx in range(len(env.jobs)):
            progress = env.current_state.next_operation_for_job[job_idx] / max_ops
            features.append(progress)
        
        # 4. Machine availability (normalized)
        current_time = max(1, env.total_time)
        machine_avail = env.current_state.machine_availability / current_time
        features.extend(machine_avail)
        
        # 5. Critical path information
        critical_path = set(env.get_critical_path())
        for job_idx in range(len(env.jobs)):
            is_critical = any((job_idx, op_idx) in critical_path 
                            for op_idx in range(len(env.jobs[job_idx].operations)))
            features.append(float(is_critical))
        
        # 6. Dependency satisfaction
        for job_idx in range(len(env.jobs)):
            features.append(float(env.is_job_executable(job_idx)))
        
        return np.array(features, dtype=np.float32)

    def _calculate_reward(self, prev_makespan: int, current_makespan: int, env: JobShopEnvironment) -> float:
        """Enhanced reward calculation with multiple objectives."""
        # 1. Makespan improvement (normalized)
        makespan_diff = (current_makespan - prev_makespan) / max(1, prev_makespan)
        makespan_reward = -makespan_diff
        
        # 2. Progress reward
        total_ops = sum(len(job.operations) for job in env.jobs)
        completed_ops = sum(env.current_state.next_operation_for_job)
        progress_reward = completed_ops / total_ops
        
        # 3. Machine utilization reward
        utilization = np.mean(env.get_machine_utilization())
        utilization_reward = utilization
        
        # 4. Critical path optimization
        critical_path = env.get_critical_path()
        critical_path_length = len(critical_path)
        critical_path_reward = -critical_path_length / total_ops
        
        # 5. Completion bonus
        completion_bonus = 5.0 if env.is_done() else 0.0
        
        # Weighted combination of rewards
        reward = (
            makespan_reward * 2.0 +
            progress_reward * 1.5 +
            utilization_reward * 1.0 +
            critical_path_reward * 1.0 +
            completion_bonus
        )
        
        return float(reward)

    def solve(self, env: JobShopEnvironment, max_steps: int = 1000) -> Tuple[List[Action], int]:
        # Initialize with correct action dimension
        state_features = self._extract_state_features(env)
        state_dim = len(state_features)
        # Set action_dim to maximum possible number of actions
        action_dim = max(len(env.get_possible_actions()), 1)  # Ensure at least 1
        
        self.policy = ActorCritic(state_dim, action_dim, self.hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=20, verbose=True
        )
        
        episode_rewards = []
        best_episode_reward = float('-inf')
        
        for episode in range(1, self.episodes + 1):
            env.reset()
            state = self._extract_state_features(env)
            done = False
            episode_reward = 0
            episode_steps = 0
            
            while not done and episode_steps < max_steps:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                state_tensor = torch.nan_to_num(state_tensor, nan=0.0, posinf=1.0, neginf=-1.0)
                
                valid_actions = env.get_possible_actions()
                if not valid_actions:
                    break
                
                with torch.no_grad():
                    action_probs, value = self.policy(state_tensor)
                    # Mask invalid actions
                    masked_probs = torch.zeros_like(action_probs)
                    masked_probs[0, :len(valid_actions)] = action_probs[0, :len(valid_actions)]
                    # Renormalize probabilities
                    if masked_probs.sum() > 0:
                        masked_probs = masked_probs / masked_probs.sum()
                    else:
                        masked_probs[0, :len(valid_actions)] = 1.0 / len(valid_actions)
                
                # Epsilon-greedy with decaying epsilon
                epsilon = max(0.01, 1.0 - episode / self.episodes)
                if random.random() < epsilon:
                    action_idx = random.randrange(len(valid_actions))
                else:
                    action_idx = self._select_action(masked_probs)
                
                chosen_action = valid_actions[action_idx % len(valid_actions)]
                
                prev_makespan = env.total_time
                env.step(chosen_action)
                next_state = self._extract_state_features(env)
                done = env.is_done()
                
                reward = self._calculate_reward(prev_makespan, env.total_time, env)
                episode_reward += reward
                
                # Store experience with masked probabilities
                self.memory.store(
                    state,
                    action_idx % action_dim,  # Ensure action index is within bounds
                    reward,
                    value.item(),
                    masked_probs[0, action_idx % action_dim].item(),
                    done
                )
                
                # Store in replay buffer
                self.replay_buffer.append(Experience(
                    state, action_idx, reward, next_state, done, action_probs[0, 0].item()
                ))
                if len(self.replay_buffer) > self.replay_buffer_size:
                    self.replay_buffer.pop(0)
                
                state = next_state
                episode_steps += 1
            
            # Compute advantages and update policy
            if episode % 5 == 0:
                with torch.no_grad():
                    _, last_value = self.policy(
                        torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    )
                    last_value = last_value.item()
                
                self.memory.compute_advantages(self.gamma, self.gae_lambda, last_value)
                self._update_policy()
                self.memory.clear()
            
            # Update best solution with deep copy of entire solution
            if env.total_time < self.best_makespan:
                self.best_makespan = env.total_time
                self.best_actions = env.action_history.copy()
                # Store complete state information
                self.best_state = {
                    'actions': env.action_history.copy(),
                    'makespan': env.total_time,
                    'machine_schedule': env.get_schedule_data(),
                    'job_progress': env.current_state.job_progress.copy(),
                    'machine_availability': env.current_state.machine_availability.copy()
                }
                logger.info(f"New best makespan: {self.best_makespan}")
                
            # At end of episode, if this wasn't the best solution, reset to best known
            if episode == self.episodes:
                if hasattr(self, 'best_state'):
                    env.reset()
                    for action in self.best_state['actions']:
                        env.step(action)
            
            # Learning rate scheduling
            episode_rewards.append(episode_reward)
            avg_reward = np.mean(episode_rewards[-100:])
            self.lr_scheduler.step(avg_reward)
            
            if episode % 10 == 0:
                logger.info(f"Episode {episode}/{self.episodes}, "
                          f"Reward: {episode_reward:.2f}, "
                          f"Best makespan: {self.best_makespan}")
        
        return self.best_actions, self.best_makespan

    def _update_policy(self):
        """Enhanced policy update with clipped PPO and auxiliary objectives."""
        for _ in range(self.epochs):
            for states, actions, advantages, returns, old_log_probs in self.memory.get_batch():
                states = torch.FloatTensor(states).to(self.device)
                actions = torch.LongTensor(actions).to(self.device)
                advantages = torch.FloatTensor(advantages).to(self.device)
                returns = torch.FloatTensor(returns).to(self.device)
                old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
                
                # Get current policy outputs
                action_probs, values = self.policy(states)
                
                # Ensure action indices are valid
                actions = torch.clamp(actions, 0, action_probs.size(-1) - 1)
                
                # Create distribution and get log probs
                dist = torch.distributions.Categorical(action_probs)
                curr_log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()
                
                # Calculate ratios and surrogate objectives
                ratios = torch.exp(curr_log_probs - old_log_probs)
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.clip_range, 1+self.clip_range) * advantages
                
                # Calculate losses with auxiliary objectives
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * (returns - values.squeeze()).pow(2).mean()
                entropy_loss = -self.entropy_coef * entropy
                
                # Total loss
                loss = policy_loss + self.value_loss_coef * value_loss + entropy_loss
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Logging detailed metrics
                with torch.no_grad():
                    approx_kl_div = ((old_log_probs - curr_log_probs) ** 2).mean()
                    clip_fraction = (abs(ratios - 1.0) > self.clip_range).float().mean()
                
                if approx_kl_div > 0.015:
                    # Early stopping if KL divergence is too high
                    break

    def _select_action(self, action_probs: torch.Tensor) -> int:
        """Select action using the policy with improved exploration."""
        # Temperature scaling for exploration
        temperature = max(0.5, 1.0 - len(self.replay_buffer) / self.replay_buffer_size)
        scaled_probs = action_probs ** (1 / temperature)
        scaled_probs = scaled_probs / scaled_probs.sum()
        
        dist = torch.distributions.Categorical(scaled_probs)
        action = dist.sample()
        return action.item()

    def save_model(self, filename: str):
        """Save the trained model."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            'best_makespan': self.best_makespan,
            'hyperparameters': {
                'learning_rate': self.lr,
                'hidden_dim': self.hidden_dim,
                'gamma': self.gamma,
                'gae_lambda': self.gae_lambda,
                'clip_range': self.clip_range,
                'value_loss_coef': self.value_loss_coef,
                'entropy_coef': self.entropy_coef
            }
        }, filename)

    def load_model(self, filename: str):
        """Load a trained model."""
        checkpoint = torch.load(filename)
        
        # Recreate the policy network with saved hyperparameters
        state_dim = next(iter(checkpoint['policy_state_dict'].values())).size(1)
        self.hidden_dim = checkpoint['hyperparameters']['hidden_dim']
        self.policy = ActorCritic(state_dim, 1, self.hidden_dim).to(self.device)
        
        # Load states
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['scheduler_state_dict'] and self.lr_scheduler:
            self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_makespan = checkpoint['best_makespan']
        
        # Update hyperparameters
        for key, value in checkpoint['hyperparameters'].items():
            setattr(self, key, value)