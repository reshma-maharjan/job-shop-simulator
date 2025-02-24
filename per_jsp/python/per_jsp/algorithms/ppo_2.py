import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import namedtuple, deque
import logging
from typing import List, Tuple, Optional
import random
import math
from dataclasses import dataclass
from per_jsp.python.per_jsp.algorithms.base import BaseScheduler
from per_jsp.python.per_jsp.environment.job_shop_environment import JobShopEnvironment, Action

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done', 'action_log_prob'])

class PrioritizedReplayBuffer:
    def __init__(self, max_size=10000, alpha=0.6):
        self.max_size = max_size
        self.alpha = alpha
        self.buffer = deque(maxlen=max_size)
        self.priorities = deque(maxlen=max_size)
        
    def add(self, experience: Experience, priority=None):
        if priority is None:
            priority = max(self.priorities) if self.priorities else 1.0
        
        self.buffer.append(experience)
        self.priorities.append(priority)
        
        if len(self.buffer) > self.max_size:
            self.buffer.popleft()
            self.priorities.popleft()
    
    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return []
        
        probs = np.array(self.priorities) ** self.alpha
        probs = probs / probs.sum()
        
        indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        return samples
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            if idx < len(self.priorities):
                self.priorities[idx] = priority

class AdaptiveActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        
        # Scale hidden dimensions based on problem size
        self.hidden_dim = min(1024, max(256, state_dim * 2))
        
        # Adaptive network depth based on problem size
        num_layers = max(2, min(4, state_dim // 64))
        
        # Build shared layers dynamically
        shared_layers = []
        current_dim = state_dim
        
        for i in range(num_layers):
            out_dim = self.hidden_dim // (2 ** i)
            shared_layers.extend([
                nn.Linear(current_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            current_dim = out_dim
        
        self.shared = nn.Sequential(*shared_layers)
        
        # Adaptive actor and critic networks
        actor_dim = max(64, self.hidden_dim // 4)
        self.actor = nn.Sequential(
            nn.Linear(current_dim, actor_dim),
            nn.ReLU(),
            nn.Linear(actor_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        critic_dim = max(64, self.hidden_dim // 4)
        self.critic = nn.Sequential(
            nn.Linear(current_dim, critic_dim),
            nn.ReLU(),
            nn.Linear(critic_dim, 1)
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
    def __init__(self, problem_size: int):
        self.max_size = max(1000, min(50000, problem_size * 100))
        self.states = deque(maxlen=self.max_size)
        self.actions = deque(maxlen=self.max_size)
        self.rewards = deque(maxlen=self.max_size)
        self.values = deque(maxlen=self.max_size)
        self.log_probs = deque(maxlen=self.max_size)
        self.dones = deque(maxlen=self.max_size)
        self.advantages = None
        self.returns = None
        
        # Dynamic batch size based on problem size
        self.batch_size = min(256, max(32, problem_size * 2))
        
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
        self.advantages = None
        self.returns = None
        
    def compute_advantages(self, gamma: float, gae_lambda: float, last_value: float):
        # Convert deques to numpy arrays properly
        rewards = np.array(list(self.rewards), dtype=np.float32)
        values_list = list(self.values)
        values_list.append(last_value)
        values = np.array(values_list, dtype=np.float32)
        dones_list = list(self.dones)
        dones_list.append(0)
        dones = np.array(dones_list, dtype=np.float32)
        
        # Initialize advantages array
        self.advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0
        
        # Compute GAE with numerical stability
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
            self.advantages[t] = gae
            
        # Compute returns and normalize advantages
        self.returns = self.advantages + np.array(list(self.values), dtype=np.float32)
        
        # Normalize with numerical stability
        adv_mean = np.mean(self.advantages)
        adv_std = np.std(self.advantages) + 1e-8
        self.advantages = (self.advantages - adv_mean) / adv_std
        
    def get_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
            
        batch_size = min(batch_size, len(self.states))
        indices = np.random.permutation(len(self.states))
        
        for start_idx in range(0, len(self.states), batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            yield (
                np.array([self.states[i] for i in batch_indices]),
                np.array([self.actions[i] for i in batch_indices]),
                self.advantages[batch_indices],
                self.returns[batch_indices],
                np.array([self.log_probs[i] for i in batch_indices])
            )


class PPOScheduler(BaseScheduler):
    def __init__(self, 
                 learning_rate: float = 3e-4,
                 epochs: int = 10,
                 batch_size: int = 64,
                 clip_range: float = 0.2,
                 value_loss_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5,
                 episodes: int = 500,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95):
        
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
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_makespan = float('inf')
        self.best_actions = None
        self.memory = PPOMemory(batch_size)
        
    def _extract_state_features(self, env: JobShopEnvironment) -> np.ndarray:
        """Enhanced state features extraction with better scaling."""
        features = []
        
        # Job completion status (normalized)
        features.extend(env.current_state.completed_jobs.astype(float))
        
        # Machine utilization (normalized)
        machine_util = np.array(env.get_machine_utilization())
        features.extend(machine_util)
        
        # Job progress (normalized)
        max_ops = max(len(job.operations) for job in env.jobs)
        for job_idx in range(len(env.jobs)):
            progress = env.current_state.next_operation_for_job[job_idx] / max_ops
            features.append(progress)
        
        # Machine availability (normalized)
        current_time = max(1, env.total_time)
        machine_avail = env.current_state.machine_availability / current_time
        features.extend(machine_avail)
        
        # Critical path information (binary)
        critical_path = set(env.get_critical_path())
        for job_idx in range(len(env.jobs)):
            is_critical = any((job_idx, op_idx) in critical_path 
                            for op_idx in range(len(env.jobs[job_idx].operations)))
            features.append(float(is_critical))
        
        # Dependency satisfaction (binary)
        for job_idx in range(len(env.jobs)):
            features.append(float(env.is_job_executable(job_idx)))
            
        # Add problem size-aware normalization
        features = np.array(features, dtype=np.float32)
        if len(features) > 0:
            features = (features - features.mean()) / (features.std() + 1e-8)
        
        return features

    def _calculate_reward(self, prev_makespan: int, current_makespan: int, env: JobShopEnvironment) -> float:
        """Enhanced reward calculation with problem size scaling."""
        # Calculate problem size for scaling
        problem_size = len(env.jobs) * max(len(job.operations) for job in env.jobs)
        scale_factor = math.sqrt(problem_size)
        
        # Scaled makespan improvement
        makespan_diff = (current_makespan - prev_makespan) / max(1, prev_makespan)
        makespan_reward = -makespan_diff * scale_factor
        
        # Scaled progress reward
        total_ops = sum(len(job.operations) for job in env.jobs)
        completed_ops = sum(env.current_state.next_operation_for_job)
        progress_reward = (completed_ops / total_ops) * scale_factor
        
        # Machine utilization with diminishing returns
        utilization = np.mean(env.get_machine_utilization())
        utilization_reward = math.log1p(utilization * 100) / 5
        
        # Critical path optimization
        critical_path = env.get_critical_path()
        critical_path_length = len(critical_path)
        critical_path_reward = -critical_path_length / total_ops
        
        # Completion bonus scaled by problem size
        completion_bonus = 10.0 * math.sqrt(problem_size) if env.is_done() else 0.0
        
        # Combine rewards with dynamic weights
        reward = (
            makespan_reward * 2.0 +
            progress_reward * 1.5 +
            utilization_reward * 1.0 +
            critical_path_reward * 1.0 +
            completion_bonus
        )
        
        return float(reward)

    def solve(self, env: JobShopEnvironment, max_steps: int = 1000) -> Tuple[List[Action], int]:
        # Initialize with problem-size aware dimensions
        state_features = self._extract_state_features(env)
        state_dim = len(state_features)
        action_dim = max(len(env.get_possible_actions()), 1)
        
        # Create networks with adaptive architecture
        self.policy = AdaptiveActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        
        # Problem size-aware memory and replay buffer
        problem_size = len(env.jobs) * max(len(job.operations) for job in env.jobs)
        self.memory = PPOMemory(problem_size)
        self.replay_buffer = PrioritizedReplayBuffer(
            max_size=max(1000, min(50000, problem_size * 100))
        )
        
        # Learning rate scheduler
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
                    masked_probs[0, :len(valid_actions)] = 1.0 / len(valid_actions)

                # Dynamic exploration rate based on problem size
                epsilon = max(0.01, min(0.3, 1.0 - episode / (self.episodes / math.sqrt(problem_size))))
                if random.random() < epsilon:
                    action_idx = random.randrange(len(valid_actions))
                else:
                    # Temperature scaling for exploration
                    temperature = max(0.5, min(2.0, problem_size / 100))
                    scaled_probs = masked_probs ** (1 / temperature)
                    scaled_probs = scaled_probs / scaled_probs.sum()
                    dist = torch.distributions.Categorical(scaled_probs)
                    action_idx = dist.sample().item()

                chosen_action = valid_actions[action_idx % len(valid_actions)]
                
                prev_makespan = env.total_time
                env.step(chosen_action)
                next_state = self._extract_state_features(env)
                done = env.is_done()
                
                reward = self._calculate_reward(prev_makespan, env.total_time, env)
                episode_reward += reward
                
                # Store experience in memory and replay buffer
                self.memory.store(
                    state,
                    action_idx % action_dim,
                    reward,
                    value.item(),
                    masked_probs[0, action_idx % action_dim].item(),
                    done
                )
                
                # Store in replay buffer with priority based on reward magnitude
                priority = abs(reward) + 1e-6
                self.replay_buffer.add(
                    Experience(state, action_idx, reward, next_state, done, 
                             masked_probs[0, action_idx % action_dim].item()),
                    priority
                )
                
                state = next_state
                episode_steps += 1
            
            # Update policy with more frequent updates for larger problems
            update_frequency = max(1, min(5, int(10 / math.sqrt(problem_size))))
            if episode % update_frequency == 0:
                with torch.no_grad():
                    _, last_value = self.policy(
                        torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    )
                    last_value = last_value.item()
                
                self.memory.compute_advantages(self.gamma, self.gae_lambda, last_value)
                self._update_policy(problem_size)
                self.memory.clear()
            
            # Update best solution
            if env.total_time < self.best_makespan:
                self.best_makespan = env.total_time
                self.best_actions = env.action_history.copy()
                self.best_state = {
                    'actions': env.action_history.copy(),
                    'makespan': env.total_time,
                    'machine_schedule': env.get_schedule_data(),
                    'job_progress': env.current_state.job_progress.copy(),
                    'machine_availability': env.current_state.machine_availability.copy()
                }
                logger.info(f"New best makespan: {self.best_makespan}")
            
            # Learning rate scheduling
            episode_rewards.append(episode_reward)
            window_size = min(100, max(10, int(self.episodes / 10)))
            avg_reward = np.mean(episode_rewards[-window_size:])
            self.lr_scheduler.step(-avg_reward)  # Negative because we want to maximize reward
            
            if episode % max(1, self.episodes // 20) == 0:
                logger.info(f"Episode {episode}/{self.episodes}, "
                          f"Reward: {episode_reward:.2f}, "
                          f"Avg Reward: {avg_reward:.2f}, "
                          f"Best makespan: {self.best_makespan}")
        
        # Reset to best solution at the end
        if hasattr(self, 'best_state'):
            env.reset()
            for action in self.best_state['actions']:
                env.step(action)
        
        return self.best_actions, self.best_makespan

    def _update_policy(self, problem_size: int):
        """Enhanced policy update with adaptive parameters."""
        # Scale number of epochs with problem size
        n_epochs = max(3, min(self.epochs, int(self.epochs / math.sqrt(problem_size))))
        
        # Adaptive batch size
        batch_size = self.memory.batch_size
        
        for _ in range(n_epochs):
            total_policy_loss = 0
            total_value_loss = 0
            total_entropy = 0
            n_batches = 0
            
            for states, actions, advantages, returns, old_log_probs in self.memory.get_batch(batch_size):
                states = torch.FloatTensor(states).to(self.device)
                actions = torch.LongTensor(actions).to(self.device)
                advantages = torch.FloatTensor(advantages).to(self.device)
                returns = torch.FloatTensor(returns).to(self.device)
                old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
                
                # Get current policy outputs
                action_probs, values = self.policy(states)
                
                # Ensure action indices are valid
                actions = torch.clamp(actions, 0, action_probs.size(-1) - 1)
                
                # Calculate policy loss with entropy bonus
                dist = torch.distributions.Categorical(action_probs)
                curr_log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()
                
                # Calculate ratios and surrogate objectives
                ratios = torch.exp(curr_log_probs - old_log_probs)
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.clip_range, 1+self.clip_range) * advantages
                
                # Calculate losses
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * (returns - values.squeeze()).pow(2).mean()
                entropy_loss = -self.entropy_coef * entropy
                
                # Combine losses with adaptive coefficients
                total_loss = (
                    policy_loss + 
                    self.value_loss_coef * value_loss + 
                    entropy_loss
                )
                
                # Optimize
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # Adaptive gradient clipping
                max_grad = max(0.1, self.max_grad_norm / math.sqrt(problem_size))
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_grad)
                
                self.optimizer.step()
                
                # Track metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                n_batches += 1
                
                # Early stopping check
                with torch.no_grad():
                    kl_div = ((old_log_probs - curr_log_probs) ** 2).mean()
                    if kl_div > 0.015:
                        break
            
            # Log average losses if in debug mode
            if logger.getEffectiveLevel() <= logging.DEBUG:
                logger.debug(f"Average losses - Policy: {total_policy_loss/n_batches:.4f}, "
                           f"Value: {total_value_loss/n_batches:.4f}, "
                           f"Entropy: {total_entropy/n_batches:.4f}")

    def save_model(self, filename: str):
        """Save the trained model and optimization state."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            'best_makespan': self.best_makespan,
            'hyperparameters': {
                'learning_rate': self.lr,
                'gamma': self.gamma,
                'gae_lambda': self.gae_lambda,
                'clip_range': self.clip_range,
                'value_loss_coef': self.value_loss_coef,
                'entropy_coef': self.entropy_coef
            }
        }, filename)

    def load_model(self, filename: str, env: JobShopEnvironment):
        """Load a trained model with environment-specific adaptation."""
        checkpoint = torch.load(filename)
        
        # Initialize policy with current environment dimensions
        state_features = self._extract_state_features(env)
        state_dim = len(state_features)
        action_dim = max(len(env.get_possible_actions()), 1)
        
        self.policy = AdaptiveActorCritic(state_dim, action_dim).to(self.device)
        
        # Load states
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['scheduler_state_dict'] and self.lr_scheduler:
            self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_makespan = checkpoint['best_makespan']
        
        # Update hyperparameters
        for key, value in checkpoint['hyperparameters'].items():
            setattr(self, key, value)

# Example usage
if __name__ == "__main__":
    # Example of creating a simple job shop problem
    job1 = Job(operations=[
        Operation(duration=3, machine=0),
        Operation(duration=2, machine=1)
    ])

    job2 = Job(operations=[
        Operation(duration=2, machine=1),
        Operation(duration=4, machine=0)
    ])
    
    # Create environment
    env = JobShopEnvironment([job1, job2])
    
    # Create and run scheduler
    scheduler = PPOScheduler(
        learning_rate=3e-4,
        episodes=500,
        entropy_coef=0.01
    )
    
    # Solve the problem
    best_actions, best_makespan = scheduler.solve(env)
    
    # Print results
    print(f"Best makespan: {best_makespan}")
    print("Schedule:")
    env.print_schedule(show_critical_path=True)
    
    # # Save the model
    # scheduler.save_model("trained_scheduler.pt")action_probs[0, :len(valid_actions)]
    #                 if masked_probs.sum() > 0:
    #                     masked_probs = masked_probs / masked_probs.sum()
    #                 else:
    #                     masked_probs[0, :len(valid_actions)] = 1.0 / len(valid_actions)
    #                 dist = torch.distributions.Categorical(masked_probs)