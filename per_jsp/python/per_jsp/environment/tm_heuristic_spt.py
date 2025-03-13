import numpy as np
import os
import yaml
import random
import torch
import tqdm
from  per_jsp.python.per_jsp.algorithms.misc.batch_buffer import Batch 
from jsp_env_gym import  JobShopGym, Job, Operation, MachineSpec # Assuming this is available in your codebase

class TPPOJobShop:
    def __init__(self, env, Policy, config):
        """
        Initialize TPPO algorithm for JobShop environments.
        
        Args:
            env: JobShopGym environment
            Policy: Policy class that contains actor and critic
            config: Configuration dictionary
        """
        self.env = env
        self.action_space_size = env.action_space.n  # Number of jobs (discrete actions)
        
        # For JobShop, the observation space is a Dict, so we need to handle it differently
        self.obs_space_size = self._calculate_obs_space_size(env.observation_space)
        
        config['action_space_size'] = self.action_space_size
        config['obs_space_size'] = self.obs_space_size

        # Print environment space information
        print("\nJobShop Environment Space Information:")
        print(f"Observation Space:")
        print(f"- Structure: Dict with multiple components")
        print(f"- Flattened dimension: {self.obs_space_size}")
        print(f"- Components: {list(env.observation_space.spaces.keys())}")
        
        print(f"\nAction Space:")
        print(f"- Size: {self.action_space_size}")
        print(f"- Type: {type(env.action_space)}")
        print(f"- Available actions: Job indices from 0 to {self.action_space_size - 1}")

        self.policy = Policy(config)
        self.batch = Batch()
        self.config = config

        # Test seeds for consistent evaluation
        self.test_random_seeds = [83811, 14593, 3279, 97197, 36049, 32099, 29257, 18290, 
                                 96531, 13435, 88697, 97081, 71483, 11396, 77398, 55303]

        self.best_score = float('inf')  # For job shop, lower makespan is better
        self.total_scores = []
        self.cur_episode = 0
        self.total_timesteps = 0

        # Setup directories
        if config["save"]:
            self._setup_directories()
            self.save_config(config)
        else:
            print('Warning: SAVING is OFF!')
            self.run_id = "unidentified_run"
            self.save_path = ""
            
        self.announce()

    def _calculate_obs_space_size(self, obs_space):
        """Calculate the flattened size of the Dict observation space."""
        total_size = 0
        for space_name, space in obs_space.spaces.items():
            if space_name == 'job_progress':
                total_size += space.shape[0] * space.shape[1]  # job_progress is 2D
            else:
                total_size += np.prod(space.shape)
        return total_size

    def flatten_observation(self, obs):
        """Flatten a Dict observation into a 1D numpy array."""
        components = []
        
        # Flatten each component of the Dict observation
        for space_name in sorted(obs.keys()):  # Sort to ensure consistent order
            if space_name == 'job_progress':
                components.append(obs[space_name].flatten())
            else:
                components.append(obs[space_name].flatten())
                
        return np.concatenate(components)

    def announce(self):
        """Print initialization message."""
        print(f'TPPO {self.run_id} for JobShop has been initialized!')
        if self.config["save"]:
            print(f'Results will be saved to: {self.save_path}')
    
    def _setup_directories(self):
        """Create and setup directory structure."""
        # Define directory paths
        base_dir = './results'
        env_dir = os.path.join(base_dir, self.config['env_name'])
        algo_dir = os.path.join(env_dir, self.config['algorithm'])
        
        # Create directories
        os.makedirs(algo_dir, exist_ok=True)
        
        # Generate run ID
        try:
            existing_runs = [d for d in os.listdir(algo_dir) if d.startswith('run_')]
            next_run_number = len(existing_runs) + 1
        except FileNotFoundError:
            next_run_number = 1
            
        self.run_id = f'run_{next_run_number}'
        self.save_path = os.path.join(algo_dir, self.run_id)
        
        # Create run directory and subdirectories
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(os.path.join(self.save_path, 'models'), exist_ok=True)
        os.makedirs(os.path.join(self.save_path, 'results'), exist_ok=True)
        
        print(f"Created directory structure at: {self.save_path}")

    def save_config(self, config):
        """Save configuration to yaml file."""
        if self.config["save"]:
            config_path = os.path.join(self.save_path, 'config.yaml')
            with open(config_path, "w") as yaml_file:
                yaml.dump(config, yaml_file, default_flow_style=False)
            print(f"Saved configuration to: {config_path}")

    def calculate_returns(self):
        """Calculate returns using advantages and values."""
        self.batch.returns = self.batch.advantages + self.batch.values[:, 0, 0]
    
    def calculate_advantage(self):
        """Calculate advantage estimates."""
        advantage = 0
        for i in reversed(range(len(self.batch.actions))):
            if self.batch.trunc[i]:
                advantage = 0
            dt = self.batch.rewards[i] + self.config["gamma"] * self.batch.next_values[i][0][0] * int(
                not self.batch.terminated[i]) - \
                 self.batch.values[i][0][0]
            advantage = dt + self.config["gamma"] * self.config["lam"] * advantage * int(not self.batch.terminated[i])
            self.batch.advantages.insert(0, advantage)

    def normalize_advantages(self):
        """Normalize advantage estimates."""
        advantages = np.array(self.batch.advantages)
        norm_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
        self.batch.advantages = norm_advantages

    def rollout(self):
        """Perform a rollout in the environment."""
        obs, info = self.env.reset()
        flat_obs = self.flatten_observation(obs)
        
        done = False
        while not done:
            action, value, entropy = self.policy.get_action(flat_obs)
            action_idx = action[0]  # Convert to scalar
            
            # Store the current observation before stepping
            old_obs = flat_obs
            
            # Step the environment
            next_obs, reward, terminated, truncated, info = self.env.step(action_idx)
            flat_next_obs = self.flatten_observation(next_obs)
            
            # Save experience to batch
            self.batch.save_experience(
                action_idx,
                value,
                self.policy.critic.predict(np.array(flat_next_obs)),
                old_obs,
                reward,
                terminated,
                truncated,
                entropy
            )
            
            # Update for next iteration
            flat_obs = flat_next_obs
            self.total_timesteps += 1
            
            # Check for termination
            done = terminated or truncated
            
            # Check if we have enough steps for training
            if len(self.batch.actions) - 1 > self.config["n_timesteps"]:
                self.batch.convert_to_numpy()
                self.calculate_advantage()
                self.calculate_returns()
                self.train()
                self.batch.clear()

    def get_update_data_actor(self):
        """Prepare data for actor update."""
        tm = [{'observations': [], 'advantages': [], 'entropy': []} for _ in
              range(self.action_space_size)]
        for i in range(len(self.batch.actions)):
            idx = self.batch.actions[i]
            tm[idx]['observations'].append(self.batch.obs[i])
            tm[idx]['advantages'].append(self.batch.advantages[i])
            tm[idx]['entropy'].append(self.batch.entropies[i])
        return tm

    def get_update_data_critic(self):
        """Prepare data for critic update."""
        tm = {'observations': [], 'target': []}
        for i in range(len(self.batch.actions)):
            tm['observations'].append(self.batch.obs[i])
            tm['target'].append(self.batch.returns[i])
        return tm

    def train(self):
        """Train actor and critic networks."""
        for _ in range(self.config["epochs"]):
            actor_update = self.get_update_data_actor()
            self.policy.actor.update_2(actor_update)

            critic_update = self.get_update_data_critic()
            self.policy.critic.update(critic_update)

    def learn(self, nr_of_episodes):
        """
        Main learning loop.
        
        Args:
            nr_of_episodes: Number of episodes to train
        """
        for episode in tqdm(range(nr_of_episodes)):
            if episode % self.config['test_freq'] == 0:
                self.test()
                
            # Early stopping if we reach a good solution
            if self.best_score < self.config.get("threshold", float('inf')) and self.cur_episode == 100:
                break
                
            self.cur_episode = episode + 1
            self.rollout()

    def test(self):
        """Test the current policy on a set of fixed seeds."""
        episode_makespans = np.array([0 for _ in range(min(len(self.test_random_seeds), 10))])
        
        for episode, seed in enumerate(self.test_random_seeds[:10]):  # Test on first 10 seeds
            obs, info = self.env.reset(seed=seed)
            flat_obs = self.flatten_observation(obs)
            
            done = False
            while not done:
                action = self.policy.get_best_action(flat_obs)
                action_idx = action[0]  # Convert to scalar
                
                next_obs, reward, terminated, truncated, info = self.env.step(action_idx)
                flat_next_obs = self.flatten_observation(next_obs)
                
                flat_obs = flat_next_obs
                done = terminated or truncated
            
            # For job shop, we measure makespan (lower is better)
            episode_makespans[episode] = info['makespan']

        mean_makespan = np.mean(episode_makespans)
        std_makespan = np.std(episode_makespans)
        self.save_results(mean_makespan, std_makespan)
        self.total_scores.append(mean_makespan)
        
        # For job shop, lower makespan is better
        if mean_makespan < self.best_score:
            self.save_model()
            self.best_score = mean_makespan
            print(f'New best mean makespan: {mean_makespan}!')
            
            # Generate Gantt chart for best solution
            if self.config["save"]:
                gantt_path = os.path.join(self.save_path, 'best_solution.html')
                self.env.env.generate_html_gantt(gantt_path)
                print(f"Saved best solution Gantt chart to: {gantt_path}")

    def save_model(self):
        """Save the model state."""
        if self.config["save"]:
            try:
                tms = []
                for tm in range(len(self.policy.actor.tms)):
                    ta_state, clause_sign, clause_output, feedback_to_clauses = self.policy.actor.tms[tm].get_params()
                    
                    # Convert parameters to numpy arrays
                    ta_state_save = np.array(ta_state, dtype=np.int32)
                    clause_sign_save = np.array(clause_sign, dtype=np.int32)
                    clause_output_save = np.array(clause_output, dtype=np.int32)
                    feedback_to_clauses_save = np.array(feedback_to_clauses, dtype=np.int32)
                    
                    tms.append({
                        'ta_state': ta_state_save,
                        'clause_sign': clause_sign_save,
                        'clause_output': clause_output_save,
                        'feedback_to_clauses': feedback_to_clauses_save
                    })
                
                save_path = os.path.join(self.save_path, 'models', 'best')
                torch.save(tms, save_path)
                print(f"Saved model to: {save_path}")
                
            except Exception as e:
                print(f"Error saving model: {str(e)}")

    def save_results(self, mean, std):
        """Save test results to CSV."""
        if self.config["save"]:
            results_path = os.path.join(self.save_path, 'results', 'test_results.csv')
            
            # Create header if file doesn't exist
            if not os.path.exists(results_path):
                with open(results_path, "w") as f:
                    f.write("mean_makespan,std_makespan,steps\n")
            
            # Append results
            with open(results_path, "a") as f:
                f.write(f"{mean},{std},{self.total_timesteps}\n")

# Example Policy class structure for JobShop problem
class JobShopTPPOPolicy:
    def __init__(self, config):
        """
        Initialize a policy for Job Shop scheduling using Tsetlin Machines.
        
        Args:
            config: Configuration dictionary
        """
        # Import from your TM libraries
        from tsetlin_machines.src.networks.actor import MultiHeadPolicy
        from tsetlin_machines.src.networks.critic import GlobalCritic
        
        # Initialize actor and critic with the appropriate dimensions
        self.actor = MultiHeadPolicy(config)
        self.critic = GlobalCritic(config)
    
    def get_action(self, obs):
        """
        Get action from the policy with exploration.
        
        Args:
            obs: Flattened observation
            
        Returns:
            action: Selected action index
            value: Value estimate
            entropy: Entropy for the action
        """
        # Forward pass through actor and critic
        action, probs, entropy = self.actor.get_action(obs)
        value = self.critic.predict(np.array(obs))
        
        return action, value, entropy
    
    def get_best_action(self, obs):
        """
        Get the best action from the policy without exploration.
        
        Args:
            obs: Flattened observation
            
        Returns:
            action: Best action index
        """
        # Forward pass through actor
        action = self.actor.get_best_action(obs)
        return action

# Example usage code:
def create_jobshop_example():
    """Create a sample job shop problem."""
    
    # Create machine specifications
    machine_specs = [
        MachineSpec(max_slots=4, compatible_tools={1, 2, 3, 4, 5}),
        MachineSpec(max_slots=3, compatible_tools={2, 3, 4}),
        MachineSpec(max_slots=5, compatible_tools={1, 3, 5})
    ]
    
    # Create jobs
    jobs = [
        Job(operations=[
            Operation(duration=4, machine=0, required_tools={1, 2}),
            Operation(duration=3, machine=1, required_tools={2, 3}),
            Operation(duration=5, machine=2, required_tools={3, 5})
        ]),
        Job(operations=[
            Operation(duration=3, machine=1, required_tools={3, 4}),
            Operation(duration=6, machine=0, required_tools={1, 5}),
            Operation(duration=4, machine=2, required_tools={1, 3})
        ]),
        Job(operations=[
            Operation(duration=2, machine=2, required_tools={3, 5}),
            Operation(duration=3, machine=0, required_tools={2, 5})
        ]),
        Job(operations=[
            Operation(duration=5, machine=0, required_tools={1, 2}),
            Operation(duration=4, machine=1, required_tools={2, 4}),
            Operation(duration=3, machine=2, required_tools={3, 5})
        ])
    ]
    
    # Add dependencies
    jobs[1].operations[1].dependent_operations = [(1, 0)]
    jobs[2].operations[1].dependent_operations = [(0, 1)]
    
    # Create environment
    env = JobShopGym(jobs, machine_specs, render_mode='ansi')
    
    return env

def main():
    """Main function to run TPPO on JobShop."""
    # Create environment
    env = create_jobshop_example()
    
    # Define config
    config = {
        'algorithm': 'tppo',
        'env_name': 'jobshop',
        'save': True,
        'gamma': 0.99,
        'lam': 0.95,
        'epochs': 5,
        'n_timesteps': 128,
        'test_freq': 5,
        'threshold': 20,  # Target makespan threshold
        # Add TM-specific parameters
        'number_of_clauses': 100,
        'number_of_t_state': 100,
        'T': 10,
        's': 3.0,
        'batch_size': 64,
        'boost_true_positive_feedback': 1,
        'type_I_threshold': 0,
        'type_II_threshold': 0,
        'alpha_1': 0.1,
        'alpha_2': 0.1,
        'beta_1': 0.95,
        'beta_2': 0.95
    }
    
    # Create policy and TPPO
    policy = JobShopTPPOPolicy(config)
    tppo = TPPOJobShop(env, JobShopTPPOPolicy, config)
    
    # Train
    tppo.learn(nr_of_episodes=100)
    
    # Evaluate best policy
    print(f"\nBest mean makespan: {tppo.best_score}")
    
    # Generate final solution visualization
    if config["save"]:
        final_gantt_path = os.path.join(tppo.save_path, 'final_solution.html')
        env.env.generate_html_gantt(final_gantt_path)
        print(f"Saved final solution Gantt chart to: {final_gantt_path}")

if __name__ == "__main__":
    main()