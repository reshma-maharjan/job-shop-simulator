import numpy as np

def create_random_job_shop_state(num_jobs=4, num_machines=3, num_operations=4, num_tools=8):
    """
    Create a random initialization for a job shop scheduling problem.
    
    Parameters:
    -----------
    num_jobs: int
        Number of jobs in the system
    num_machines: int
        Number of available machines
    num_operations: int
        Maximum number of operations per job
    num_tools: int
        Number of tools per machine
        
    Returns:
    --------
    dict: A dictionary containing the state representation
    """
    # Random job progress (how far each job has progressed through its operations)
    # Most will be 0 initially with some random progress
    job_progress = np.zeros((num_jobs, num_operations), dtype=np.float32)
    
    # Randomly set some jobs to have partial progress (10-30%)
    for i in range(num_jobs):
        for j in range(num_operations):
            if np.random.random() < 0.2:  # 20% chance of having progress
                job_progress[i, j] = np.random.uniform(0.1, 0.3)
    
    # Machine availability (0 = available, 1 = busy)
    machine_availability = np.random.choice([0, 1], size=num_machines, 
                                           p=[0.7, 0.3]).astype(np.float32)
    
    # Next operation for each job (integer index)
    next_operation_for_job = np.random.randint(0, num_operations, size=num_jobs, dtype=np.int32)
    
    # Completed jobs (binary)
    completed_jobs = np.zeros(num_jobs, dtype=np.int8)
    
    # Randomly mark some jobs as completed
    for i in range(num_jobs):
        if np.random.random() < 0.15:  # 15% chance of being completed
            completed_jobs[i] = 1
    
    # Job readiness (binary) - whether a job is ready to be processed
    job_ready = np.random.choice([0, 1], size=num_jobs, p=[0.4, 0.6]).astype(np.int8)
    
    # Tool state matrix (which tools are available on which machines)
    tool_state = np.random.choice([0, 1], size=(num_machines, num_tools), 
                                 p=[0.7, 0.3]).astype(np.int8)
    
    # Create the observation dictionary
    observation = {
        'job_progress': job_progress,
        'machine_availability': machine_availability,
        'next_operation_for_job': next_operation_for_job,
        'completed_jobs': completed_jobs,
        'job_ready': job_ready,
        'tool_state': tool_state
    }
    
    return observation

# Generate a random observation
random_obs = create_random_job_shop_state()

# Print the observation in a format similar to the example
print("random_obs: {")
for key, value in random_obs.items():
    print(f"    '{key}': {repr(value)},")
print("}")

# Generate multiple observations for training
def generate_training_dataset(num_samples=100):
    """Generate multiple random observations for training a Tsetlin Machine"""
    dataset = []
    for _ in range(num_samples):
        obs = create_random_job_shop_state()
        dataset.append(obs)
    return dataset

# Example usage to generate 10 random states
#training_data = generate_training_dataset(5)
#print(f"\nGenerated {len(training_data)} random observations for training")
# Save data to text file
def save_observations_to_txt(observations, filename="per_jsp/python/per_jsp/algorithms/misc/tsetlin_job_shop_data.txt"):
    """
    Save a list of observations to a text file in a format that can be easily parsed later.
    
    Parameters:
    -----------
    observations: list
        List of observation dictionaries
    filename: str
        Name of the file to save the data to
    """
    with open(filename, 'w') as f:
        # Write the number of observations
        f.write(f"{len(observations)}\n")
        
        # For each observation
        for obs in observations:
            # Write each component of the observation
            for key, value in obs.items():
                f.write(f"{key}\n")
                
                # Handle different array shapes
                if len(value.shape) == 1:
                    f.write(f"{value.shape[0]}\n")
                    for i in range(value.shape[0]):
                        f.write(f"{value[i]} ")
                    f.write("\n")
                elif len(value.shape) == 2:
                    f.write(f"{value.shape[0]} {value.shape[1]}\n")
                    for i in range(value.shape[0]):
                        for j in range(value.shape[1]):
                            f.write(f"{value[i, j]} ")
                        f.write("\n")
            
            f.write("---\n")  # Separator between observations
            
    print(f"Saved {len(observations)} observations to {filename}")
# first generate some 100 random data
training_data = generate_training_dataset(100)

# save the data to a text file
save_observations_to_txt(training_data)

# Function to load the saved data
def load_observations_from_txt(filename="tsetlin_job_shop_data.txt"):
    """
    Load observations from a text file.
    
    Parameters:
    -----------
    filename: str
        Name of the file to load the data from
        
    Returns:
    --------
    list: A list of observation dictionaries
    """
    observations = []
    
    with open(filename, 'r') as f:
        num_obs = int(f.readline().strip())
        
        for _ in range(num_obs):
            obs = {}
            while True:
                key = f.readline().strip()
                if key == "---":
                    break
                
                shape_line = f.readline().strip()
                shape_parts = shape_line.split()
                
                if len(shape_parts) == 1:
                    # 1D array
                    size = int(shape_parts[0])
                    values = f.readline().strip().split()
                    if key in ['next_operation_for_job']:
                        array = np.array([int(x) for x in values[:size]], dtype=np.int32)
                    elif key in ['completed_jobs', 'job_ready', 'tool_state']:
                        array = np.array([int(x) for x in values[:size]], dtype=np.int8)
                    else:
                        array = np.array([float(x) for x in values[:size]], dtype=np.float32)
                    obs[key] = array
                else:
                    # 2D array
                    rows, cols = int(shape_parts[0]), int(shape_parts[1])
                    array = np.zeros((rows, cols))
                    for i in range(rows):
                        values = f.readline().strip().split()
                        for j in range(cols):
                            if key == 'tool_state':
                                array[i, j] = int(values[j])
                            else:
                                array[i, j] = float(values[j])
                    
                    if key == 'job_progress':
                        obs[key] = array.astype(np.float32)
                    elif key == 'tool_state':
                        obs[key] = array.astype(np.int8)
                    else:
                        obs[key] = array
            
            observations.append(obs)
    
    print(f"Loaded {len(observations)} observations from {filename}")
    return observations
