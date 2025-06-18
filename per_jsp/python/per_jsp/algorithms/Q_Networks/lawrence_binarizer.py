import numpy as np
import os

# Import required classes - adjust these imports to match your project structure
from per_jsp.python.per_jsp.environment.job_shop_environment import Job, Operation, MachineSpec
from per_jsp.python.per_jsp.environment.jsp_env_gym import JobShopGym
from JSPBinarizer import JSPBinarizer

def binarize_lawrence(file_path, feature_bits=8, save_to_file=None):
    """
    Simple function to binarize a Lawrence JSP instance.
    
    Args:
        file_path: Path to the Lawrence instance file
        feature_bits: Number of bits for feature representation (default: 8)
        save_to_file: Path to save the binary features (if None, won't save)
        
    Returns:
        binary_features: NumPy array of binary features
    """
    # Parse the Lawrence instance
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # First line contains number of jobs and machines
    first_line = lines[0].strip().split()
    num_jobs = int(first_line[0])
    num_machines = int(first_line[1])
    
    # Process the processing times
    processing_times = []
    for i in range(1, num_jobs + 1):
        times = [int(t) for t in lines[i].strip().split()]
        processing_times.append(times)
    
    # Process the machine sequences
    machine_sequences = []
    for i in range(num_jobs + 1, 2 * num_jobs + 1):
        machines = [int(m) - 1 for m in lines[i].strip().split()]  # Convert to 0-based indexing
        machine_sequences.append(machines)
    
    # Create Job objects
    jobs = []
    for job_idx in range(num_jobs):
        operations = []
        for op_idx in range(num_machines):
            machine_id = machine_sequences[job_idx][op_idx]
            duration = processing_times[job_idx][op_idx]
            operations.append(Operation(
                duration=duration,
                machine=machine_id,
                required_tools=set()  # No tools in Lawrence instances
            ))
        jobs.append(Job(operations=operations))
    
    print(f"Parsed {len(jobs)} jobs with {num_machines} machines")
    
    # Create machine specs
    machine_specs = [MachineSpec() for _ in range(num_machines)]
    
    # Create the environment
    env = JobShopGym(jobs, machine_specs, render_mode=None)
    
    # Create binarizer
    binarizer = JSPBinarizer(env, feature_bits=feature_bits)
    
    # Get observation and transform
    obs, _ = env.reset()
    binary_features = binarizer.transform(obs)
    
    # Print summary
    print(f"Binary features shape: {binary_features.shape}")
    print(f"Non-zero elements: {np.count_nonzero(binary_features)} out of {binary_features.size} "
          f"({np.count_nonzero(binary_features)/binary_features.size*100:.2f}%)")
    
    # Save if requested
    if save_to_file:
        np.save(save_to_file, binary_features)
        print(f"Binary features saved to {save_to_file}")
    
    return binary_features

# Simple command-line usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python lawrence_binarizer.py <lawrence_file_path> [output_file.npy]")
        sys.exit(1)
    
    file_path = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    binarize_lawrence(file_path, save_to_file=output_file)