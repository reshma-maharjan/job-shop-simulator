
from job_shop_env import JobShopGymEnv, Job, Operation  # Import necessary classes

# Create jobs
jobs = [
    Job(operations=[
        Operation(duration=3, machine=0),
        Operation(duration=2, machine=1)
    ]),
    Job(operations=[
        Operation(duration=2, machine=1),
        Operation(duration=4, machine=0)
    ])
]

# Create environment
env = JobShopGymEnv(jobs)

# Use environment
obs = env.reset()
done = False
step = 0

print("Initial State:")
env.render()

while not done:
    step += 1
    print(f"\nStep {step}:")
    
    action = env.action_space.sample()  # Random action
    obs, reward, done, truncated, info = env.step(action)
    
    print(f"Action taken: Job {action // (env.num_machines * env.max_operations)}, "
          f"Machine {(action % (env.num_machines * env.max_operations)) // env.max_operations}, "
          f"Operation {action % env.max_operations}")
    print(f"Reward: {reward:.2f}")
    print(f"Info: {info}")
    print("\nCurrent Schedule:")
    env.render()

print("\nFinal Schedule:")
env.render()