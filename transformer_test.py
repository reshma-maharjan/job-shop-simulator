import numpy as np
from dataclasses import dataclass
from typing import List
from per_jsp.python.per_jsp.algorithms.qtm import HybridTsetlinQLearningScheduler, JSSPFeatureTransformer
from pyTsetlinMachine.tm import MultiClassTsetlinMachine

from dataclasses import dataclass
import numpy as np
from pyTsetlinMachine.tm import MultiClassTsetlinMachine

@dataclass
class Operation:
    """Represents a single operation in a job"""
    job_id: int
    machine: int  
    processing_time: int
    operation_id: int

@dataclass
class Job:
    """Represents a job with multiple operations"""
    job_id: int
    operations: list[Operation]

# Create a simple example
# Job 1: (M1, 3) -> (M2, 2)  # First on M1 for 3 time units, then M2 for 2
# Job 2: (M2, 4) -> (M1, 1)  # First on M2 for 4 time units, then M1 for 1

jobs = [
    Job(job_id=0, operations=[
        Operation(job_id=0, machine=0, processing_time=3, operation_id=0),
        Operation(job_id=0, machine=1, processing_time=2, operation_id=1)
    ]),
    Job(job_id=1, operations=[
        Operation(job_id=1, machine=1, processing_time=4, operation_id=0),
        Operation(job_id=1, machine=0, processing_time=1, operation_id=1)
    ])
]

class SimpleJSSPState:
    def __init__(self, jobs):
        self.jobs = jobs
        self.num_machines = 2
        self.machine_availability = np.zeros(self.num_machines)
        self.job_progress = [[0, 0], [0, 0]]  # Progress for each operation in each job

# Initialize the state
state = SimpleJSSPState(jobs)
print("Initial state:")
print(f"Machine availability: {state.machine_availability}")
print(f"Job progress: {state.job_progress}")

# Initialize feature transformer
feature_transformer = JSSPFeatureTransformer(
    n_jobs=2,
    n_machines=2,
    machine_time_bits=4,
    job_status_bits=2
)

# Transform initial state
initial_features = feature_transformer.transform(state)
print("\nInitial state features:", initial_features)

# Initialize Tsetlin Machine
tm = MultiClassTsetlinMachine(
    number_of_clauses=10,
    T=10,
    s=1.5
)

# Train TM on initial state (assuming class 0 for demonstration)
tm.fit(initial_features.reshape(1, -1), np.array([0]), epochs=1)

# Simulate first operation: Job 1's first operation on M1
state.machine_availability[0] = 3  # M1 busy for 3 time units
state.job_progress[0][0] = 1  # Mark first operation of Job 1 as complete

print("\nAfter scheduling first operation:")
print(f"Machine availability: {state.machine_availability}")
print(f"Job progress: {state.job_progress}")

# Transform new state
new_features = feature_transformer.transform(state)
print("\nNew state features:", new_features)

# Get TM prediction for new state
prediction = tm.predict(new_features.reshape(1, -1))
print("\nTM prediction for new state:", prediction)