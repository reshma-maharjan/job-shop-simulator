from ortools.sat.python import cp_model
from per_jsp.python.per_jsp.algorithms.base import BaseScheduler
from per_jsp.python.per_jsp.environment.job_shop_environment import JobShopEnvironment, Action
import logging

logger = logging.getLogger(__name__)

class ORToolsScheduler(BaseScheduler):
    def __init__(self):
        super().__init__()
        self.time_limit = 60  # Default time limit in seconds

    def set_time_limit(self, time_limit: int):
        """Set time limit for the solver in seconds."""
        self.time_limit = time_limit

    def solve(self, env: JobShopEnvironment, max_steps: int = 1000):
        """
        Constraint Programming scheduler using Google OR-Tools
        """
        # Reset environment state
        env.reset()
        
        # Create the model
        model = cp_model.CpModel()

        # Calculate horizon from operation durations
        horizon = sum(op.duration for job in env.jobs for op in job.operations)
        
        # Create variables
        all_tasks = {}
        all_intervals = {}
        machine_to_intervals = {}

        # Create variables for each operation
        for job_id, job in enumerate(env.jobs):
            for op_id, op in enumerate(job.operations):
                start = model.NewIntVar(0, horizon, f'start_{job_id}_{op_id}')
                duration = op.duration
                end = model.NewIntVar(0, horizon, f'end_{job_id}_{op_id}')
                
                interval = model.NewIntervalVar(start, duration, end,
                                             f'interval_{job_id}_{op_id}')
                
                all_tasks[(job_id, op_id)] = {
                    'start': start,
                    'duration': duration,
                    'end': end,
                    'interval': interval,
                    'machine': op.machine
                }

                # Add to machine intervals
                if op.machine not in machine_to_intervals:
                    machine_to_intervals[op.machine] = []
                machine_to_intervals[op.machine].append(interval)

        # Add precedence constraints (operations within the same job)
        for job_id, job in enumerate(env.jobs):
            for op_id in range(len(job.operations) - 1):
                model.Add(
                    all_tasks[(job_id, op_id)]['end'] <= 
                    all_tasks[(job_id, op_id + 1)]['start']
                )

        # Add job dependencies
        for job_id, job in enumerate(env.jobs):
            for dep_job_id in job.dependent_jobs:
                # Last operation of dependent job must complete before first operation of current job
                dep_last_op = len(env.jobs[dep_job_id].operations) - 1
                model.Add(
                    all_tasks[(dep_job_id, dep_last_op)]['end'] <= 
                    all_tasks[(job_id, 0)]['start']
                )

        # Add operation dependencies
        for job_id, job in enumerate(env.jobs):
            for op_id, op in enumerate(job.operations):
                for dep_job_id, dep_op_id in op.dependent_operations:
                    model.Add(
                        all_tasks[(dep_job_id, dep_op_id)]['end'] <= 
                        all_tasks[(job_id, op_id)]['start']
                    )

        # Add machine constraints
        for machine_intervals in machine_to_intervals.values():
            model.AddNoOverlap(machine_intervals)

        # Objective: minimize makespan
        makespan = model.NewIntVar(0, horizon, 'makespan')
        for job_id in range(len(env.jobs)):
            last_op_id = len(env.jobs[job_id].operations) - 1
            model.Add(all_tasks[(job_id, last_op_id)]['end'] <= makespan)
        
        model.Minimize(makespan)

        # Create solver and solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.time_limit
        
        # Enable logging for debugging
        logger.info(f"Starting solver with time limit {self.time_limit} seconds...")
        status = solver.Solve(model)
        logger.info(f"Solver finished with status: {solver.StatusName(status)}")

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            # Convert solution to action sequence
            operation_starts = []
            for job_id in range(len(env.jobs)):
                for op_id in range(len(env.jobs[job_id].operations)):
                    start_time = solver.Value(all_tasks[(job_id, op_id)]['start'])
                    machine = env.jobs[job_id].operations[op_id].machine
                    operation_starts.append((start_time, job_id, op_id, machine))
            
            # Sort operations by start time
            operation_starts.sort()
            
            # Convert to actions and apply them to the environment
            actions = []
            for _, job_id, op_id, machine in operation_starts:
                action = Action(
                    job=job_id,
                    operation=op_id,
                    machine=machine
                )
                actions.append(action)
                
                # Apply action to environment
                env.step(action)
            
            # Get makespan value
            final_makespan = solver.Value(makespan)
            
            logger.info(f"Found solution with makespan: {final_makespan}")
            return actions, final_makespan
        else:
            error_msg = f"No solution found! Solver status: {solver.StatusName(status)}"
            if status == cp_model.INFEASIBLE:
                # Add debugging information
                error_msg += "\nProblem is infeasible. Check constraints:"
                error_msg += f"\n- Number of jobs: {len(env.jobs)}"
                error_msg += f"\n- Horizon calculated: {horizon}"
                error_msg += "\n- Machine assignments and durations:"
                for job_id, job in enumerate(env.jobs):
                    for op_id, op in enumerate(job.operations):
                        error_msg += f"\n  Job {job_id} Op {op_id}: Machine {op.machine}, Duration {op.duration}"
            raise RuntimeError(error_msg)