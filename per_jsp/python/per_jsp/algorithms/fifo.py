from per_jsp.python.per_jsp.algorithms.base import BaseScheduler
from per_jsp.python.per_jsp.environment.job_shop_environment import JobShopEnvironment, Action


class FIFOScheduler(BaseScheduler):
    def solve(self, env: JobShopEnvironment, max_steps: int = 1000):
        """
        Improved FIFO scheduler that considers machine availability
        """
        env.reset()
        actions = []
        
        # Keep track of next operation for each job
        job_queue = {i: 0 for i in range(len(env.jobs))}
        
        while not env.is_done() and len(actions) < max_steps:
            possible_actions = env.get_possible_actions()
            if not possible_actions:
                break
                
            # Group actions by machine
            machine_actions = {}
            for action in possible_actions:
                machine = env.jobs[action.job].operations[action.operation].machine
                if machine not in machine_actions:
                    machine_actions[machine] = []
                machine_actions[machine].append(action)
            
            # For each machine, take the first queued action
            for machine in sorted(machine_actions.keys()):
                machine_queue = machine_actions[machine]
                if machine_queue:
                    # Take the first action that's ready for this machine
                    selected_action = min(machine_queue, key=lambda x: x.job)
                    
                    env.step(selected_action)
                    actions.append(selected_action)
                    
                    # Update job's next operation
                    job_queue[selected_action.job] = selected_action.operation + 1
                    
                    # Break inner loop if environment is done
                    if env.is_done():
                        break
            
        makespan = env.total_time
        return actions, makespan