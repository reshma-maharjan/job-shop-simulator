machines = ['engine_hoist', 'wheel_station', 'inspector']

job_1 = [
    {'name': 'add_engine_1', 'machine': 'engine_hoist', 'duration': 30, 'next': 'add_wheels_1'},
    {'name': 'add_wheels_1', 'machine': 'wheel_station', 'duration': 30, 'next': 'inspect_1'},
    {'name': 'inspect_1', 'machine': 'inspector', 'duration': 10, 'next': None}
]

job_2 = [
    {'name': 'add_engine_2', 'machine': 'engine_hoist', 'duration': 60, 'next': 'add_wheels_2'},
    {'name': 'add_wheels_2', 'machine': 'wheel_station', 'duration': 15, 'next': 'inspect_2'},
    {'name': 'inspect_2', 'machine': 'inspector', 'duration': 10, 'next': None}
]

jobs = [job_1, job_2]

print(job_2[1])

schedule_1 = {
    'engine_hoist': [(job_1[0], 0), (job_2[0], 30)],
    'wheel_station': [(job_1[1], 30), (job_2[1], 90)],
    'inspector': [(job_1[2], 60), (job_2[2], 105)]
}

schedule_2 = {
    'engine_hoist': [(job_2[0], 0), (job_1[0], 60)],
    'wheel_station': [(job_2[1], 60), (job_1[1], 90)],
    'inspector': [(job_2[2], 75), (job_1[2], 120)]
}

# Calculate the makespan of a schedule
def makespan(schedule):
    last_finish_time = 0
    for machine_schedule in schedule.values():
        last_processing = machine_schedule[-1]
        machine_finish_time = last_processing[1] + last_processing[0]['duration']
        
        if machine_finish_time > last_finish_time:
            last_finish_time = machine_finish_time
            
    return last_finish_time

print(f'Schedule 1 makespan = {makespan(schedule_1)}')
print(f'Schedule 2 makespan = {makespan(schedule_2)}')

def create_init_state(machines, jobs):
    init_state = {
        "machine_idle_time": {},
        "operation_ready_time": {},
        "schedule": {},
        "completed": [] # no operation is completed initially
    }

    # Initially, the idle time of all the machines are 0.
    for m in machines:
        init_state["machine_idle_time"][m] = 0

    # Initially, only the first operation of each job is ready at time 0
    for job in jobs:
        init_state["operation_ready_time"][job[0]["name"]] = 0

    # Initially, the machine schedules are empty
    for m in machines:
        init_state["schedule"][m] = []
    
    return init_state

init_state = create_init_state(machines, jobs)

print(f'Initial machine idle time: {init_state["machine_idle_time"]}')
print(f'Initial operation ready time: {init_state["operation_ready_time"]}')



# Import the copy library to generate new states conveniently
import copy

def state_space_search(machines, jobs):
    # Create the operation dictionary for state space search
    operations = {}
    for job in jobs:
        for op in job:
            operations[op['name']] = op

    # Initially, the best schedule is None (no schedule is found), and best makespan is infinity
    best_schedule = None
    best_makespan = float('inf')

    # Initially, the fringe has the initial state only
    init_state = create_init_state(machines, jobs)
    fringe = [init_state]

    while len(fringe) > 0:
        # Arbitrarily pick the first state in the fringe
        state = fringe[0] 

        # Check if the current state is a goal state
        if len(state["completed"]) == len(operations):
            # Calculate the makespan of the complete schedule
            new_makespan = makespan(state["schedule"])

            if new_makespan < best_makespan:
                best_schedule = state["schedule"]
                best_makespan = new_makespan

        # Each ready operation has a non-delay applicable action to process it.
        # Its start time is max[operation_ready_time, machine_idle_time].
        applicable_actions = []
        for opname in state["operation_ready_time"].keys():
            op = operations[opname]
            machine = op['machine']        
            time = state["operation_ready_time"][opname]        
            if time < state["machine_idle_time"][machine]:
                time = state["machine_idle_time"][machine]

            applicable_actions.append((op, time))

        # Expand the branches, each by an applicable action
        for a in applicable_actions:
            op = a[0] # action operation
            machine = op['machine'] # action machine
            time = a[1] # action start time
            finish_time = time + op["duration"] # action finish time

            # Apply the action to the current state, create the next state
            next_state = copy.deepcopy(state) # deep copy
            next_state["schedule"][machine].append(a) # add action to corresponding machine schedule sequence
            next_state["completed"].append(op['name']) # add operation name to completed
            next_state["operation_ready_time"].pop(op['name']) # delete completed operation name from ready time
            next_state["machine_idle_time"][machine] = finish_time # machine will become idle after completion

            # If the operation is not the last operation of the job, then its next operation becomes ready
            if op['next'] != None:
                next_op = operations[op['next']]
                next_state["operation_ready_time"][next_op['name']] = finish_time

            fringe.append(next_state)
        fringe.remove(state)
    
    return best_schedule


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# Convert schedule into DataFrame
def schedule_data_frame(schedule):
    schedule_dict = []
    for machine_schedule in schedule.values():
        for action in machine_schedule:
            a_dict = {
                'Operation': action[0]['name'],
                'Machine': action[0]['machine'],
                'Start': action[1],
                'Duration': action[0]['duration'],
                'Finish': action[1] + action[0]['duration']
            }
            schedule_dict.append(a_dict)
    
    return pd.DataFrame(schedule_dict)

# Plot gantt chart from schedule
def gantt_chart(schedule):
    schedule = schedule_data_frame(schedule)
    
    JOBS = sorted(list(schedule['Operation'].unique()))
    MACHINES = sorted(list(schedule['Machine'].unique()))
    makespan = schedule['Finish'].max()
    
    bar_style = {'alpha':1.0, 'lw':25, 'solid_capstyle':'butt'}
    text_style = {'color':'white', 'weight':'bold', 'ha':'center', 'va':'center'}
    colors = mpl.cm.Dark2.colors

    schedule.sort_values(by=['Operation', 'Start'])
    schedule.set_index(['Operation', 'Machine'], inplace=True)

    fig, ax = plt.subplots(2,1, figsize=(12, 5+(len(JOBS)+len(MACHINES))/4))

    for jdx, j in enumerate(JOBS, 1):
        for mdx, m in enumerate(MACHINES, 1):
            if (j,m) in schedule.index:
                xs = schedule.loc[(j,m), 'Start']
                xf = schedule.loc[(j,m), 'Finish']
                ax[0].plot([xs, xf], [jdx]*2, c=colors[mdx%7], **bar_style)
                ax[0].text((xs + xf)/2, jdx, m, **text_style)
                ax[1].plot([xs, xf], [mdx]*2, c=colors[jdx%7], **bar_style)
                ax[1].text((xs + xf)/2, mdx, j, **text_style)
                
    ax[0].set_title('Job Schedule')
    ax[0].set_ylabel('Operation')
    ax[1].set_title('Machine Schedule')
    ax[1].set_ylabel('Machine')
    
    for idx, s in enumerate([JOBS, MACHINES]):
        ax[idx].set_ylim(0.5, len(s) + 0.5)
        ax[idx].set_yticks(range(1, 1 + len(s)))
        ax[idx].set_yticklabels(s)
        ax[idx].text(makespan, ax[idx].get_ylim()[0]-0.2, "{0:0.1f}".format(makespan), ha='center', va='top')
        ax[idx].plot([makespan]*2, ax[idx].get_ylim(), 'r--')
        ax[idx].set_xlabel('Time')
        ax[idx].grid(True)
        
    fig.tight_layout()

best_schedule = state_space_search(machines, jobs)
gantt_chart(best_schedule)


machines = ['engine_hoist', 'wheel_station', 'inspector']

job_1 = [
    {'name': 'add_engine_1', 'machine': 'engine_hoist', 'duration': 30, 'next': 'add_wheels_1'},
    {'name': 'add_wheels_1', 'machine': 'wheel_station', 'duration': 30, 'next': 'inspect_1'},
    {'name': 'inspect_1', 'machine': 'inspector', 'duration': 10, 'next': None}
]

job_2 = [
    {'name': 'add_wheels_2', 'machine': 'wheel_station', 'duration': 15, 'next': 'add_engine_2'},
    {'name': 'add_engine_2', 'machine': 'engine_hoist', 'duration': 60, 'next': 'inspect_2'},    
    {'name': 'inspect_2', 'machine': 'inspector', 'duration': 10, 'next': None}
]

jobs = [job_1, job_2]


schedule_2 = state_space_search(machines, jobs)
gantt_chart(schedule_2)