from stable_baselines3.common.env_util import make_vec_env

def job_shop_env_creator():
    from job_shop_env import JobShopGymEnv, Job, Operation
    job1 = Job([Operation(3, 0), Operation(5, 1)])
    job2 = Job([Operation(2, 1), Operation(7, 2)])
    jobs = [job1, job2]
    return JobShopGymEnv(jobs=jobs)

# Create the environment without register_env
env = make_vec_env(job_shop_env_creator, n_envs=1)

import sys
print(sys.executable)