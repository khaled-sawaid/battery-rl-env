
import pandas as pd
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.utils import set_random_seed

from supersuit import pettingzoo_env_to_vec_env_v1, concat_vec_envs_v1

from envs.multi_agent import BatteryParallelEnv


COL = "Day-ahead Price (EUR/MWh)"
data = pd.read_csv("datasets/energy_prices_2024_france.csv", usecols=[COL])
prices_mwh = pd.to_numeric(data[COL], errors="coerce").dropna()
prices_kwh = (prices_mwh / 1000.0).to_numpy(dtype=np.float32)


SEED = 42
n_agents = 5
episode_length = 24
step_hours = 1.0
NUM_PARALLEL_ENVS = 8  # how many full PettingZoo env copies

LEARNING_RATE = 3e-4
N_STEPS = 2048
BATCH_SIZE = 64
GAMMA = 0.99
TOTAL_TIMESTEPS = 200_000


par_env = BatteryParallelEnv(
    n_agents=n_agents,
    episode_length=episode_length,
    step_hours=step_hours,
    price_series=prices_kwh,
    seed=SEED,          # seeds the internal RNG of your env
    render_mode=None,   # required attribute for supersuit
)

# Convert PZ ParallelEnv -> vector env
vec_env = pettingzoo_env_to_vec_env_v1(par_env)

# Replicate for parallel training (SB3-compatible when base_class is specified)
vec_env = concat_vec_envs_v1(vec_env, NUM_PARALLEL_ENVS, num_cpus=1, base_class="stable_baselines3")

# Monitoring wrapper
vec_env = VecMonitor(vec_env)


set_random_seed(SEED)


model = PPO(
    policy="MlpPolicy",
    env=vec_env,
    verbose=1,
    learning_rate=LEARNING_RATE,
    n_steps=N_STEPS,
    batch_size=BATCH_SIZE,
    gamma=GAMMA,
    # IMPORTANT: do NOT pass seed here; ConcatVecEnv has no .seed()
)

print("\n Starting training...")
model.learn(total_timesteps=TOTAL_TIMESTEPS)
model.save("ppo_battery_parallel_shared_policy")

vec_env.close()
print("\n Training finished. Model saved as 'ppo_battery_parallel_shared_policy.zip'")
