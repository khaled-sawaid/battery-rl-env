import pandas as pd
import numpy as np
from envs.single_agent import BatteryEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO


COL = "Day-ahead Price (EUR/MWh)"
data = pd.read_csv("datasets/energy_prices_2024_france.csv", usecols=[COL])
prices_mwh = pd.to_numeric(data[COL], errors="coerce")
prices_mwh = prices_mwh.dropna()
prices_kwh = (prices_mwh / 1000.0).to_numpy(dtype=np.float32)

SEED = 42
episode_length = 24
step_hours = 1.0

base_env = BatteryEnv(price_series=prices_kwh, seed=SEED, episode_length=episode_length, step_hours=step_hours)
env = Monitor(base_env)

model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048, 
    batch_size=64,
    gamma=0.99,
)

model.learn(total_timesteps=200_000)
model.save("trained_agents/ppo_battery_env")
