import pandas as pd
import numpy as np
from battery_env import BatteryEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO


COL = "Day-ahead Price (EUR/MWh)"
data = pd.read_csv("/datasets/GUI_ENERGY_PRICES_202312312300-202412312300.csv", usecols=[COL])
prices_mwh = pd.to_numeric(data[COL], errors="coerce")
prices_mwh = prices_mwh.dropna()
prices_kwh = (prices_mwh / 1000.0).to_numpy(dtype=np.float32)

SEED = 42
episode_length = 24
step_hours = 1.0

env = BatteryEnv(price_series=prices_kwh, seed=SEED, episode_length=episode_length, step_hours=step_hours)

env = Monitor(env)

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
model.save("ppo_battery_env")
