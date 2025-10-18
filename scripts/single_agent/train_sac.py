import pandas as pd
import numpy as np
from envs.single_agent import BatteryEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import SAC  



COL = "Day-ahead Price (EUR/MWh)"
data = pd.read_csv("datasets/energy_prices_2024_france.csv", usecols=[COL])
prices_mwh = pd.to_numeric(data[COL], errors="coerce").dropna()
prices_kwh = (prices_mwh / 1000.0).to_numpy(dtype=np.float32)


SEED = 42
episode_length = 24
step_hours = 1.0

base_env = BatteryEnv(
    price_series=prices_kwh,
    seed=SEED,
    episode_length=episode_length,
    step_hours=step_hours,
)
env = Monitor(base_env)


model = SAC(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    learning_rate=3e-4,
    gamma=0.99,
    buffer_size=1_000_000,
    batch_size=256,         
    learning_starts=10_000,  
    train_freq=(1, "step"),  
    gradient_steps=1,       
    tau=0.005,               
    ent_coef="auto",         
    seed=SEED,
)

model.learn(total_timesteps=500_000)
model.save("sac_battery_env")
