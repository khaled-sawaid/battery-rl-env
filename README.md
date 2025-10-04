# BatteryEnv: Reinforcement Learning Environment for Battery Storage

## Features
- Implements the standard **Gymnasium API** (`reset`, `step`, `render`).
- Supports **continuous** or **discrete** action spaces:
  - Continuous: charging/discharging rate scaled to `[-1, 1]`.
  - Discrete: `{0=discharge, 1=idle, 2=charge}`.
- Observations:
  - Battery state-of-charge (SoC, as % of capacity).
  - Current electricity price.
- Reward: **profit** from buying/selling electricity.
- Configurable parameters (episode length, charge/discharge rates, efficiencies, min/max capacity).
- Flexible: works with any electricity price time series.

## Planned Work
- Implement a **Q-learning agent**.
- Experiment with more advanced RL algorithms (e.g., DQN, PPO).
- Compare policies across different price datasets.

## Dataset
Currently using **2024 France electricity market prices** (public dataset).  
Easily replaceable with your own price series (any numeric 1D array).

## Example Usage
```python
import numpy as np
from battery_env import BatteryEnv

# Example: simple random agent
price_series = np.random.uniform(50, 150, size=1000)  # replace with real data
env = BatteryEnv(price_series=price_series)

obs, info = env.reset()
done = False
while not done:
    action = env.action_space.sample()  # random action
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    print(obs, reward, info)
