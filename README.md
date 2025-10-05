# ⚡ BatteryEnv

A simple **Reinforcement Learning environment** for optimizing **battery energy storage** based on electricity prices.  
Implements the **Gymnasium** API and works directly with **Stable-Baselines3**.

---

## Features
- Standard Gymnasium interface (`reset`, `step`, `render`)
- Supports **continuous** (`[-1, 1]`) or **discrete** (`{0=discharge, 1=idle, 2=charge}`) actions  
- Observations: battery **state-of-charge** and **current price**
- Reward = **profit** from buying/selling electricity  
- Fully configurable (episode length, rates, efficiencies, capacity, etc.)
- Compatible with any 1D electricity price series

---

## Planned Work
- Train agents using **PPO**, **SAC**, and **DQN** (via Stable-Baselines3)  
- Add normalization and logging  
- Compare results across different price datasets  

---

## Example
```python
import numpy as np
from battery_env import BatteryEnv

price_series = np.random.uniform(50, 150, size=1000)
env = BatteryEnv(price_series=price_series, continuous_action=True)

obs, info = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    print(obs, reward)
