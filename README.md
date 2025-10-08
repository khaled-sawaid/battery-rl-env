# ⚡ Battery RL Environment

A lightweight yet extendable **Reinforcement Learning project** for optimizing **battery energy storage** using real-world electricity prices.  
Implements the **Gymnasium** API for single-agent training and is being extended into a **multi-agent environment** using **PettingZoo**.  
Built for learning and experimentation with **Stable-Baselines3** and **MARL frameworks**.

---

## 📘 Overview

This repo contains:
- **Single-agent environment** (`BatteryEnv`) in `envs/single_agent.py`
- **Training scripts** (PPO, SAC) under `scripts/single_agent/`
- **Evaluation** tools comparing trained vs. random policies
- A **multi-agent extension** (in progress) in `envs/multi_agent.py`
- Real electricity **price datasets** under `datasets/`

---

## 🚀 Features

✅ Standard **Gymnasium interface** (`reset`, `step`, `render`)  
✅ Works directly with **Stable-Baselines3**  
✅ Supports **continuous** (`[-1, 1]`) or **discrete** (`{0=discharge, 1=idle, 2=charge}`) actions  
✅ Observations: battery **state-of-charge** and **current price**  
✅ Reward = **profit** from buying/selling electricity  
✅ Configurable (episode length, capacities, rates, efficiencies, etc.)  
✅ Fully reproducible via seeding  
✅ Clean structure for scaling to **multi-agent** (PettingZoo-ready)

---

## 🧩 Project Structure

```
battery-rl-env/
├─ envs/
│  ├─ single_agent.py          # BatteryEnv (Gymnasium)
│  └─ multi_agent.py           # WIP: MARL version (PettingZoo)
├─ scripts/
│  ├─ single_agent/
│  │  ├─ train_ppo.py          # Train PPO agent
│  │  ├─ train_sac.py          # Train SAC agent
│  │  └─ eval.py               # Evaluate trained vs random agent
│  └─ multi_agent/
│     ├─ train_independent.py  # (planned) Independent multi-agent training
│     └─ eval.py               # (planned) MARL evaluation
├─ utils/
│  ├─ pricing.py               # load/validate price series
│  ├─ metrics.py               # profit summaries & win-rate
│  └─ seeding.py               # reproducible RNG helpers
├─ datasets/
│  ├─ energy_prices_2023_france.csv
│  └─ energy_prices_2024_france.csv
└─ tests/                      # small sanity checks
```

---

## 🧠 Planned Work
- ✅ Train & evaluate **PPO** and **SAC** agents on real datasets  
- 🧩 Implement **multi-agent (PettingZoo)** version  
- 📈 Add logging, normalization, and richer evaluation metrics  
- 🔁 Experiment with **shared-grid / price-coupled MARL setups**

---

## 💻 Quick Example

```python
import numpy as np
from envs.single_agent import BatteryEnv

# Random price series for demo
price_series = np.random.uniform(50, 150, size=1000)

env = BatteryEnv(price_series=price_series, continuous_action=True)
obs, info = env.reset()

done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    print(obs, reward)
```

---

## 📚 Requirements
```
gymnasium
numpy
pandas
stable-baselines3
pettingzoo      # (for future MARL)
```

Install with:
```bash
pip install -r requirements.txt
```
