#  BatteryEnv — Reinforcement Learning Environment for Battery Arbitrage

A lightweight yet extendable **Reinforcement Learning environment** for optimizing **battery energy storage** using real-world electricity prices.  
Implements the **Gymnasium** API for single-agent control and a **PettingZoo** parallel environment for multi-agent setups.  
Built for learning, experimentation, and benchmarking with **Stable-Baselines3** and **MARL frameworks**.

---

##  Overview

BatteryEnv simulates a battery that **charges when electricity is cheap** and **discharges when it’s expensive** to maximize profit.

It includes:
-  **Single-agent environment** (`BatteryEnv`)
-  **Multi-agent environment** (`BatteryParallelEnv`)
-  Pretrained **PPO**, **SAC**, and **TD3** agents
-  Evaluation scripts comparing learned policies vs. random baselines

---

##  Features

-  Standard **Gymnasium interface** (`reset`, `step`, `render`)  
-  Continuous (`[-1, 1]`) or discrete (`{0=discharge, 1=idle, 2=charge}`) actions  
-  Observations: battery **state-of-charge** and **current price**  
-  Reward = profit from buying/selling electricity  
-  Fully configurable: episode length, capacity, rates, efficiencies  
-  Real 2023 & 2024 electricity price datasets  
-  Multi-agent support via **PettingZoo ParallelEnv**  
-  Evaluation tools with win-rate and profit summaries

---

##  Project Structure

```
battery-rl-env/
├─ envs/
│  ├─ single_agent.py            # Single-agent environment
│  └─ multi_agent.py             # Multi-agent environment
├─ scripts/
│  ├─ single_agent/
│  │  ├─ train_ppo.py            # Train PPO agent
│  │  ├─ train_sac.py            # Train SAC agent
│  │  ├─ train_td3.py            # Train TD3 agent
│  │  └─ eval.py                 # Evaluate trained vs random
│  └─ multi_agent/
│     ├─ train_ppo_marl.py       # Shared-policy PPO for MARL
│     └─ eval.py                 # Multi-agent evaluation
├─ datasets/
│  ├─ energy_prices_2023_france.csv
│  └─ energy_prices_2024_france.csv
└─ trained_agents/               # Pretrained models (PPO, SAC, TD3)
```

---

##  Training & Evaluation

- Train PPO, SAC, or TD3 agents using Stable-Baselines3.  
- Evaluate trained models against random baselines.  
- Train shared-policy PPO in the multi-agent setup with PettingZoo + SuperSuit.  
- Compare policies with mean, median, standard deviation, and win-rate metrics.

---

##  Requirements

- `gymnasium`  
- `numpy`  
- `pandas`  
- `stable-baselines3`  
- `pettingzoo` *(for multi-agent)*  
- `supersuit` *(for PettingZoo → SB3 adapter)*

Install all dependencies with:

```
pip install -r requirements.txt
```
