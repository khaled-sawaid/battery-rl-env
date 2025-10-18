import os
import pandas as pd
import numpy as np
import gymnasium as gym
from envs.single_agent import BatteryEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO, TD3, SAC

# ------------- Data -------------
COL = "Day-ahead Price (EUR/MWh)"
data = pd.read_csv("datasets/energy_prices_2023_france.csv", usecols=[COL])
prices_mwh = pd.to_numeric(data[COL], errors="coerce").dropna()
prices_kwh = (prices_mwh / 1000.0).to_numpy(dtype=np.float32)

# ------------- Env -------------
SEED = 42
episode_length = 24
step_hours = 1.0

base_env = BatteryEnv(price_series=prices_kwh, seed=SEED, episode_length=episode_length, step_hours=step_hours)
env = Monitor(base_env)

# ------------- Load models if available -------------
models = {}
paths = {
    "PPO": "trained_agents/ppo_battery_env",
    "TD3": "trained_agents/td3_battery_env",
    "SAC": "trained_agents/sac_battery_env",
}
loaders = {"PPO": PPO, "TD3": TD3, "SAC": SAC}

for name, path in paths.items():
    if os.path.exists(path + ".zip"):
        models[name] = loaders[name].load(path, env=env)
        print(f"Loaded {name} from {path}.zip")
    else:
        print(f"WARNING: {name} checkpoint not found at {path}.zip – skipping.")

# ------------- Policies -------------
def make_model_policy(model, deterministic=True):
    def _pol(obs):
        action, _ = model.predict(obs, deterministic=deterministic)
        return action
    return _pol

def make_random_policy(action_space):
    def _rand(_obs):
        return action_space.sample()
    return _rand

policies = {}
for name, model in models.items():
    # Deterministic for evaluation (SAC/TD3 mean action; PPO greedy)
    policies[name] = make_model_policy(model, deterministic=True)
policies["Random"] = make_random_policy(env.action_space)

# ------------- Eval helpers -------------
def run_one_episode(env, policy_fn, start_idx=None):
    if start_idx is None:
        obs, info = env.reset(seed=None)
    else:
        obs, info = env.reset(options={"start": int(start_idx)})
    ep_return = 0.0
    terminated = truncated = False
    while not (terminated or truncated):
        action = policy_fn(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_return += float(reward)
    return ep_return, info

def summarize(name, arr):
    arr = np.asarray(arr, dtype=np.float32)
    return f"{name:>8} | mean={arr.mean(): .4f} | median={np.median(arr): .4f} | std={arr.std(ddof=1): .4f} | min={arr.min(): .4f} | max={arr.max(): .4f}"

# ------------- Evaluation set (shared) -------------
rng = np.random.RandomState(SEED)
max_start = len(prices_kwh) - episode_length
N_EPISODES = 100
starts_pool = rng.choice(np.arange(0, max_start + 1), size=200, replace=False)
starts = starts_pool[:N_EPISODES]

# ------------- Run eval -------------
returns = {name: [] for name in policies.keys()}

for s in starts:
    for name, pol in policies.items():
        r, _ = run_one_episode(env, pol, start_idx=s)
        returns[name].append(r)

# ------------- Print summaries -------------
print("\n=== Per-policy results ===")
for name in sorted(returns.keys()):
    print(summarize(name, returns[name]))

# ------------- Win-rates vs Random -------------
if "Random" in returns:
    print("\n=== Win-rate vs Random ===")
    rnd = np.array(returns["Random"], dtype=np.float32)
    for name in sorted(returns.keys()):
        if name == "Random":
            continue
        arr = np.array(returns[name], dtype=np.float32)
        diff = arr - rnd
        win = (diff > 0).mean()
        print(f"{name:>8} beats Random in {win*100:.1f}% of episodes (avg margin {diff.mean():.4f}).")

# ------------- Pairwise win-rates among learned agents -------------
learned = [k for k in returns.keys() if k != "Random"]
if len(learned) >= 2:
    print("\n=== Pairwise win-rates (learned agents) ===")
    for i in range(len(learned)):
        for j in range(i + 1, len(learned)):
            a, b = learned[i], learned[j]
            A = np.array(returns[a], dtype=np.float32)
            B = np.array(returns[b], dtype=np.float32)
            win = (A > B).mean()
            margin = (A - B).mean()
            print(f"{a:>8} vs {b:<8}: {win*100:5.1f}% wins (avg margin {margin:.4f})")

# ------------- Sample lines -------------
print("\n=== First few episodes ===")
K = min(5, N_EPISODES)
for i in range(K):
    line = [f"start={starts[i]:5d}"]
    for name in sorted(returns.keys()):
        line.append(f"{name}={returns[name][i]:.4f}")
    print(" | ".join(line))
