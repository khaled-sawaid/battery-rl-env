import pandas as pd
import numpy as np
from battery_env import BatteryEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO


COL = "Day-ahead Price (EUR/MWh)"
data = pd.read_csv("datasets/energy_prices_2023_france.csv", usecols=[COL])
prices_mwh = pd.to_numeric(data[COL], errors="coerce")
prices_mwh = prices_mwh.dropna()
prices_kwh = (prices_mwh / 1000.0).to_numpy(dtype=np.float32)

SEED = 42
episode_length = 24
step_hours = 1.

base_env = BatteryEnv(price_series=prices_kwh, seed=SEED, episode_length=episode_length, step_hours=step_hours)
env = Monitor(base_env)

model = PPO.load("ppo_battery_env", env=env)

# --- Helpers ---
def run_one_episode(env, policy_fn, start_idx=None):
    if start_idx is None:
        obs, info = env.reset(seed=None)
    else:
        obs, info = env.reset(options={"start": int(start_idx)})
    
    terminated = False
    truncated = False
    ep_return = 0.0
    while not (terminated or truncated):
        action = policy_fn(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_return += float(reward)
    return ep_return, info  

def ppo_policy(obs):
    action, _ = model.predict(obs, deterministic=True)
    return action

def random_policy_factory(action_space):
    def _random(_obs):
        return action_space.sample()
    return _random


# --- Evalutaion ---
rng = np.random.RandomState(SEED)
max_start = len(prices_kwh) - episode_length
valid_starts = rng.choice(np.arange(0, max_start+1), size=200, replace=False)

N_EPISODES = 100
starts = valid_starts[:N_EPISODES]

ppo_returns = []
random_returns = []
random_policy = random_policy_factory(env.action_space)

for s in starts:
    r_ppo, info_ppo = run_one_episode(env, ppo_policy, start_idx=s)
    ppo_returns.append(r_ppo)

    r_rnd, info_rnd = run_one_episode(env, random_policy, start_idx=s)
    random_returns.append(r_rnd)

ppo_returns = np.array(ppo_returns, dtype=np.float32)
random_returns = np.array(random_returns, dtype=np.float32)

def summary(name, arr):
    return (
        f"{name:>8} | mean={arr.mean(): .4f} | median={np.median(arr): .4f} | "
        f"std={arr.std(ddof=1): .4f} | min={arr.min(): .4f} | max={arr.max(): .4f}"
    )

print(summary("PPO", ppo_returns))
print(summary("Random", random_returns))

diff = ppo_returns - random_returns
win_rate = (diff > 0).mean()
print(f"Head-to-head: PPO beats Random on {win_rate*100:.1f}% of episodes "
      f"(avg margin {diff.mean():.4f} per episode).")


for i in range(min(5, N_EPISODES)):
    print(f"Episode {i} start={starts[i]} | PPO={ppo_returns[i]:.4f} | Random={random_returns[i]:.4f}")

