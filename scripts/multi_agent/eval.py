# eval_marl.py
import os
import numpy as np
import pandas as pd

from envs.multi_agent import BatteryParallelEnv
from stable_baselines3 import PPO

# ------------- Data -------------
COL = "Day-ahead Price (EUR/MWh)"
data = pd.read_csv("datasets/energy_prices_2023_france.csv", usecols=[COL])
prices_mwh = pd.to_numeric(data[COL], errors="coerce").dropna()
prices_kwh = (prices_mwh / 1000.0).to_numpy(dtype=np.float32)

# ------------- Env -------------
SEED = 42
n_agents = 5
episode_length = 24
step_hours = 1.0

def make_env(start_idx=None):
    env = BatteryParallelEnv(
        n_agents=n_agents,
        episode_length=episode_length,
        step_hours=step_hours,
        price_series=prices_kwh,
        seed=SEED,
    )
    if start_idx is None:
        obs, info = env.reset()
    else:
        obs, info = env.reset(options={"start": int(start_idx)})
    return env, obs, info

# ------------- Load models if available -------------
models = {}
paths = {
    "PPO_shared": "ppo_battery_parallel_shared_policy",
}
for name, path in paths.items():
    if os.path.exists(path + ".zip"):
        # Model was trained through the SB3+SuperSuit VecEnv adapter,
        # but for eval we call the policy per-agent on raw observations.
        models[name] = PPO.load(path)
        print(f"Loaded {name} from {path}.zip")
    else:
        print(f"WARNING: {name} checkpoint not found at {path}.zip – skipping.")

# ------------- Policies -------------
def make_model_policy(model, deterministic=True):
    """
    Returns a function mapping obs_dict -> actions_dict for current live agents.
    Each agent gets its own forward pass (shared weights).
    """
    def _pol(obs_dict, live_agents, env):
        actions = {}
        for a in live_agents:
            obs = obs_dict[a]
            # SB3 .predict accepts unbatched obs; returns (action, state)
            act, _ = model.predict(obs, deterministic=deterministic)
            # Ensure correct dtype/shape for env (Box([-1,1], shape=(1,)))
            actions[a] = np.asarray(act, dtype=np.float32)
        return actions
    return _pol

def make_random_policy():
    def _rand(obs_dict, live_agents, env):
        return {a: env.action_space(a).sample() for a in live_agents}
    return _rand

policies = {}
for name, model in models.items():
    policies[name] = make_model_policy(model, deterministic=True)
policies["Random"] = make_random_policy()

# ------------- Eval helpers -------------
def run_one_episode(start_idx, policy_fn):
    env, obs, info = make_env(start_idx=start_idx)
    sum_rewards_per_agent = {a: 0.0 for a in env.agents}

    terminations = {a: False for a in env.agents}
    truncations  = {a: False for a in env.agents}

    while len(env.agents) > 0:
        live_agents = env.agents[:]  # order provided by env
        actions = policy_fn(obs, live_agents, env)

        obs, rewards, terminations, truncations, infos = env.step(actions)

        # rewards is a dict over *previous* live agents
        for a, r in rewards.items():
            sum_rewards_per_agent[a] = sum_rewards_per_agent.get(a, 0.0) + float(r)

        # loop continues until env.agents becomes [] (episode done)
    env.close()

    # Aggregate to team stats
    team_return = float(sum(sum_rewards_per_agent.values()))
    return team_return, sum_rewards_per_agent

def summarize(name, arr):
    arr = np.asarray(arr, dtype=np.float32)
    return f"{name:>12} | mean={arr.mean(): .4f} | median={np.median(arr): .4f} | std={arr.std(ddof=1): .4f} | min={arr.min(): .4f} | max={arr.max(): .4f}"

# ------------- Evaluation set (shared) -------------
rng = np.random.RandomState(SEED)
max_start = len(prices_kwh) - episode_length
N_EPISODES = 100
starts_pool = rng.choice(np.arange(0, max_start + 1), size=min(200, max_start + 1), replace=False)
starts = starts_pool[:N_EPISODES]

# ------------- Run eval -------------
team_returns = {name: [] for name in policies.keys()}
per_agent_returns = {name: [] for name in policies.keys()}  # list of dicts

for s in starts:
    for name, pol in policies.items():
        team_r, per_agent_r = run_one_episode(s, pol)
        team_returns[name].append(team_r)
        per_agent_returns[name].append(per_agent_r)

# ------------- Print summaries (team sum) -------------
print("\n=== Per-policy TEAM results (sum over agents) ===")
for name in sorted(team_returns.keys()):
    print(summarize(name, team_returns[name]))

# ------------- Per-agent summaries (averaged across episodes) -------------
print("\n=== Per-agent average returns (by policy) ===")
for name in sorted(per_agent_returns.keys()):
    # Collect per-agent arrays aligned by agent id
    agents = [f"battery_{i}" for i in range(n_agents)]
    avgs = []
    for a in agents:
        vals = [d.get(a, 0.0) for d in per_agent_returns[name]]
        avgs.append(np.mean(vals))
    line = " | ".join([f"{a}={v:.4f}" for a, v in zip(agents, avgs)])
    print(f"{name:>12} | {line}")

# ------------- Win-rates vs Random (team sum) -------------
if "Random" in team_returns:
    print("\n=== Win-rate vs Random (TEAM returns) ===")
    rnd = np.array(team_returns["Random"], dtype=np.float32)
    for name in sorted(team_returns.keys()):
        if name == "Random":
            continue
        arr = np.array(team_returns[name], dtype=np.float32)
        diff = arr - rnd
        win = (diff > 0).mean()
        print(f"{name:>12} beats Random in {win*100:.1f}% of episodes (avg margin {diff.mean():.4f}).")

# ------------- Pairwise win-rates among learned agents -------------
learned = [k for k in team_returns.keys() if k != "Random"]
if len(learned) >= 2:
    print("\n=== Pairwise win-rates (TEAM returns, learned agents) ===")
    for i in range(len(learned)):
        for j in range(i + 1, len(learned)):
            a, b = learned[i], learned[j]
            A = np.array(team_returns[a], dtype=np.float32)
            B = np.array(team_returns[b], dtype=np.float32)
            win = (A > B).mean()
            margin = (A - B).mean()
            print(f"{a:>12} vs {b:<12}: {win*100:5.1f}% wins (avg margin {margin:.4f})")

# ------------- Sample lines -------------
print("\n=== First few episodes (TEAM returns) ===")
K = min(5, N_EPISODES)
for i in range(K):
    line = [f"start={starts[i]:5d}"]
    for name in sorted(team_returns.keys()):
        line.append(f"{name}={team_returns[name][i]:.4f}")
    print(" | ".join(line))
