import numpy as np
from gymnasium import spaces
from pettingzoo.parallel import ParallelEnv

class BatteryParallelEnv(ParallelEnv):
    def __init__(self,
                n_agents=5,
                episode_length=24,
                step_hours=1.0,
                max_capacity_kwh=100.0,
                min_capacity_kwh=0.0,
                max_charge_rate_kw=25.0,
                max_discharge_rate_kw=25.0,
                initial_soc_frac=0.5, 
                charge_efficiency=0.95,
                discharge_efficiency=0.95,
                continuous_action=True,
                price_series=None,
                seed=None):
        super().__init__()

        if not isinstance(n_agents, int) or n_agents <= 0:
            raise ValueError("n_agents must be a positive integer")
        if not isinstance(episode_length, int) or episode_length <= 0:
            raise ValueError("episode_length must be a positive integer")
        if step_hours <= 0:
            raise ValueError("step_hours must be a positive number")
        if max_capacity_kwh <= 0:
            raise ValueError("max_capacity_kwh must be a positive number")
        if min_capacity_kwh < 0 or min_capacity_kwh >= max_capacity_kwh:
            raise ValueError("min_capacity_kwh must be a non-negative number and less than max_capacity_kwh")
        if max_charge_rate_kw <= 0 or max_discharge_rate_kw <= 0:
            raise ValueError("max_charge_rate_kw and max_discharge_rate_kw must be positive")
        if not (0.0 <= initial_soc_frac <= 1.0):
            raise ValueError("initial_soc_frac must be in the range [0, 1]")
        if not(0.0 < charge_efficiency <= 1.0):
            raise ValueError("charge_efficiency must be in the range (0,1]")
        if not(0.0 < discharge_efficiency <= 1.0):
            raise ValueError("discharge_efficiency must be in the range (0,1]")

        if price_series is None:
            raise ValueError("price_series must be provided: a 1-D list of length bigger or equal to episode_length")
        ps = np.asarray(price_series, dtype=np.float32)
        if ps.ndim != 1:
            raise ValueError("price_series must be 1-D")
        if ps.size < episode_length:
            raise ValueError("len(price_series) must be >= episode_length")
        if np.isnan(ps).any():
            raise ValueError("price_series contains NaNs; clean or fill before use")
        

        self.n_agents               = int(n_agents)
        self.possible_agents        = [f"battery_{i}" for i in range(self.n_agents)]
        self.agents                 = []
    
        self.episode_length         = episode_length
        self.step_hours             = float(step_hours)
        self.max_capacity_kwh       = float(max_capacity_kwh)
        self.min_capacity_kwh       = float(min_capacity_kwh)   
        self.max_charge_rate_kw     = float(max_charge_rate_kw)
        self.max_discharge_rate_kw  = float(max_discharge_rate_kw) 
        self.initial_soc_frac       = float(initial_soc_frac)  
        self.charge_efficiency      = float(charge_efficiency)
        self.discharge_efficiency   = float(discharge_efficiency)           
        self.continuous_action      = bool(continuous_action)     
        self.price_series           = ps
        self.series_len             = int(ps.size)

        self._rng = np.random.default_rng(seed)
        self._seed_value = seed

        if self.continuous_action:
            self._action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        else:
            self._action_space = spaces.Discrete(3)

        self._min_price = -10000.0
        self._max_price =  10000.0
        self._observation_space = spaces.Box(
            low=np.array([0.0, self._min_price], dtype=np.float32),
            high=np.array([1.0, self._max_price], dtype=np.float32),
            dtype=np.float32
        )

        self.current_step = None
        self._start_index = None
        self._index       = None
        self.current_price= None

        self.energy_kwh = None
        self.battery_pct = None
        self.cumulative_profit = None


    def observation_space(self, agent):
        return self._observation_space
    
    def action_space(self, agent):
        return self._action_space
    

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self._seed_value = seed
        
        if options is not None and "start" in options:
            start = int(options["start"])
            if start < 0 or start + self.episode_length > self.series_len:
                raise ValueError("Invalid options['start'] for this episode_length.")
            self._start_index = start
        else:
            self._start_index = int(self._rng.integers(0, self.series_len - self.episode_length + 1))
        

        self._index = self._start_index
        self.current_price = float(self.price_series[self._start_index])
        self.current_step = 0

        self.agents = self.possible_agents[:]

        init_energy = np.clip(self.initial_soc_frac * self.max_capacity_kwh, self.min_capacity_kwh, self.max_capacity_kwh)
        self.energy_kwh = np.full(self.n_agents, init_energy, dtype=np.float32)
        self.battery_pct = self.energy_kwh / self.max_capacity_kwh
        self.cumulative_profit = np.zeros(self.n_agents, dtype=np.float32)


        observations = {
            a: np.array([self.battery_pct[i], self.current_price], dtype=np.float32)
            for i, a in enumerate(self.agents)
        }

        info = {
            a: {
                "start_index": int(self._start_index),
                "abs_index": int(self._index),
                "battery_pct": float(self.battery_pct[i])
            } for i, a in enumerate(self.agents)
        }

        return observations, info
    

    def step(self, actions):
        if len(self.agents) == 0:
            return {}, {}, {}, {}, {}

        price = self.current_price
        dt = self.step_hours

        rewards = {a: 0.0 for a in self.agents}
        terminations = {a: False for a in self.agents}
        truncations  = {a: False for a in self.agents}
        infos = {}

        power_kw = np.zeros(self.n_agents, dtype=np.float32)

        for i, a in enumerate(self.agents):
            act = actions.get(a, None)

            if act is None:
                raise ValueError(f"Missing action for agent '{a}'")
            
            if self.continuous_action:
                val = float(np.asarray(act).item() if isinstance(act, (list, np.ndarray, np.generic)) else act)
                val = np.clip(val, -1.0, 1.0)
                power_kw[i] = val * (self.max_charge_rate_kw if val >= 0.0 else self.max_discharge_rate_kw)
            else:
                d = int(act)
                if d not in (0, 1 , 2):
                    raise ValueError("Discrete action must be one of {0,1,2}.")
                power_kw[i] =  {0:  -self.max_discharge_rate_kw, 1: 0.0, 2: self.max_charge_rate_kw}[d]


        revenue = np.zeros(self.n_agents, dtype=np.float32)
        cost    = np.zeros(self.n_agents, dtype=np.float32)

        charge_mask = power_kw >= 0
        if np.any(charge_mask):
            requested_energy_kwh = power_kw[charge_mask] * dt
            stored_kwh = requested_energy_kwh * self.charge_efficiency
            room_kwh =  (self.max_capacity_kwh - self.energy_kwh[charge_mask]).clip(min=0.0)
            stored_kwh = np.minimum(stored_kwh, room_kwh)
            requested_energy_kwh = stored_kwh / self.charge_efficiency
            self.energy_kwh[charge_mask] += stored_kwh
            cost[charge_mask] += requested_energy_kwh * price
        
        discharge_mask = power_kw < 0
        if np.any(discharge_mask):
            discharge_kw = (-power_kw[discharge_mask]).astype(np.float32)
            requested_batt_out_kwh = discharge_kw * dt
            avail_energy_kwh = (self.energy_kwh[discharge_mask] - self.min_capacity_kwh).clip(min=0.0)
            batt_out_kwh = np.minimum(avail_energy_kwh, requested_batt_out_kwh)
            energy_sold_kwh = batt_out_kwh * self.discharge_efficiency
            self.energy_kwh[discharge_mask] -= batt_out_kwh
            revenue[discharge_mask] = energy_sold_kwh * price

        
        self.battery_pct = self.energy_kwh / self.max_capacity_kwh
        step_reward = (revenue - cost).astype(np.float32)
        self.cumulative_profit += step_reward

        for i, a in enumerate(self.agents):
            rewards[a] = float(step_reward[i])
            infos[a] = {
                "cumulative_profit": float(self.cumulative_profit[i]),
                "price": float(price),
                "soc_kwh": float(self.energy_kwh[i]),
                "battery_pct": float(self.battery_pct[i]),
                "start_index": int(self._start_index),
                "abs_index": int(self._index),
            }
        
        self.current_step += 1
        done = self.current_step >= self.episode_length

        if not done:
            self._index += 1
            self.current_price = float(self.price_series[self._index])
            observations = {
                a: np.array([self.battery_pct[i], self.current_price], dtype=np.float32)
                for i, a in enumerate(self.agents)
            }
        else:
            for a in terminations:
                terminations[a] = True
            observations = {
                a: np.array([self.battery_pct[i], price], dtype=np.float32)
                for i, a in enumerate(self.agents)
            }
            self.agents = []
        
        return observations, rewards, terminations, truncations, infos 


    def render(self):
        pass

    def close(self):
        pass

   