import gymnasium as gym
from gymnasium import spaces
import numpy as np

class BatteryEnv(gym.Env):

    def __init__(self,
                episode_length=48,
                step_hours=0.5,
                max_capacity_kwh=100.0,
                min_capacity_kwh=0.0,
                max_charge_rate_kw=25.0,
                max_discharge_rate_kw=25.0,
                initial_soc_frac=0.5, 
                charge_efficiency=0.95,
                discharge_efficiency=0.95,
                continuous_action=True,
                price_series=None,
                seed=None,):
        """
        Initializes the BatteryEnv.

        Args:
            episode_length: Number of steps per episode.
            step_hours: Hours per step.
            max_capacity_kwh: Max state of charge [kWh].
            min_capacity_kwh: Min state of charge [kWh].
            max_charge_rate_kw: Max charging power [kW].
            max_discharge_rate_kw: Max discharging power [kW].
            initial_soc_frac: Initial SoC as a fraction of max capacity.
            charge_efficiency: One-way charging efficiency (0-1].
            discharge_efficiency: One-way discharging efficiency (0-1].
            continuous_action: If True, actions are continuous; otherwise discrete.
            price_series: per-step price [currency/kWh].
            seed: Optional random seed for reproducibility.

        Notes:
            - Units: power=kW, energy=kWh, time=hours.
        """
        super().__init__()
        
        # --- Validate params ---
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
        if np.any(ps < 0):
            raise ValueError("price_series must contain only non-negative values")
        
        # --- Store params ---
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

        # --- RNG --- 
        self.np_random, self.seed = gym.utils.seeding.np_random(seed)

        # --- Action space ---
        if self.continuous_action:
            self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(3)

        # --- Observation space ---
        self._max_price = float(ps.max())
        self._min_price = float(ps.min())
        # obs = [battery pct, price of energy]
        self.observation_space = spaces.Box(
            low=np.array([0.0, self._min_price], dtype=np.float32),
            high=np.array([1.0, self._max_price], dtype=np.float32),
            dtype=np.float32
        )

        # --- Episode state ---
        self.current_step = None
        self.current_price = None
        self.battery_pct = None
        self.cumulative_profit = None 
        self._index = None
        self._start = None



    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.np_random, self.seed = gym.utils.seeding.np_random(seed)


        if options is not None and "start" in options:
            start = int(options["start"])
            if start < 0 or start + self.episode_length > self.series_len:
                raise ValueError("Invalid options['start'] for this episode_length.")
            self._start = start
        else:
            self._start = int(self.np_random.integers(0, self.series_len - self.episode_length + 1))
        
        self._index = self._start

        self.current_step = 0
        self.current_price = float(self.price_series[self._index])
        self.cumulative_profit = 0.0


        self.energy_kwh = self.initial_soc_frac * self.max_capacity_kwh
        self.battery_pct = self.energy_kwh / self.max_capacity_kwh

        obs = np.array([self.battery_pct, self.current_price], dtype=np.float32)
    
        
        return obs, {}


    def step(self, action):
        price = self.current_price
        t = self.current_step
        dt = self.step_hours

        revenue = 0.0
        cost = 0.0

        if self.continuous_action:
            a = float(np.clip(np.asarray(action).item() if isinstance(action, (list, np.ndarray, np.generic)) else action, -1.0, 1.0))
            power_kw = a * (self.max_charge_rate_kw if a >= 0.0 else self.max_discharge_rate_kw)
        else:
            a = int(action)
            if a not in (0, 1, 2):
                raise ValueError("Discrete action must be one of {0,1,2}.")
            power_kw = {0: -self.max_discharge_rate_kw, 1: 0.0, 2: self.max_charge_rate_kw}[a]  
        

        if power_kw >= 0:
            # CHARGING
            requested_energy_kwh = power_kw * dt
            stored_kwh = requested_energy_kwh * self.charge_efficiency
            room_kwh = self.max_capacity_kwh - self.energy_kwh
            if room_kwh < stored_kwh:
                stored_kwh = room_kwh
                requested_energy_kwh = stored_kwh / self.charge_efficiency
            self.energy_kwh += stored_kwh
            cost = requested_energy_kwh * price
        else:
            # DISCHARGING
            discharge_kw = -power_kw
            requested_batt_out_kwh = discharge_kw * dt
            avail_energy_kwh = max(0.0, self.energy_kwh - self.min_capacity_kwh)
            batt_out_kwh = min(avail_energy_kwh, requested_batt_out_kwh)
            energy_sold_kwh = batt_out_kwh * self.discharge_efficiency
            self.energy_kwh -= batt_out_kwh
            revenue += energy_sold_kwh * price 
        
        self.battery_pct = self.energy_kwh / self.max_capacity_kwh
        reward = float(revenue - cost) 
        self.cumulative_profit += reward


        self.current_step += 1
        terminated = self.current_step >= self.episode_length
        truncated = False

        if not terminated:
            self._index += 1
            self.current_price = float(self.price_series[self   ._index])
            obs = np.array([self.battery_pct, self.current_price], dtype=np.float32)
        else:

            obs = np.array([self.battery_pct, price], dtype=np.float32)
        
        info = {
            "cumulative_profit": float(self.cumulative_profit),
            "price": float(price),
            "soc_kwh": float(self.energy_kwh),
            "battery_pct": float(self.battery_pct),
            "start_index": int(self._start),
            "abs_index": int(self._index if not terminated else self._index - 1),
        }
        
        return obs, reward, terminated, truncated, info