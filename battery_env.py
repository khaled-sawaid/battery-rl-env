import gymnasium as gym
from gymnasium import spaces
import numpy as np

class BatteryEnv(gym.Env):

    def __init__(self,
                episode_length=48,
                step_hours=0.5,
                max_capacity=100.0,
                min_capacity=0.0,
                initial_capacity=0.5, # as a fraction of max_capacity
                charge_max=25.0,
                discharge_max=25.0,
                charge_efficiency=0.95,
                discharge_efficiency=0.95,
                continuous_action=True,
                price_series=None,
                seed=None):
        """
        Initialize the Battery Environment.

            Args:
                episode_length (int) = how many steps are in each episode 
                step_hours (float) = how many hours is each step
                max_capacity (float) = max amount a battery can store
                min_capacity (float) = min amount a battery can store
                initial_capacity (float) = a fraction of max_capacity which the battery starts with
                charge_max (float) = max charging rate 
                discharge_max (float) = max discharging rate
                charge_efficiency (float) = fraction of energy actually stored in the battery
                discharge_efficiency (float)= fraction of stored energy that actually reaches the grid
                continuous_action (bool) =
                    no -> full charge, full discharge, idle
                    yes -> action is continous in [-1, 1]
                price_series (array or None) = array of electricity prices (length = episode_length)
                seed (int or None) = for reproducobility
            """
        super().__init__()

        # --- Validate parameters --- 
        if not isinstance(episode_length, int) or episode_length <= 0:
            raise ValueError("episode_length must be a positive integer")
        if step_hours <= 0:
            raise ValueError("step_hours must be a positive number")
        if max_capacity <= 0:
            raise ValueError("max_capacity must be a positive number")
        if min_capacity < 0 or min_capacity >= max_capacity:
            raise ValueError("min_capacity must be non-negative and less than max_capacity")
        if not (0.0 <= initial_capacity <= 1.0):
            raise ValueError("initial_capacity must be in the range [0, 1]")
        if charge_max < 0 or discharge_max < 0:
            raise ValueError("charge_max and discharge_max must be >= 0")
        if not (0.0 < charge_efficiency <= 1.0):
            raise ValueError("charge_efficiency must be in (0, 1]")
        if not (0.0 < discharge_efficiency <= 1.0):
            raise ValueError("discharge_efficiency must be in (0, 1]")
        if price_series is not None:
            ps = np.asarray(price_series, dtype=np.float32)
            if ps.shape != (episode_length,):
                raise ValueError("price_series must have shape (episode_length,)")
            if np.any(ps < 0):
                raise ValueError("price_series must be non-negative")
            self.price_series = ps
        else:
            self.price_series = None
                
        # --- Store params ---
        self.episode_length         = episode_length
        self.step_hours             = float(step_hours)
        self.max_capacity           = float(max_capacity)
        self.min_capacity           = float(min_capacity)
        self.initial_capacity       = float(initial_capacity)
        self.charge_max             = float(charge_max)
        self.discharge_max          = float(discharge_max)
        self.charge_efficiency      = float(charge_efficiency)
        self.discharge_efficiency   = float(discharge_efficiency)
        self.continuous_action      = bool(continuous_action)

        # --- RNG ---
        self.np_random, self.seed = gym.utils.seeding.np_random(seed)

        # --- Action space ---
        # Continuous: action in [-1, 1], representing [-discharge_max, charging_max]
        if self.continuous_action:
            self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        # Discrete: {-1, 0, 1} mapped to full discharge / idle / full charge
        else:
            self.action_space = spaces.Discrete(3) 

        # --- Observation space ---

