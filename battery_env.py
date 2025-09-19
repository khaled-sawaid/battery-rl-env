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
            episode_length = how mant steps are in each episode
            step_hours = how many hours is each step
            max
        
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
        if initial_capacity < 0 or initial_capacity > 1:
            raise ValueError("initial_capacity must be in the range [0, 1]")

