import numpy as np
import gymnasium as gym
from gymnasium import spaces
from adversary_market_env import AdversaryMarketEnvironment
from reward_utils import Adversary_reward

class AdversaryEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self,
                 max_steps=200,
                 max_time=1.0,
                 lam_bounds = (300, 500),
                 volatility_bounds=(0.2, 2.0),
                 inventory_bounds = (-500, 500),
                 zeta=0,
                 eta=0.01,
                 use_time_limit=False,
                 seed=None):
        super().__init__()

        self.max_steps = max_steps
        self.max_time = max_time
        self.use_time_limit = use_time_limit
        self.cur_step = 0
        self.zeta = zeta
        self.eta = eta
        self.max_inventory = inventory_bounds[1]
        self.prev_cash = 0.0
        self.prev_inventory = 0.0
        self.prev_price = 0.0

        self.env = AdversaryMarketEnvironment(
            strategy_params={"gamma": 1, "eta": self.eta},
            seed=seed,
            inventory_bounds=inventory_bounds,
            lam_bounds=lam_bounds,
            volatility_bounds=volatility_bounds,
        )
        
        # Observation: [price, time]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([np.inf, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # Scaled action space for SB3 (-1 to 1)
        self.raw_action_low = np.array([lam_bounds[0], volatility_bounds[0]], dtype=np.float32)
        self.raw_action_high = np.array([lam_bounds[1], volatility_bounds[1]], dtype=np.float32)
        self.action_space = spaces.Box(
                    low  = np.array([-1.0, -1.0], dtype=np.float32),
                    high = np.array([ 1.0,  1.0], dtype=np.float32),
                    dtype=np.float32
                )



    def _rescale_action(self, action: np.ndarray) -> np.ndarray:
        return self.raw_action_low + (action + 1.0) * 0.5 * (self.raw_action_high - self.raw_action_low)

    def reset(self,*, seed=None, options=None):
        self.cur_step = 0
        state= self.env.reset()
        self.prev_cash      = state['cash_Adversary']
        self.prev_inventory = state['inventory_Adversary']
        self.prev_price     = state['price']
        return self._obs_from_state(state),{}

    def step(self, action):
        Adversary_action = self._rescale_action(action)
        self.cur_step += 1
        action_dict = {'lam': float(Adversary_action[0]), 'volatility': float(Adversary_action[1])}
        state, total_arrivals = self.env.step_with_adversary(action_dict)

        pnl_prev = self.prev_cash + self.prev_inventory * self.prev_price
        pnl_curr = state['cash_Adversary'] + state['inventory_Adversary'] * state['price']
        pnl_diff = pnl_curr - pnl_prev

        self.prev_cash = state['cash_Adversary']
        self.prev_inventory = state['inventory_Adversary']
        self.prev_price = state['price']

        done = self.cur_step >= self.max_steps
        if self.use_time_limit:
            done = done or state['time'] >= self.max_time

        terminated = done
        truncated  = False
        info = {
            "market_state": state,
            "total_arrivals": total_arrivals
        }

        reward = Adversary_reward(
            pnl_diff,
            inventory=state['inventory_Adversary'],
            final=done,
            max_inventory=self.max_inventory,
            zeta=self.zeta,
            eta=self.eta
        )

        
        return self._obs_from_state(state), reward, terminated, truncated, info
    

    def _obs_from_state(self, state):
        return np.array([
            state['price'],
            state['time']
        ], dtype=np.float32)

    def render(self, mode="human"):
        return self.env.render(mode=mode)

    def close(self):
        self.env.close()

