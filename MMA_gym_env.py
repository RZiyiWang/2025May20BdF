import numpy as np
from gymnasium import Env, spaces
from MMA_market_env import MMAMarketEnvironment
from reward_utils import MMA_reward

class MMAGymEnv(Env):

    metadata = {"render_modes": ["human"]}

    def __init__(self,
                 adversary_model=None,
                 max_steps=200,
                 max_time=1.0,
                 inventory_bounds=(-500, 500),
                 zeta=0,
                 eta=0.01,
                 use_time_limit=False,
                 seed=None,
                 action_bounds=(0, 5)  # bid/ask offset bounds
                 ):
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

        self.env = MMAMarketEnvironment(
            adversary_model=adversary_model,
            seed=seed,
            inventory_bounds=inventory_bounds
        )

        # Observation: [price, time, inventory, cash]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, inventory_bounds[0], -np.inf], dtype=np.float32),
            high=np.array([np.inf, 1.0, inventory_bounds[1], np.inf], dtype=np.float32),
            dtype=np.float32
        )

        # Normalized action space [-1, 1] for bid_offset and ask_offset
        self.raw_action_low = np.array([action_bounds[0], action_bounds[0]], dtype=np.float32)
        self.raw_action_high = np.array([action_bounds[1], action_bounds[1]], dtype=np.float32)

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([ 1.0,  1.0], dtype=np.float32),
            dtype=np.float32
        )

    def _rescale_action(self, action: np.ndarray) -> np.ndarray:
        return self.raw_action_low + (action + 1.0) * 0.5 * (self.raw_action_high - self.raw_action_low)

    def reset(self, seed=None, options=None):
        self.cur_step = 0
        state = self.env.reset()
        self.prev_cash = state['cash_A']
        self.prev_inventory = state['inventory_A']
        self.prev_price = state['price']
        return self._obs_from_state(state), {}

    def step(self, action):
        action = self._rescale_action(action)  # [bid_offset, ask_offset]
        self.cur_step += 1
        state, total_arrivals = self.env.step_with_agent(action)

        pnl_prev = self.prev_cash + self.prev_inventory * self.prev_price
        pnl_curr = state['cash_A'] + state['inventory_A'] * state['price']
        pnl_diff = pnl_curr - pnl_prev

        self.prev_cash = state['cash_A']
        self.prev_inventory = state['inventory_A']
        self.prev_price = state['price']

        done = self.cur_step >= self.max_steps
        if self.use_time_limit:
            done = done or state['time'] >= self.max_time

        terminated = done
        truncated = False
        info = {
            "market_state": state,
            "total_arrivals": total_arrivals
        }

        reward = MMA_reward(
            pnl_diff,
            inventory=state['inventory_A'],
            final=done,
            max_inventory=self.max_inventory,
            zeta=self.zeta,
            eta=self.eta
        )

        return self._obs_from_state(state), reward, terminated, truncated, info

    def _obs_from_state(self, state):
        return np.array([
            state['price'],
            state['time'],
            state['inventory_A'],
            state['cash_A']
        ], dtype=np.float32)

    def render(self, mode="human"):
        return self.env.render(mode=mode)

    def close(self):
        self.env.close()
