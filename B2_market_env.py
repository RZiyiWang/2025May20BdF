import numpy as np
from stable_baselines3 import PPO
from market_environment import MarketEnvironment
import torch

class B2MarketEnv(MarketEnvironment):


    def __init__(self,
                 adversary_model=None,
                 agent_A=None,
                 seed=None,
                 max_fill_per_direction=3,
                 dt=0.005,
                 lam_bounds=(300, 500),
                 volatility_bounds=(0.2, 2.0),
                 inventory_bounds=(-500, 500)):

        super().__init__(seed=seed, max_fill_per_direction=max_fill_per_direction, dt=dt)

        self.lam_bounds = lam_bounds
        self.volatility_bounds = volatility_bounds
        self.inventory_bounds = inventory_bounds
        self.fixed_lam = 400.0
        self.fixed_volatility = 1.1

        self.inventory_A = 0.0
        self.inventory_B2 = 0.0
        self.cash_A = 0.0
        self.cash_B2 = 0.0
        self.pnl_A = 0.0
        self.pnl_B2 = 0.0

        if adversary_model:
            self.adversary_model = PPO.load(adversary_model)
            print("Adversary model loaded. Using strategic market parameters.")
        else:
            self.adversary_model = None
            self.arrival_model.lam = self.fixed_lam
            self.midprice_model.volatility = self.fixed_volatility
            print(f"No adversary model loaded. Using fixed λ={self.fixed_lam}, volatility={self.fixed_volatility}")

        if agent_A:
            self.agent_A = PPO.load(agent_A)
            print("Loaded pre-trained Agent A. Initiating B2 training via opponent modeling.")
        else:
            self.agent_A = None
            print("No Agent A loaded. Running in single-agent mode.")

    def reset(self):
        super().reset()
        self.inventory_A = 0.0
        self.inventory_B2 = 0.0
        self.cash_A = 0.0
        self.cash_B2 = 0.0
        self.pnl_A = 0.0
        self.pnl_B2 = 0.0
        return self.get_state()

    def step_with_agent_B(self, action_B2):
        if self.adversary_model:
            Adversary_obs = np.array([self.price, self.time], dtype=np.float32).reshape(1, -1)
            with torch.no_grad():
                adv_action = self.adversary_model.predict(Adversary_obs, deterministic=False)[0]
            adv_action = np.array(adv_action).flatten()
            # Convert the normalized action back to the quote offset range
            lam = self.lam_bounds[0] + (adv_action[0] + 1.0) * 0.5 * (self.lam_bounds[1] - self.lam_bounds[0])
            vol = self.volatility_bounds[0] + (adv_action[1] + 1.0) * 0.5 * (self.volatility_bounds[1] - self.volatility_bounds[0])
            self.arrival_model.lam = np.clip(lam, *self.lam_bounds)
            self.midprice_model.volatility = np.clip(vol, *self.volatility_bounds)

        # === Generate quotes for Agent A ===
        A_obs = np.array([self.price, self.time, self.inventory_A, self.cash_A], dtype=np.float32).reshape(1, -1)
        with torch.no_grad():
            action_A = self.agent_A.predict(A_obs, deterministic=False)[0]
        action_A = np.array(action_A).flatten()  # Force shape to be (2,)
        # Denormalize action to quote offset range (action_bounds = (0, 5))
        action_A = 2.5 * (action_A + 1.0)  


        # Quotes from the loaded pre-trained Agent A
        A_bid_offset = float(action_A[0])
        A_ask_offset = float(action_A[1])
        A_bid_price = self.price - A_bid_offset
        A_ask_price = self.price + A_ask_offset

         # Agent B2's quote spread is set by the agent's action for easier control; price is calculated later
        B2_bid_offset = float(action_B2[0])
        B2_ask_offset = float(action_B2[1])
        B2_bid_price = self.price - B2_bid_offset
        B2_ask_price = self.price + B2_ask_offset


        # 构造 agent_quotes
        agent_quotes = {
            'MM_A': {'bid_price': A_bid_price, 'ask_price': A_ask_price},
            'MM_B2': {'bid_price': B2_bid_price, 'ask_price': B2_ask_price}
        }

        # Limit the maximum trade volume before matching
        A_max_bid_fill = max(0, self.inventory_bounds[1] - self.inventory_A)
        A_max_ask_fill = max(0, self.inventory_A - self.inventory_bounds[0])
        B2_max_bid_fill = max(0, self.inventory_bounds[1] - self.inventory_B2)
        B2_max_ask_fill = max(0, self.inventory_B2 - self.inventory_bounds[0])

        self.max_fill_override = {
            'MM_A': {'max_bid_fill': A_max_bid_fill,'max_ask_fill': A_max_ask_fill},
            'MM_B2': {'max_bid_fill': B2_max_bid_fill,'max_ask_fill': B2_max_ask_fill}
        }

        # matching
        market_state, total_arrivals, agent_fills = super().step(agent_quotes)

        A_fill = agent_fills.get('MM_A', {})
        A_bid_fill = min(A_fill.get('bid_fills', 0), A_max_bid_fill)
        A_ask_fill = min(A_fill.get('ask_fills', 0), A_max_ask_fill)
        A_bid_exec_price = A_fill.get('bid_price', A_bid_price)
        A_ask_exec_price = A_fill.get('ask_price', A_ask_price)

        B2_fill = agent_fills.get('MM_B2', {})
        B2_bid_fill = min(B2_fill.get('bid_fills', 0), B2_max_bid_fill)
        B2_ask_fill = min(B2_fill.get('ask_fills', 0), B2_max_ask_fill)
        B2_bid_exec_price = B2_fill.get('bid_price', B2_bid_price)
        B2_ask_exec_price = B2_fill.get('ask_price', B2_ask_price)

        # Update A
        self.inventory_A += A_bid_fill - A_ask_fill
        self.cash_A += A_ask_fill * A_ask_exec_price - A_bid_fill * A_bid_exec_price
        self.pnl_A = self.cash_A + self.inventory_A * self.price 

        # Update B2
        self.inventory_B2 += B2_bid_fill - B2_ask_fill
        self.cash_B2 += B2_ask_fill * B2_ask_exec_price - B2_bid_fill * B2_bid_exec_price
        self.pnl_B2 = self.cash_B2 + self.inventory_B2 * self.price 

        # === Agent A ===
        self.bid_price_A = A_bid_price
        self.ask_price_A = A_ask_price
        self.bid_fill_A = A_bid_fill
        self.ask_fill_A = A_ask_fill


        # === Agent B2 ===
        self.bid_price_B2 = B2_bid_price
        self.ask_price_B2 = B2_ask_price
        self.bid_fill_B2 = B2_bid_fill
        self.ask_fill_B2 = B2_ask_fill

        # Return the states of Agent A and B2 to the market
        market_state = self.get_state()
        market_state.update({
            'bid_price_A': self.bid_price_A,
            'ask_price_A': self.ask_price_A,
            'bid_fill_A': self.bid_fill_A,
            'ask_fill_A': self.ask_fill_A,
            'inventory_A': self.inventory_A,
            'cash_A': self.cash_A,
            'pnl_A': self.pnl_A,

            'bid_price_B2': self.bid_price_B2,
            'ask_price_B2': self.ask_price_B2,
            'bid_fill_B2': self.bid_fill_B2,
            'ask_fill_B2': self.ask_fill_B2,
            'inventory_B2': self.inventory_B2,
            'cash_B2': self.cash_B2,
            'pnl_B2': self.pnl_B2,

            'lam': self.arrival_model.lam,
            'volatility': self.midprice_model.volatility

        })

        return market_state, total_arrivals