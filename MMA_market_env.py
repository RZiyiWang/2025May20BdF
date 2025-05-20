import numpy as np
from stable_baselines3 import PPO
from market_environment import MarketEnvironment
import torch


class MMAMarketEnvironment(MarketEnvironment):

    def __init__(self,
                 adversary_model=None,
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

        self.inventory = 0.0
        self.cash = 0.0
        self.pnl = 0.0


        if adversary_model is not None:
            self.adversary_model = PPO.load(adversary_model)
            print("Adversary model loaded. Using strategic market parameters.")
        else:
            self.adversary_model = None 
            self.arrival_model.lam = self.fixed_lam
            self.midprice_model.volatility = self.fixed_volatility
            print("No adversary model loaded. Market operates under default dynamics.Using fixed market parameters: "
                  f"λ = {self.fixed_lam}, volatility = {self.fixed_volatility}")


    def reset(self):
        super().reset()
        self.inventory = 0.0
        self.cash = 0.0
        self.pnl = 0.0
        return self.get_state()

    def step_with_agent(self, agent_action):
        """
        agent_action: [bid_offset, ask_offset] → offset, it's easier to control action space
        agent_quotes: {'MM_A': {'bid_price': bid_price, 'ask_price': ask_price}} → quoting price
        """
        # λ and σ are controlled by the trained adversary if available; otherwise default to λ = 400.0 and σ = 1.1
        if self.adversary_model:
            Adversary_obs = np.array([self.price, self.time], dtype=np.float32).reshape(1, -1)
            with torch.no_grad():
                adv_action = self.adversary_model.predict(Adversary_obs, deterministic=False)[0]
            adv_action = np.array(adv_action).flatten()
            lam = self.lam_bounds[0] + (adv_action[0] + 1.0) * 0.5 * (self.lam_bounds[1] - self.lam_bounds[0])
            volatility = self.volatility_bounds[0] + (adv_action[1] + 1.0) * 0.5 * (self.volatility_bounds[1] - self.volatility_bounds[0])

            self.arrival_model.lam = np.clip(lam, *self.lam_bounds)
            self.midprice_model.volatility = np.clip(volatility, *self.volatility_bounds)

        bid_offset = float(agent_action[0])
        ask_offset = float(agent_action[1])
        bid_price = self.price - bid_offset
        ask_price = self.price + ask_offset

       
        bid_offset = float(agent_action[0])
        ask_offset = float(agent_action[1])
        bid_price = self.price - bid_offset
        ask_price = self.price + ask_offset

        
        agent_quotes = {'MM_A': {'bid_price': bid_price, 'ask_price': ask_price}}

        # Limit the maximum trade volume before matching
        max_bid_fill = max(0, self.inventory_bounds[1] - self.inventory)
        max_ask_fill = max(0, self.inventory - self.inventory_bounds[0])
        self.max_fill_override = {
            'MM_A': {
                'max_bid_fill': max_bid_fill,
                'max_ask_fill': max_ask_fill
            }
        }
        
        # matching
        market_state, total_arrivals, agent_fills = super().step(agent_quotes)

        fills = agent_fills.get('MM_A', {})
        bid_fill = min(fills.get('bid_fills', 0), max_bid_fill)
        ask_fill = min(fills.get('ask_fills', 0), max_ask_fill)
        bid_exec_price = fills.get('bid_price', bid_price)
        ask_exec_price = fills.get('ask_price', ask_price)

        # # Updates after trade execution
        self.inventory += bid_fill - ask_fill
        self.cash += ask_fill * ask_exec_price - bid_fill * bid_exec_price
        self.pnl = self.cash + self.inventory * self.price # mark-to-market PnL

        self.bid_price_A = bid_price
        self.ask_price_A = ask_price
        self.bid_fill_A = bid_fill
        self.ask_fill_A = ask_fill
        self.inventory_A = self.inventory
        self.cash_A = self.cash
        self.pnl_A = self.pnl

        market_state = self.get_state()
        market_state.update({
            'bid_price_A': self.bid_price_A,
            'ask_price_A': self.ask_price_A,
            'bid_fill_A': self.bid_fill_A,
            'ask_fill_A': self.ask_fill_A,
            'inventory_A': self.inventory_A,
            'cash_A': self.cash_A,
            'pnl_A': self.pnl_A,

            'lam': self.arrival_model.lam,
            'volatility': self.midprice_model.volatility
        })

        return market_state, total_arrivals
