import numpy as np
from market_environment import MarketEnvironment
from strategies import LinearUtilityTerminalPenaltyStrategy


class AdversaryMarketEnvironment(MarketEnvironment):
    def __init__(self,
                 strategy_params,
                 seed=None,
                 max_fill_per_direction=3,
                 dt=0.005,
                 control_lam=True,
                 control_volatility=True,
                 lam_bounds = (300, 500),
                 volatility_bounds=(0.2, 2.0),
                 inventory_bounds = (-500, 500)):
        super().__init__(seed=seed,
                         max_fill_per_direction=max_fill_per_direction,
                         dt=dt)

        self.control_lam = control_lam
        self.control_volatility = control_volatility
        self.lam_bounds = lam_bounds
        self.volatility_bounds = volatility_bounds
        self.inventory_bounds = inventory_bounds

        self.inventory = 0.0
        self.cash = 0.0
        self.pnl = 0.0
        self.strategy = LinearUtilityTerminalPenaltyStrategy(**strategy_params)


    def reset(self):
        super().reset()
        self.inventory = 0.0
        self.cash = 0.0
        self.pnl = 0.0
        self.strategy.reset()
        return self.get_state()
    
    def apply_adversary_action(self, action):
        if self.control_lam and 'lam' in action:
            self.arrival_model.lam = np.clip(action['lam'], *self.lam_bounds)

        if self.control_volatility and 'volatility' in action:
            self.midprice_model.volatility = np.clip(action['volatility'], *self.volatility_bounds)

    def step_with_adversary(self, adversary_action):
        self.apply_adversary_action(adversary_action)

        quote = self.strategy.get_quote(
            price=self.price,
            inventory=self.inventory,
            time=self.time
        )

        agent_quotes = {
            'Adversary': {'bid_price': quote['bid_price'], 'ask_price': quote['ask_price']}
        }


        max_bid_fill = max(0, self.inventory_bounds[1] - self.inventory)
        max_ask_fill = max(0, self.inventory - self.inventory_bounds[0])

        self.max_fill_override = {
            'Adversary': {
                'max_bid_fill': max_bid_fill,
                'max_ask_fill': max_ask_fill
            }
        }

        market_state, total_arrivals, agent_fills = super().step(agent_quotes)

        fills = agent_fills.get('Adversary', {})
        bid_fill = min(fills.get('bid_fills', 0), max_bid_fill)
        ask_fill = min(fills.get('ask_fills', 0), max_ask_fill)
        bid_exec_price = fills.get('bid_price', quote['bid_price'])
        ask_exec_price = fills.get('ask_price', quote['ask_price'])

        self.inventory += bid_fill - ask_fill
        self.cash += ask_fill * ask_exec_price - bid_fill * bid_exec_price
        self.pnl = self.cash + self.inventory * self.price  

        self.bid_price_Adversary = quote['bid_price']
        self.ask_price_Adversary = quote['ask_price']
        self.bid_fill_Adversary = bid_fill
        self.ask_fill_Adversary = ask_fill
        self.inventory_Adversary = self.inventory
        self.cash_Adversary = self.cash
        self.pnl_Adversary = self.pnl

        market_state = self.get_state()
        market_state.update({
            'lam': self.arrival_model.lam,
            'volatility': self.midprice_model.volatility, 

            'bid_price_Adversary': self.bid_price_Adversary,
            'ask_price_Adversary': self.ask_price_Adversary,
            'bid_fill_Adversary': self.bid_fill_Adversary,
            'ask_fill_Adversary': self.ask_fill_Adversary,
            'inventory_Adversary': self.inventory_Adversary,
            'cash_Adversary': self.cash_Adversary,
            'pnl_Adversary': self.pnl_Adversary
        })

        return market_state, total_arrivals
