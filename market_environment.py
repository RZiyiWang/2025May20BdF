from midprice_model import BrownianMotionWithDrift
from arrival_model import PoissonArrivalModel
from fill_probability_model import ExponentialFillModel

import numpy as np

class MarketEnvironment:
    def __init__(self, seed=None, max_fill_per_direction=None, dt=0.005):
        self.price = 100.0
        self.time = 0.0          
        self.dt = dt             
        self.max_fill = max_fill_per_direction
        self.rng = np.random.default_rng(seed)

        self.midprice_model = BrownianMotionWithDrift(dt=dt, seed=seed)
        self.arrival_model = PoissonArrivalModel(dt=dt, seed=seed)
        self.fill_probability_model = ExponentialFillModel()

    def reset(self):
        self.price = 100.0
        self.time = 0.0
        return self.get_state()

    def step(self, agent_quotes):
        price_change = self.midprice_model.sample_increment()
        self.price += price_change
        self.time += self.dt   

        total_arrivals = self.arrival_model.next_arrivals()
        buy_orders = total_arrivals // 2
        sell_orders = total_arrivals - buy_orders

        fill_probs = self.fill_probability_model.compute(self.price, agent_quotes)
        agent_fills = self.allocate_fills(buy_orders, sell_orders, fill_probs, agent_quotes)

        return self.get_state(), total_arrivals, agent_fills

    def allocate_fills(self, buy_orders, sell_orders, fill_probs, agent_quotes):
        agent_fills = {}

        ask_probs = {a: p['ask_prob'] for a, p in fill_probs.items()}
        total_ask_prob = sum(ask_probs.values())
        if total_ask_prob > 0 and buy_orders > 0:
            norm_probs = np.array([ask_probs[a] / total_ask_prob for a in ask_probs])
            allocs = np.random.multinomial(buy_orders, norm_probs)
            for i, agent in enumerate(ask_probs):
                ask_fill = allocs[i]
                if self.max_fill is not None:
                    ask_fill = min(ask_fill, self.max_fill)
                agent_fills.setdefault(agent, {})['ask_fills'] = ask_fill
                agent_fills[agent]['ask_price'] = agent_quotes[agent]['ask_price']

        bid_probs = {a: p['bid_prob'] for a, p in fill_probs.items()}
        total_bid_prob = sum(bid_probs.values())
        if total_bid_prob > 0 and sell_orders > 0:
            norm_probs = np.array([bid_probs[a] / total_bid_prob for a in bid_probs])
            allocs = np.random.multinomial(sell_orders, norm_probs)
            for i, agent in enumerate(bid_probs):
                bid_fill = allocs[i]
                if self.max_fill is not None:
                    bid_fill = min(bid_fill, self.max_fill)
                agent_fills.setdefault(agent, {})['bid_fills'] = bid_fill
                agent_fills[agent]['bid_price'] = agent_quotes[agent]['bid_price']

        return agent_fills

    def get_state(self):
        return {
            'price': self.price,
            'time': self.time,
            'lam': self.arrival_model.lam,
            'volatility': self.midprice_model.volatility,

            'bid_price_Adversary': getattr(self, 'bid_price_Adversary', 0.0),
            'ask_price_Adversary': getattr(self, 'ask_price_Adversary', 0.0),
            'bid_fill_Adversary': getattr(self, 'bid_fill_Adversary', 0.0),
            'ask_fill_Adversary': getattr(self, 'ask_fill_Adversary', 0.0),
            'inventory_Adversary': getattr(self, 'inventory_Adversary', 0.0),
            'cash_Adversary': getattr(self, 'cash_Adversary', 0.0),
            'pnl_Adversary': getattr(self, 'pnl_Adversary', 0.0),

            'bid_price_A': getattr(self, 'bid_price_A', 0.0),
            'ask_price_A': getattr(self, 'ask_price_A', 0.0),
            'bid_fill_A': getattr(self, 'bid_fill_A', 0.0),
            'ask_fill_A': getattr(self, 'ask_fill_A', 0.0),
            'inventory_A': getattr(self, 'inventory_A', 0.0),
            'cash_A': getattr(self, 'cash_A', 0.0),
            'pnl_A': getattr(self, 'pnl_A', 0.0),

            'bid_price_B1': getattr(self, 'bid_price_B1', 0.0),
            'ask_price_B1': getattr(self, 'ask_price_B1', 0.0),
            'bid_fill_B1': getattr(self, 'bid_fill_B1', 0.0),
            'ask_fill_B1': getattr(self, 'ask_fill_B1', 0.0),
            'inventory_B1': getattr(self, 'inventory_B1', 0.0),
            'cash_B1': getattr(self, 'cash_B1', 0.0),
            'pnl_B1': getattr(self, 'pnl_B1', 0.0),

            'bid_price_B2': getattr(self, 'bid_price_B2', 0.0),
            'ask_price_B2': getattr(self, 'ask_price_B2', 0.0),
            'bid_fill_B2': getattr(self, 'bid_fill_B2', 0.0),
            'ask_fill_B2': getattr(self, 'ask_fill_B2', 0.0),
            'inventory_B2': getattr(self, 'inventory_B2', 0.0),
            'cash_B2': getattr(self, 'cash_B2', 0.0),
            'pnl_B2': getattr(self, 'pnl_B2', 0.0)

        }





