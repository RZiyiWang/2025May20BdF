import numpy as np

class ExponentialFillModel:
    def __init__(self, fill_exponent=1.5, threshold=0.01):
        self.fill_exponent = fill_exponent
        self.max_depth = -np.log(threshold) / fill_exponent 
        # At fill_exponent=1.5 and threshold=0.01 (1% match probability), 
        # max_depth ≈ 3.07; fill probability should drop to zero at this price difference.


    def compute(self, price, agent_quotes):
        probs = {}
        for agent, quotes in agent_quotes.items():
            bid_offset = price - quotes['bid_price']  
            ask_offset = quotes['ask_price'] - price

           # Fill probability is 0 if deviation is negative or exceeds max depth.
            if bid_offset < 0 or bid_offset > self.max_depth:
                bid_prob = 0.0
            else:
                bid_prob = np.exp(-self.fill_exponent * bid_offset)

            if ask_offset < 0 or ask_offset > self.max_depth:
                ask_prob = 0.0
            else:
                ask_prob = np.exp(-self.fill_exponent * ask_offset)

            probs[agent] = {'bid_prob': bid_prob, 'ask_prob': ask_prob}
        return probs
