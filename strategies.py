#!/usr/bin/env python

from math import log

class LinearUtilityTerminalPenaltyStrategy:
    def __init__(self, gamma=1, eta=0.01):
        self.gamma = gamma      # risk factor
        self.eta = eta  # inventory penalty intensity

    def compute(self, price, inventory):
        # clearer variable names
        reservation_price = price - 2.0 * inventory * self.eta
        spread = 2.0 / self.gamma + self.eta
        half_spread = spread / 2.0
        bid_offset = reservation_price + half_spread - price
        ask_offset = price - (reservation_price - half_spread)

        return bid_offset, ask_offset

    def get_quote(self, price, inventory, time=None):
        bid_offset, ask_offset = self.compute(price, inventory)
        bid_price = price - bid_offset
        ask_price = price + ask_offset
        return {
            'bid_price': bid_price,
            'ask_price': ask_price
        }

    def reset(self):
        # stateless strategy; nothing to reset
        pass
