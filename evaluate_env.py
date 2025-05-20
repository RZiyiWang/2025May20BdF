import numpy as np
from stable_baselines3 import PPO
from market_environment import MarketEnvironment
import torch
from reward_utils import MMA_reward,MMB1_reward,MMB2_reward

#A + adv/fix adv
class MMA_eval_Environment(MarketEnvironment):

    def __init__(self,
                 adversary_model=None,
                 agent_A=None,
                 seed=42,
                 max_fill_per_direction=3,
                 dt=0.005,
                 lam_bounds=(300, 500),
                 volatility_bounds=(0.2, 2.0),
                 inventory_bounds=(-500, 500),
                 zeta=0.0,
                 eta=0.01):
        super().__init__(seed=seed, max_fill_per_direction=max_fill_per_direction, dt=dt)

        self.lam_bounds = lam_bounds
        self.volatility_bounds = volatility_bounds
        self.inventory_bounds = inventory_bounds
        self.zeta = zeta
        self.eta = eta

        self.fixed_lam = 400.0
        self.fixed_volatility = 1.1

        self.inventory_A = 0.0
        self.cash_A = 0.0
        self.pnl_A = 0.0
        self.prev_value = 0.0
        self.reward_list = []

        self.pnl_A_list = []
        self.spread_list = []
        self.price_list = []
        self.inventory_list = []
        self.aggressiveness_list = []
        self.fill_list = []
        self.arrival_list = []
        self.zero_fill_steps = 0

        if adversary_model is not None:
            self.adversary_model = PPO.load(adversary_model)
            print("Adversary model loaded. Using strategic market parameters.")
        else:
            self.adversary_model = None
            self.arrival_model.lam = self.fixed_lam
            self.midprice_model.volatility = self.fixed_volatility

        if agent_A:
            self.agent_A = PPO.load(agent_A)
            print("Pre-trained Market-Making Agent loaded successfully. Starting evaluation...")
        else:
            self.agent_A = None
            print("No Market-Making Agent found. Please verify the input path.")

    def reset(self):
        super().reset()
        self.inventory_A = 0.0
        self.cash_A = 0.0
        self.pnl_A = 0.0
        self.prev_value = 0.0
        self.reward_list.clear()

        self.pnl_A_list.clear()
        self.spread_list.clear()
        self.price_list.clear()
        self.inventory_list.clear()
        self.aggressiveness_list.clear()
        self.fill_list.clear()
        self.arrival_list.clear()
        self.zero_fill_steps = 0

        return self.get_state()

    def step_with_agent(self, final=False):
        if self.adversary_model:
            Adversary_obs = np.array([self.price, self.time], dtype=np.float32).reshape(1, -1)
            with torch.no_grad():
                adv_action = self.adversary_model.predict(Adversary_obs, deterministic=False)[0]
            adv_action = np.array(adv_action).flatten()
            lam = self.lam_bounds[0] + (adv_action[0] + 1.0) * 0.5 * (self.lam_bounds[1] - self.lam_bounds[0])
            volatility = self.volatility_bounds[0] + (adv_action[1] + 1.0) * 0.5 * (self.volatility_bounds[1] - self.volatility_bounds[0])

            self.arrival_model.lam = np.clip(lam, *self.lam_bounds)
            self.midprice_model.volatility = np.clip(volatility, *self.volatility_bounds)

        A_obs = np.array([self.price, self.time, self.inventory_A, self.cash_A], dtype=np.float32).reshape(1, -1)
        with torch.no_grad():
            action_A = self.agent_A.predict(A_obs, deterministic=False)[0]
        action_A = np.array(action_A).flatten()
        action_A = 2.5 * (action_A + 1.0)

        A_bid_offset = float(action_A[0])
        A_ask_offset = float(action_A[1])
        A_bid_price = self.price - A_bid_offset
        A_ask_price = self.price + A_ask_offset

        agent_quotes = {'MM_A': {'bid_price': A_bid_price, 'ask_price': A_ask_price}}

        A_max_bid_fill = max(0, self.inventory_bounds[1] - self.inventory_A)
        A_max_ask_fill = max(0, self.inventory_A - self.inventory_bounds[0])
        self.max_fill_override = {
            'MM_A': {
                'max_bid_fill': A_max_bid_fill,
                'max_ask_fill': A_max_ask_fill
            }
        }

        market_state, total_arrivals, agent_fills = super().step(agent_quotes)

        A_fill = agent_fills.get('MM_A', {})
        A_bid_fill = min(A_fill.get('bid_fills', 0), A_max_bid_fill)
        A_ask_fill = min(A_fill.get('ask_fills', 0), A_max_ask_fill)
        A_bid_exec_price = A_fill.get('bid_price', A_bid_price)
        A_ask_exec_price = A_fill.get('ask_price', A_ask_price)

        self.inventory_A += A_bid_fill - A_ask_fill
        self.cash_A += A_ask_fill * A_ask_exec_price - A_bid_fill * A_bid_exec_price
        self.pnl_A = self.cash_A + self.inventory_A * self.price
        self.pnl_A_list.append(self.pnl_A)

        current_value = self.pnl_A
        pnl_diff = current_value - self.prev_value
        self.prev_value = current_value
        reward = MMA_reward(pnl_diff, inventory=self.inventory_A, final=final, max_inventory=self.inventory_bounds[1], zeta=self.zeta, eta=self.eta)
        self.reward_list.append(reward)

        self.bid_price_A = A_bid_price
        self.ask_price_A = A_ask_price
        self.bid_fill_A = A_bid_fill
        self.ask_fill_A = A_ask_fill

        self.spread_list.append(self.ask_price_A - self.bid_price_A)
        self.price_list.append(self.price)
        self.inventory_list.append(self.inventory_A)
        self.aggressiveness_list.append((A_bid_offset + A_ask_offset) / 2.0)
        self.fill_list.append(A_bid_fill + A_ask_fill)
        self.arrival_list.append(total_arrivals)
        if A_bid_fill + A_ask_fill == 0:
            self.zero_fill_steps += 1

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

        return market_state, total_arrivals, reward

    def evaluate_metrics(self):
        metrics = {
            "PnL_mean": np.mean(self.pnl_A_list),
            "PnL_std": np.std(self.pnl_A_list),
            "Sharpe_ratio": np.mean(self.pnl_A_list) / (np.std(self.pnl_A_list) + 1e-6),
            "Inventory_volatility": np.std(self.inventory_list),
            "Quote_aggressiveness": np.mean(self.aggressiveness_list),
            "Market_share": np.sum(self.fill_list) / (np.sum(self.arrival_list) + 1e-6),
            "Avg_spread": np.mean(self.spread_list),
            "Price_volatility": np.std(self.price_list),
            "Zero_fill_steps": self.zero_fill_steps,
            "Mean_reward": np.mean(self.reward_list)
        }
        return metrics

#B1 + adv/fix adv
class MMB1_eval_Environment(MarketEnvironment):

    def __init__(self,
                 adversary_model=None,
                 agent_B1=None,
                 seed=42,
                 max_fill_per_direction=3,
                 dt=0.005,
                 lam_bounds=(300, 500),
                 volatility_bounds=(0.2, 2.0),
                 inventory_bounds=(-500, 500),
                 zeta=0.0,
                 eta=0.01):
        super().__init__(seed=seed, max_fill_per_direction=max_fill_per_direction, dt=dt)

        self.lam_bounds = lam_bounds
        self.volatility_bounds = volatility_bounds
        self.inventory_bounds = inventory_bounds
        self.zeta = zeta
        self.eta = eta

        self.fixed_lam = 400.0
        self.fixed_volatility = 1.1

        self.inventory_B1 = 0.0
        self.cash_B1 = 0.0
        self.pnl_B1 = 0.0
        self.prev_value = 0.0
        self.reward_list = []

        self.pnl_B1_list = []
        self.spread_list = []
        self.price_list = []
        self.inventory_list = []
        self.aggressiveness_list = []
        self.fill_list = []
        self.arrival_list = []
        self.zero_fill_steps = 0

        if adversary_model is not None:
            self.adversary_model = PPO.load(adversary_model)
            print("Adversary model loaded. Using strategic market parameters.")
        else:
            self.adversary_model = None
            self.arrival_model.lam = self.fixed_lam
            self.midprice_model.volatility = self.fixed_volatility

        if agent_B1:
            self.agent_B1 = PPO.load(agent_B1)
            print("Pre-trained Market-Making Agent loaded successfully. Starting evaluation...")
        else:
            self.agent_B1 = None
            print("No Market-Making Agent found. Please verify the input path.")

    def reset(self):
        super().reset()
        self.inventory_B1 = 0.0
        self.cash_B1 = 0.0
        self.pnl_B1 = 0.0
        self.prev_value = 0.0
        self.reward_list.clear()

        self.pnl_B1_list.clear()
        self.spread_list.clear()
        self.price_list.clear()
        self.inventory_list.clear()
        self.aggressiveness_list.clear()
        self.fill_list.clear()
        self.arrival_list.clear()
        self.zero_fill_steps = 0

        return self.get_state()

    def step_with_agent(self, final=False):
        if self.adversary_model:
            Adversary_obs = np.array([self.price, self.time], dtype=np.float32).reshape(1, -1)
            with torch.no_grad():
                adv_action = self.adversary_model.predict(Adversary_obs, deterministic=False)[0]
            adv_action = np.array(adv_action).flatten()
            lam = self.lam_bounds[0] + (adv_action[0] + 1.0) * 0.5 * (self.lam_bounds[1] - self.lam_bounds[0])
            volatility = self.volatility_bounds[0] + (adv_action[1] + 1.0) * 0.5 * (self.volatility_bounds[1] - self.volatility_bounds[0])

            self.arrival_model.lam = np.clip(lam, *self.lam_bounds)
            self.midprice_model.volatility = np.clip(volatility, *self.volatility_bounds)

        B1_obs = np.array([self.price, self.time, self.inventory_B1, self.cash_B1], dtype=np.float32).reshape(1, -1)
        with torch.no_grad():
            action_B1 = self.agent_B1.predict(B1_obs, deterministic=False)[0]
        action_B1 = np.array(action_B1).flatten()
        action_B1 = 2.5 * (action_B1 + 1.0)

        B1_bid_offset = float(action_B1[0])
        B1_ask_offset = float(action_B1[1])
        B1_bid_price = self.price - B1_bid_offset
        B1_ask_price = self.price + B1_ask_offset

        agent_quotes = {'MM_B1': {'bid_price': B1_bid_price, 'ask_price': B1_ask_price}}

        B1_max_bid_fill = max(0, self.inventory_bounds[1] - self.inventory_B1)
        B1_max_ask_fill = max(0, self.inventory_B1 - self.inventory_bounds[0])
        self.max_fill_override = {
            'MM_B1': {
                'max_bid_fill': B1_max_bid_fill,
                'max_ask_fill': B1_max_ask_fill
            }
        }

        market_state, total_arrivals, agent_fills = super().step(agent_quotes)

        B1_fill = agent_fills.get('MM_B1', {})
        B1_bid_fill = min(B1_fill.get('bid_fills', 0), B1_max_bid_fill)
        B1_ask_fill = min(B1_fill.get('ask_fills', 0), B1_max_ask_fill)
        B1_bid_exec_price = B1_fill.get('bid_price', B1_bid_price)
        B1_ask_exec_price = B1_fill.get('ask_price', B1_ask_price)

        self.inventory_B1 += B1_bid_fill - B1_ask_fill
        self.cash_B1 += B1_ask_fill * B1_ask_exec_price - B1_bid_fill * B1_bid_exec_price
        self.pnl_B1 = self.cash_B1 + self.inventory_B1 * self.price
        self.pnl_B1_list.append(self.pnl_B1)

        current_value = self.pnl_B1
        pnl_diff = current_value - self.prev_value
        self.prev_value = current_value
        reward = MMB1_reward(pnl_diff, inventory=self.inventory_B1, final=final, max_inventory=self.inventory_bounds[1], zeta=self.zeta, eta=self.eta)
        self.reward_list.append(reward)

        self.bid_price_B1 = B1_bid_price
        self.ask_price_B1 = B1_ask_price
        self.bid_fill_B1 = B1_bid_fill
        self.ask_fill_B1 = B1_ask_fill

        self.spread_list.append(self.ask_price_B1 - self.bid_price_B1)
        self.price_list.append(self.price)
        self.inventory_list.append(self.inventory_B1)
        self.aggressiveness_list.append((B1_bid_offset + B1_ask_offset) / 2.0)
        self.fill_list.append(B1_bid_fill + B1_ask_fill)
        self.arrival_list.append(total_arrivals)
        if B1_bid_fill + B1_ask_fill == 0:
            self.zero_fill_steps += 1

        market_state = self.get_state()
        market_state.update({
            'bid_price_B1': self.bid_price_B1,
            'ask_price_B1': self.ask_price_B1,
            'bid_fill_B1': self.bid_fill_B1,
            'ask_fill_B1': self.ask_fill_B1,
            'inventory_B1': self.inventory_B1,
            'cash_B1': self.cash_B1,
            'pnl_B1': self.pnl_B1,
            'lam': self.arrival_model.lam,
            'volatility': self.midprice_model.volatility
        })

        return market_state, total_arrivals, reward

    def evaluate_metrics(self):
        metrics = {
            "PnL_mean": np.mean(self.pnl_B1_list),
            "PnL_std": np.std(self.pnl_B1_list),
            "Sharpe_ratio": np.mean(self.pnl_B1_list) / (np.std(self.pnl_B1_list) + 1e-6),
            "Inventory_volatility": np.std(self.inventory_list),
            "Quote_aggressiveness": np.mean(self.aggressiveness_list),
            "Market_share": np.sum(self.fill_list) / (np.sum(self.arrival_list) + 1e-6),
            "Avg_spread": np.mean(self.spread_list),
            "Price_volatility": np.std(self.price_list),
            "Zero_fill_steps": self.zero_fill_steps,
            "Mean_reward": np.mean(self.reward_list)
        }
        return metrics

#A + B1 + fix adv
class MultiAgent_eval_Environment_A_B1(MarketEnvironment):

    def __init__(self,
                 adversary_model=None,
                 agent_A=None,
                 agent_B1=None,
                 herding_threshold=0.01,
                 seed=42,
                 max_fill_per_direction=3,
                 dt=0.005,
                 lam_bounds=(300, 500),
                 volatility_bounds=(0.2, 2.0),
                 inventory_bounds=(-500, 500),
                 zeta=0.0,
                 eta=0.01):

        super().__init__(seed=seed, max_fill_per_direction=max_fill_per_direction, dt=dt)

        self.lam_bounds = lam_bounds
        self.volatility_bounds = volatility_bounds
        self.inventory_bounds = inventory_bounds
        self.zeta = zeta
        self.eta = eta

        self.fixed_lam = 400.0
        self.fixed_volatility = 1.1

        self.inventory_A = 0.0
        self.cash_A = 0.0
        self.pnl_A = 0.0
        self.prev_value_A = 0.0
        self.reward_list_A = []

        self.inventory_B1 = 0.0
        self.cash_B1 = 0.0
        self.pnl_B1 = 0.0
        self.prev_value_B1 = 0.0
        self.reward_list_B1 = []

        # === Metric buffers ===
        self.pnl_A_list = []
        self.pnl_B1_list = []
        self.spread_list = []
        self.price_list = []

        self.inventory_A_list = []
        self.inventory_B1_list = []

        self.aggressiveness_A = []
        self.aggressiveness_B1 = []

        self.fill_A_list = []
        self.fill_B1_list = []

        self.arrival_list = []
        self.zero_fill_steps = 0

        self.drawdown_overlap = []
        self.herding_threshold = herding_threshold
        self.quote_similarity = []
        self.quote_distance_bid = []
        self.quote_distance_ask = []
        self.same_side_quote = []
        self.fill_overlap = []

        if adversary_model:
            self.adversary_model = PPO.load(adversary_model)
            print("Adversary model loaded. Using strategic market parameters.")
        else:
            self.adversary_model = None
            self.arrival_model.lam = self.fixed_lam
            self.midprice_model.volatility = self.fixed_volatility
            print(f"No adversary model loaded. Using fixed λ={self.fixed_lam}, volatility={self.fixed_volatility}")

        self.agent_A = PPO.load(agent_A) if agent_A else None
        self.agent_B1 = PPO.load(agent_B1) if agent_B1 else None

    def reset(self):
        super().reset()
        # === Agents ===
        self.inventory_A = 0.0
        self.cash_A = 0.0
        self.pnl_A = 0.0
        self.prev_value_A = 0.0
        self.reward_list_A.clear()

        self.inventory_B1 = 0.0
        self.cash_B1 = 0.0
        self.pnl_B1 = 0.0
        self.prev_value_B1 = 0.0
        self.reward_list_B1.clear()

        # === Metrics ===
        self.pnl_A_list.clear()
        self.pnl_B1_list.clear()
        self.spread_list.clear()
        self.price_list.clear()
        self.inventory_A_list.clear()
        self.inventory_B1_list.clear()
        self.aggressiveness_A.clear()
        self.aggressiveness_B1.clear()
        self.fill_A_list.clear()
        self.fill_B1_list.clear()
        self.arrival_list.clear()
        self.zero_fill_steps = 0

        self.drawdown_overlap.clear()
        self.quote_similarity.clear()
        self.quote_distance_bid.clear()
        self.quote_distance_ask.clear()
        self.same_side_quote.clear()
        self.fill_overlap.clear()

        return self.get_state()

    def step_with_agent(self, final=False):
        if self.adversary_model:
            obs = np.array([self.price, self.time], dtype=np.float32).reshape(1, -1)
            adv_action = self.adversary_model.predict(obs, deterministic=False)[0]
            adv_action = np.array(adv_action).flatten()
            lam = self.lam_bounds[0] + (adv_action[0] + 1.0) * 0.5 * (self.lam_bounds[1] - self.lam_bounds[0])
            volatility = self.volatility_bounds[0] + (adv_action[1] + 1.0) * 0.5 * (self.volatility_bounds[1] - self.volatility_bounds[0])
            self.arrival_model.lam = np.clip(lam, *self.lam_bounds)
            self.midprice_model.volatility = np.clip(volatility, *self.volatility_bounds)

        A_obs = np.array([self.price, self.time, self.inventory_A, self.cash_A], dtype=np.float32).reshape(1, -1)
        action_A = self.agent_A.predict(A_obs, deterministic=False)[0]
        action_A = 2.5 * (np.array(action_A).flatten() + 1.0)

        B1_obs = np.array([self.price, self.time, self.inventory_B1, self.cash_B1], dtype=np.float32).reshape(1, -1)
        action_B1 = self.agent_B1.predict(B1_obs, deterministic=False)[0]
        action_B1 = 2.5 * (np.array(action_B1).flatten() + 1.0)

        A_bid_price = self.price - action_A[0]
        A_ask_price = self.price + action_A[1]
        B1_bid_price = self.price - action_B1[0]
        B1_ask_price = self.price + action_B1[1]

        agent_quotes = {
            'MM_A': {'bid_price': A_bid_price, 'ask_price': A_ask_price},
            'MM_B1': {'bid_price': B1_bid_price, 'ask_price': B1_ask_price}
        }

        self.max_fill_override = {
            'MM_A': {
                'max_bid_fill': max(0, self.inventory_bounds[1] - self.inventory_A),
                'max_ask_fill': max(0, self.inventory_A - self.inventory_bounds[0])
            },
            'MM_B1': {
                'max_bid_fill': max(0, self.inventory_bounds[1] - self.inventory_B1),
                'max_ask_fill': max(0, self.inventory_B1 - self.inventory_bounds[0])
            }
        }

        market_state, total_arrivals, agent_fills = super().step(agent_quotes)

        def update_agent(agent, fill, bid_offset, ask_offset):
            bid_fill = fill.get('bid_fills', 0)
            ask_fill = fill.get('ask_fills', 0)
            bid_price = fill.get('bid_price', self.price - bid_offset)
            ask_price = fill.get('ask_price', self.price + ask_offset)
            return bid_fill, ask_fill, bid_price, ask_price

        A_bid_fill, A_ask_fill, A_exec_bid, A_exec_ask = update_agent('A', agent_fills.get('MM_A', {}), action_A[0], action_A[1])
        B1_bid_fill, B1_ask_fill, B1_exec_bid, B1_exec_ask = update_agent('B1', agent_fills.get('MM_B1', {}), action_B1[0], action_B1[1])

        self.inventory_A += A_bid_fill - A_ask_fill
        self.cash_A += A_ask_fill * A_exec_ask - A_bid_fill * A_exec_bid
        self.pnl_A = self.cash_A + self.inventory_A * self.price
        self.pnl_A_list.append(self.pnl_A)

        self.inventory_B1 += B1_bid_fill - B1_ask_fill
        self.cash_B1 += B1_ask_fill * B1_exec_ask - B1_bid_fill * B1_exec_bid
        self.pnl_B1 = self.cash_B1 + self.inventory_B1 * self.price
        self.pnl_B1_list.append(self.pnl_B1)

        pnl_diff_A = self.pnl_A - self.prev_value_A
        self.prev_value_A = self.pnl_A
        self.reward_list_A.append(MMA_reward(pnl_diff_A, self.inventory_A, final, self.inventory_bounds[1], self.zeta, self.eta))

        pnl_diff_B1 = self.pnl_B1 - self.prev_value_B1
        self.prev_value_B1 = self.pnl_B1
        self.reward_list_B1.append(MMB1_reward(pnl_diff_B1, self.inventory_B1, final, self.inventory_bounds[1], self.zeta, self.eta))

        self.spread_list.append((A_ask_price - A_bid_price + B1_ask_price - B1_bid_price) / 2)
        self.price_list.append(self.price)
        self.inventory_A_list.append(self.inventory_A)
        self.inventory_B1_list.append(self.inventory_B1)
        self.aggressiveness_A.append((action_A[0] + action_A[1]) / 2)
        self.aggressiveness_B1.append((action_B1[0] + action_B1[1]) / 2)
        self.fill_A_list.append(A_bid_fill + A_ask_fill)
        self.fill_B1_list.append(B1_bid_fill + B1_ask_fill)
        self.arrival_list.append(total_arrivals)
        if (A_bid_fill + A_ask_fill + B1_bid_fill + B1_ask_fill) == 0:
            self.zero_fill_steps += 1

        self.drawdown_overlap.append(int(pnl_diff_A < 0 and pnl_diff_B1 < 0))
        self.quote_similarity.append(int(abs(A_bid_price - B1_bid_price) < self.herding_threshold and abs(A_ask_price - B1_ask_price) < self.herding_threshold))
        self.quote_distance_bid.append(abs(A_bid_price - B1_bid_price))
        self.quote_distance_ask.append(abs(A_ask_price - B1_ask_price))
        self.same_side_quote.append(int((A_bid_price > self.price and B1_bid_price > self.price) or (A_ask_price < self.price and B1_ask_price < self.price)))
        self.fill_overlap.append(int((A_bid_fill > 0 and B1_bid_fill > 0) or (A_ask_fill > 0 and B1_ask_fill > 0)))

        return self.get_state(), total_arrivals

    def evaluate_metrics(self):
        total_arrivals = np.sum(self.arrival_list) + 1e-6
        return {
            # Agent-level
            "PnL_mean_A": np.mean(self.pnl_A_list),
            "PnL_std_A": np.std(self.pnl_A_list),
            "Sharpe_ratio_A": np.mean(self.pnl_A_list) / (np.std(self.pnl_A_list) + 1e-6),
            "Inventory_volatility_A": np.std(self.inventory_A_list),
            "Quote_aggressiveness_A": np.mean(self.aggressiveness_A),
            "Market_share_A": np.sum(self.fill_A_list) / total_arrivals,
            "Mean_reward_A": np.mean(self.reward_list_A),

            "PnL_mean_B1": np.mean(self.pnl_B1_list),
            "PnL_std_B1": np.std(self.pnl_B1_list),
            "Sharpe_ratio_B1": np.mean(self.pnl_B1_list) / (np.std(self.pnl_B1_list) + 1e-6),
            "Inventory_volatility_B1": np.std(self.inventory_B1_list),
            "Quote_aggressiveness_B1": np.mean(self.aggressiveness_B1),
            "Market_share_B1": np.sum(self.fill_B1_list) / total_arrivals,
            "Mean_reward_B1": np.mean(self.reward_list_B1),

            # Market/system-level
            "Avg_spread": np.mean(self.spread_list),
            "Price_volatility": np.std(self.price_list),
            "Fill_ratio": (np.sum(self.fill_A_list) + np.sum(self.fill_B1_list)) / total_arrivals,
            "Zero_fill_steps": self.zero_fill_steps,

            # Interaction/systemic
            "Joint_drawdown_ratio": np.mean(self.drawdown_overlap),
            "Herding_ratio": np.mean(self.quote_similarity),
            "Inventory_divergence": np.std(np.array(self.inventory_A_list) - np.array(self.inventory_B1_list)),
            "Quote_distance_bid": np.mean(self.quote_distance_bid),
            "Quote_distance_ask": np.mean(self.quote_distance_ask),
            "Same_side_quote_ratio": np.mean(self.same_side_quote),
            "Fill_overlap_ratio": np.mean(self.fill_overlap)
        }

#A + B2+fix adv
class MultiAgent_eval_Environment_A_B2(MarketEnvironment):

    def __init__(self,
                 adversary_model=None,
                 agent_A=None,
                 agent_B2=None,
                 herding_threshold=0.01,
                 seed=42,
                 max_fill_per_direction=3,
                 dt=0.005,
                 lam_bounds=(300, 500),
                 volatility_bounds=(0.2, 2.0),
                 inventory_bounds=(-500, 500),
                 zeta=0.0,
                 eta=0.01):

        super().__init__(seed=seed, max_fill_per_direction=max_fill_per_direction, dt=dt)

        self.lam_bounds = lam_bounds
        self.volatility_bounds = volatility_bounds
        self.inventory_bounds = inventory_bounds
        self.zeta = zeta
        self.eta = eta

        self.fixed_lam = 400.0
        self.fixed_volatility = 1.1

        self.inventory_A = 0.0
        self.cash_A = 0.0
        self.pnl_A = 0.0
        self.prev_value_A = 0.0
        self.reward_list_A = []

        self.inventory_B2 = 0.0
        self.cash_B2 = 0.0
        self.pnl_B2 = 0.0
        self.prev_value_B2 = 0.0
        self.reward_list_B2 = []

        # === Metric buffers ===
        self.pnl_A_list = []
        self.pnl_B2_list = []
        self.spread_list = []
        self.price_list = []

        self.inventory_A_list = []
        self.inventory_B2_list = []

        self.aggressiveness_A = []
        self.aggressiveness_B2 = []

        self.fill_A_list = []
        self.fill_B2_list = []

        self.arrival_list = []
        self.zero_fill_steps = 0

        self.drawdown_overlap = []
        self.herding_threshold = herding_threshold
        self.quote_similarity = []
        self.quote_distance_bid = []
        self.quote_distance_ask = []
        self.same_side_quote = []
        self.fill_overlap = []

        if adversary_model:
            self.adversary_model = PPO.load(adversary_model)
            print("Adversary model loaded. Using strategic market parameters.")
        else:
            self.adversary_model = None
            self.arrival_model.lam = self.fixed_lam
            self.midprice_model.volatility = self.fixed_volatility
            print(f"No adversary model loaded. Using fixed λ={self.fixed_lam}, volatility={self.fixed_volatility}")

        self.agent_A = PPO.load(agent_A) if agent_A else None
        self.agent_B2 = PPO.load(agent_B2) if agent_B2 else None

    def reset(self):
        super().reset()
        # === Agents ===
        self.inventory_A = 0.0
        self.cash_A = 0.0
        self.pnl_A = 0.0
        self.prev_value_A = 0.0
        self.reward_list_A.clear()

        self.inventory_B2 = 0.0
        self.cash_B2 = 0.0
        self.pnl_B2 = 0.0
        self.prev_value_B2 = 0.0
        self.reward_list_B2.clear()

        # === Metrics ===
        self.pnl_A_list.clear()
        self.pnl_B2_list.clear()
        self.spread_list.clear()
        self.price_list.clear()
        self.inventory_A_list.clear()
        self.inventory_B2_list.clear()
        self.aggressiveness_A.clear()
        self.aggressiveness_B2.clear()
        self.fill_A_list.clear()
        self.fill_B2_list.clear()
        self.arrival_list.clear()
        self.zero_fill_steps = 0

        self.drawdown_overlap.clear()
        self.quote_similarity.clear()
        self.quote_distance_bid.clear()
        self.quote_distance_ask.clear()
        self.same_side_quote.clear()
        self.fill_overlap.clear()

        return self.get_state()

    def step_with_agent(self, final=False):
        if self.adversary_model:
            obs = np.array([self.price, self.time], dtype=np.float32).reshape(1, -1)
            adv_action = self.adversary_model.predict(obs, deterministic=False)[0]
            adv_action = np.array(adv_action).flatten()
            lam = self.lam_bounds[0] + (adv_action[0] + 1.0) * 0.5 * (self.lam_bounds[1] - self.lam_bounds[0])
            volatility = self.volatility_bounds[0] + (adv_action[1] + 1.0) * 0.5 * (self.volatility_bounds[1] - self.volatility_bounds[0])
            self.arrival_model.lam = np.clip(lam, *self.lam_bounds)
            self.midprice_model.volatility = np.clip(volatility, *self.volatility_bounds)

        A_obs = np.array([self.price, self.time, self.inventory_A, self.cash_A], dtype=np.float32).reshape(1, -1)
        action_A = self.agent_A.predict(A_obs, deterministic=False)[0]
        action_A = 2.5 * (np.array(action_A).flatten() + 1.0)

        B2_obs = np.array([self.price, self.time, self.inventory_B2, self.cash_B2], dtype=np.float32).reshape(1, -1)
        action_B2 = self.agent_B2.predict(B2_obs, deterministic=False)[0]
        action_B2 = 2.5 * (np.array(action_B2).flatten() + 1.0)

        A_bid_price = self.price - action_A[0]
        A_ask_price = self.price + action_A[1]
        B2_bid_price = self.price - action_B2[0]
        B2_ask_price = self.price + action_B2[1]

        agent_quotes = {
            'MM_A': {'bid_price': A_bid_price, 'ask_price': A_ask_price},
            'MM_B2': {'bid_price': B2_bid_price, 'ask_price': B2_ask_price}
        }

        self.max_fill_override = {
            'MM_A': {
                'max_bid_fill': max(0, self.inventory_bounds[1] - self.inventory_A),
                'max_ask_fill': max(0, self.inventory_A - self.inventory_bounds[0])
            },
            'MM_B2': {
                'max_bid_fill': max(0, self.inventory_bounds[1] - self.inventory_B2),
                'max_ask_fill': max(0, self.inventory_B2 - self.inventory_bounds[0])
            }
        }

        market_state, total_arrivals, agent_fills = super().step(agent_quotes)

        def update_agent(fill, bid_offset, ask_offset):
            bid_fill = fill.get('bid_fills', 0)
            ask_fill = fill.get('ask_fills', 0)
            bid_price = fill.get('bid_price', self.price - bid_offset)
            ask_price = fill.get('ask_price', self.price + ask_offset)
            return bid_fill, ask_fill, bid_price, ask_price

        A_bid_fill, A_ask_fill, A_exec_bid, A_exec_ask = update_agent(agent_fills.get('MM_A', {}), action_A[0], action_A[1])
        B2_bid_fill, B2_ask_fill, B2_exec_bid, B2_exec_ask = update_agent(agent_fills.get('MM_B2', {}), action_B2[0], action_B2[1])

        self.inventory_A += A_bid_fill - A_ask_fill
        self.cash_A += A_ask_fill * A_exec_ask - A_bid_fill * A_exec_bid
        self.pnl_A = self.cash_A + self.inventory_A * self.price
        self.pnl_A_list.append(self.pnl_A)

        self.inventory_B2 += B2_bid_fill - B2_ask_fill
        self.cash_B2 += B2_ask_fill * B2_exec_ask - B2_bid_fill * B2_exec_bid
        self.pnl_B2 = self.cash_B2 + self.inventory_B2 * self.price
        self.pnl_B2_list.append(self.pnl_B2)

        pnl_diff_A = self.pnl_A - self.prev_value_A
        self.prev_value_A = self.pnl_A
        self.reward_list_A.append(MMA_reward(pnl_diff_A, self.inventory_A, final, self.inventory_bounds[1], self.zeta, self.eta))

        pnl_diff_B2 = self.pnl_B2 - self.prev_value_B2
        self.prev_value_B2 = self.pnl_B2
        self.reward_list_B2.append(MMB2_reward(pnl_diff_A, self.inventory_A, final, self.inventory_bounds[1], self.zeta, self.eta)) #opposite of reward A

        self.spread_list.append((A_ask_price - A_bid_price + B2_ask_price - B2_bid_price) / 2)
        self.price_list.append(self.price)
        self.inventory_A_list.append(self.inventory_A)
        self.inventory_B2_list.append(self.inventory_B2)
        self.aggressiveness_A.append((action_A[0] + action_A[1]) / 2)
        self.aggressiveness_B2.append((action_B2[0] + action_B2[1]) / 2)
        self.fill_A_list.append(A_bid_fill + A_ask_fill)
        self.fill_B2_list.append(B2_bid_fill + B2_ask_fill)
        self.arrival_list.append(total_arrivals)
        if (A_bid_fill + A_ask_fill + B2_bid_fill + B2_ask_fill) == 0:
            self.zero_fill_steps += 1

        self.drawdown_overlap.append(int(pnl_diff_A < 0 and pnl_diff_B2 < 0))
        self.quote_similarity.append(int(abs(A_bid_price - B2_bid_price) < self.herding_threshold and abs(A_ask_price - B2_ask_price) < self.herding_threshold))
        self.quote_distance_bid.append(abs(A_bid_price - B2_bid_price))
        self.quote_distance_ask.append(abs(A_ask_price - B2_ask_price))
        self.same_side_quote.append(int((A_bid_price > self.price and B2_bid_price > self.price) or (A_ask_price < self.price and B2_ask_price < self.price)))
        self.fill_overlap.append(int((A_bid_fill > 0 and B2_bid_fill > 0) or (A_ask_fill > 0 and B2_ask_fill > 0)))

        return self.get_state(), total_arrivals

    def evaluate_metrics(self):
        total_arrivals = np.sum(self.arrival_list) + 1e-6
        return {
            # Agent-level
            "PnL_mean_A": np.mean(self.pnl_A_list),
            "PnL_std_A": np.std(self.pnl_A_list),
            "Sharpe_ratio_A": np.mean(self.pnl_A_list) / (np.std(self.pnl_A_list) + 1e-6),
            "Inventory_volatility_A": np.std(self.inventory_A_list),
            "Quote_aggressiveness_A": np.mean(self.aggressiveness_A),
            "Market_share_A": np.sum(self.fill_A_list) / total_arrivals,
            "Mean_reward_A": np.mean(self.reward_list_A),

            "PnL_mean_B2": np.mean(self.pnl_B2_list),
            "PnL_std_B2": np.std(self.pnl_B2_list),
            "Sharpe_ratio_B2": np.mean(self.pnl_B2_list) / (np.std(self.pnl_B2_list) + 1e-6),
            "Inventory_volatility_B2": np.std(self.inventory_B2_list),
            "Quote_aggressiveness_B2": np.mean(self.aggressiveness_B2),
            "Market_share_B2": np.sum(self.fill_B2_list) / total_arrivals,
            "Mean_reward_B2": np.mean(self.reward_list_B2),

            # Market/system-level
            "Avg_spread": np.mean(self.spread_list),
            "Price_volatility": np.std(self.price_list),
            "Fill_ratio": (np.sum(self.fill_A_list) + np.sum(self.fill_B2_list)) / total_arrivals,
            "Zero_fill_steps": self.zero_fill_steps,

            # Interaction/systemic
            "Joint_drawdown_ratio": np.mean(self.drawdown_overlap),
            "Herding_ratio": np.mean(self.quote_similarity),
            "Inventory_divergence": np.std(np.array(self.inventory_A_list) - np.array(self.inventory_B2_list)),
            "Quote_distance_bid": np.mean(self.quote_distance_bid),
            "Quote_distance_ask": np.mean(self.quote_distance_ask),
            "Same_side_quote_ratio": np.mean(self.same_side_quote),
            "Fill_overlap_ratio": np.mean(self.fill_overlap)
        }

#B1+B2+fix adv
class MultiAgent_eval_Environment_B1_B2(MarketEnvironment):

    def __init__(self,
                 adversary_model=None,
                 agent_B1=None,
                 agent_B2=None,
                 herding_threshold=0.01,
                 seed=42,
                 max_fill_per_direction=3,
                 dt=0.005,
                 lam_bounds=(300, 500),
                 volatility_bounds=(0.2, 2.0),
                 inventory_bounds=(-500, 500),
                 zeta=0.0,
                 eta=0.01):

        super().__init__(seed=seed, max_fill_per_direction=max_fill_per_direction, dt=dt)

        self.lam_bounds = lam_bounds
        self.volatility_bounds = volatility_bounds
        self.inventory_bounds = inventory_bounds
        self.zeta = zeta
        self.eta = eta

        self.fixed_lam = 400.0
        self.fixed_volatility = 1.1

        self.inventory_B1 = 0.0
        self.cash_B1 = 0.0
        self.pnl_B1 = 0.0
        self.prev_value_B1 = 0.0
        self.reward_list_B1 = []

        self.inventory_B2 = 0.0
        self.cash_B2 = 0.0
        self.pnl_B2 = 0.0
        self.prev_value_B2 = 0.0
        self.reward_list_B2 = []

        # === Metric buffers ===
        self.pnl_B1_list = []
        self.pnl_B2_list = []
        self.spread_list = []
        self.price_list = []

        self.inventory_B1_list = []
        self.inventory_B2_list = []

        self.aggressiveness_B1 = []
        self.aggressiveness_B2 = []

        self.fill_B1_list = []
        self.fill_B2_list = []

        self.arrival_list = []
        self.zero_fill_steps = 0

        self.drawdown_overlap = []
        self.herding_threshold = herding_threshold
        self.quote_similarity = []
        self.quote_distance_bid = []
        self.quote_distance_ask = []
        self.same_side_quote = []
        self.fill_overlap = []

        if adversary_model:
            self.adversary_model = PPO.load(adversary_model)
            print("Adversary model loaded. Using strategic market parameters.")
        else:
            self.adversary_model = None
            self.arrival_model.lam = self.fixed_lam
            self.midprice_model.volatility = self.fixed_volatility
            print(f"No adversary model loaded. Using fixed λ={self.fixed_lam}, volatility={self.fixed_volatility}")

        self.agent_B1 = PPO.load(agent_B1) if agent_B1 else None
        self.agent_B2 = PPO.load(agent_B2) if agent_B2 else None

    def reset(self):
        super().reset()
        # === Agents ===
        self.inventory_B1 = 0.0
        self.cash_B1 = 0.0
        self.pnl_B1 = 0.0
        self.prev_value_B1 = 0.0
        self.reward_list_B1.clear()

        self.inventory_B2 = 0.0
        self.cash_B2 = 0.0
        self.pnl_B2 = 0.0
        self.prev_value_B2 = 0.0
        self.reward_list_B2.clear()

        # === Metrics ===
        self.pnl_B1_list.clear()
        self.pnl_B2_list.clear()
        self.spread_list.clear()
        self.price_list.clear()
        self.inventory_B1_list.clear()
        self.inventory_B2_list.clear()
        self.aggressiveness_B1.clear()
        self.aggressiveness_B2.clear()
        self.fill_B1_list.clear()
        self.fill_B2_list.clear()
        self.arrival_list.clear()
        self.zero_fill_steps = 0

        self.drawdown_overlap.clear()
        self.quote_similarity.clear()
        self.quote_distance_bid.clear()
        self.quote_distance_ask.clear()
        self.same_side_quote.clear()
        self.fill_overlap.clear()

        return self.get_state()

    def step_with_agent(self, final=False):
        if self.adversary_model:
            obs = np.array([self.price, self.time], dtype=np.float32).reshape(1, -1)
            adv_action = self.adversary_model.predict(obs, deterministic=False)[0]
            adv_action = np.array(adv_action).flatten()
            lam = self.lam_bounds[0] + (adv_action[0] + 1.0) * 0.5 * (self.lam_bounds[1] - self.lam_bounds[0])
            volatility = self.volatility_bounds[0] + (adv_action[1] + 1.0) * 0.5 * (self.volatility_bounds[1] - self.volatility_bounds[0])
            self.arrival_model.lam = np.clip(lam, *self.lam_bounds)
            self.midprice_model.volatility = np.clip(volatility, *self.volatility_bounds)

        B1_obs = np.array([self.price, self.time, self.inventory_B1, self.cash_B1], dtype=np.float32).reshape(1, -1)
        action_B1 = self.agent_B1.predict(B1_obs, deterministic=False)[0]
        action_B1 = 2.5 * (np.array(action_B1).flatten() + 1.0)

        B2_obs = np.array([self.price, self.time, self.inventory_B2, self.cash_B2], dtype=np.float32).reshape(1, -1)
        action_B2 = self.agent_B2.predict(B2_obs, deterministic=False)[0]
        action_B2 = 2.5 * (np.array(action_B2).flatten() + 1.0)

        B1_bid_price = self.price - action_B1[0]
        B1_ask_price = self.price + action_B1[1]
        B2_bid_price = self.price - action_B2[0]
        B2_ask_price = self.price + action_B2[1]

        agent_quotes = {
            'MM_B1': {'bid_price': B1_bid_price, 'ask_price': B1_ask_price},
            'MM_B2': {'bid_price': B2_bid_price, 'ask_price': B2_ask_price}
        }

        self.max_fill_override = {
            'MM_B1': {
                'max_bid_fill': max(0, self.inventory_bounds[1] - self.inventory_B1),
                'max_ask_fill': max(0, self.inventory_B1 - self.inventory_bounds[0])
            },
            'MM_B2': {
                'max_bid_fill': max(0, self.inventory_bounds[1] - self.inventory_B2),
                'max_ask_fill': max(0, self.inventory_B2 - self.inventory_bounds[0])
            }
        }

        market_state, total_arrivals, agent_fills = super().step(agent_quotes)

        def update_agent(fill, bid_offset, ask_offset):
            bid_fill = fill.get('bid_fills', 0)
            ask_fill = fill.get('ask_fills', 0)
            bid_price = fill.get('bid_price', self.price - bid_offset)
            ask_price = fill.get('ask_price', self.price + ask_offset)
            return bid_fill, ask_fill, bid_price, ask_price

        B1_bid_fill, B1_ask_fill, B1_exec_bid, B1_exec_ask = update_agent(agent_fills.get('MM_B1', {}), action_B1[0], action_B1[1])
        B2_bid_fill, B2_ask_fill, B2_exec_bid, B2_exec_ask = update_agent(agent_fills.get('MM_B2', {}), action_B2[0], action_B2[1])

        self.inventory_B1 += B1_bid_fill - B1_ask_fill
        self.cash_B1 += B1_ask_fill * B1_exec_ask - B1_bid_fill * B1_exec_bid
        self.pnl_B1 = self.cash_B1 + self.inventory_B1 * self.price
        self.pnl_B1_list.append(self.pnl_B1)

        self.inventory_B2 += B2_bid_fill - B2_ask_fill
        self.cash_B2 += B2_ask_fill * B2_exec_ask - B2_bid_fill * B2_exec_bid
        self.pnl_B2 = self.cash_B2 + self.inventory_B2 * self.price
        self.pnl_B2_list.append(self.pnl_B2)

        pnl_diff_B1 = self.pnl_B1 - self.prev_value_B1
        self.prev_value_B1 = self.pnl_B1
        self.reward_list_B1.append(MMB1_reward(pnl_diff_B1, self.inventory_B1, final, self.inventory_bounds[1], self.zeta, self.eta))

        pnl_diff_B2 = self.pnl_B2 - self.prev_value_B2
        self.prev_value_B2 = self.pnl_B2
        self.reward_list_B2.append(MMB2_reward(pnl_diff_B1, self.inventory_B1, final, self.inventory_bounds[1], self.zeta, self.eta)) #opposite of B1

        self.spread_list.append((B1_ask_price - B1_bid_price + B2_ask_price - B2_bid_price) / 2)
        self.price_list.append(self.price)
        self.inventory_B1_list.append(self.inventory_B1)
        self.inventory_B2_list.append(self.inventory_B2)
        self.aggressiveness_B1.append((action_B1[0] + action_B1[1]) / 2)
        self.aggressiveness_B2.append((action_B2[0] + action_B2[1]) / 2)
        self.fill_B1_list.append(B1_bid_fill + B1_ask_fill)
        self.fill_B2_list.append(B2_bid_fill + B2_ask_fill)
        self.arrival_list.append(total_arrivals)
        if (B1_bid_fill + B1_ask_fill + B2_bid_fill + B2_ask_fill) == 0:
            self.zero_fill_steps += 1

        self.drawdown_overlap.append(int(pnl_diff_B1 < 0 and pnl_diff_B2 < 0))
        self.quote_similarity.append(int(abs(B1_bid_price - B2_bid_price) < self.herding_threshold and abs(B1_ask_price - B2_ask_price) < self.herding_threshold))
        self.quote_distance_bid.append(abs(B1_bid_price - B2_bid_price))
        self.quote_distance_ask.append(abs(B1_ask_price - B2_ask_price))
        self.same_side_quote.append(int((B1_bid_price > self.price and B2_bid_price > self.price) or (B1_ask_price < self.price and B2_ask_price < self.price)))
        self.fill_overlap.append(int((B1_bid_fill > 0 and B2_bid_fill > 0) or (B1_ask_fill > 0 and B2_ask_fill > 0)))

        return self.get_state(), total_arrivals

    def evaluate_metrics(self):
        total_arrivals = np.sum(self.arrival_list) + 1e-6
        return {
            # Agent-level
            "PnL_mean_B1": np.mean(self.pnl_B1_list),
            "PnL_std_B1": np.std(self.pnl_B1_list),
            "Sharpe_ratio_B1": np.mean(self.pnl_B1_list) / (np.std(self.pnl_B1_list) + 1e-6),
            "Inventory_volatility_B1": np.std(self.inventory_B1_list),
            "Quote_aggressiveness_B1": np.mean(self.aggressiveness_B1),
            "Market_share_B1": np.sum(self.fill_B1_list) / total_arrivals,
            "Mean_reward_B1": np.mean(self.reward_list_B1),

            "PnL_mean_B2": np.mean(self.pnl_B2_list),
            "PnL_std_B2": np.std(self.pnl_B2_list),
            "Sharpe_ratio_B2": np.mean(self.pnl_B2_list) / (np.std(self.pnl_B2_list) + 1e-6),
            "Inventory_volatility_B2": np.std(self.inventory_B2_list),
            "Quote_aggressiveness_B2": np.mean(self.aggressiveness_B2),
            "Market_share_B2": np.sum(self.fill_B2_list) / total_arrivals,
            "Mean_reward_B2": np.mean(self.reward_list_B2),

            # Market/system-level
            "Avg_spread": np.mean(self.spread_list),
            "Price_volatility": np.std(self.price_list),
            "Fill_ratio": (np.sum(self.fill_B1_list) + np.sum(self.fill_B2_list)) / total_arrivals,
            "Zero_fill_steps": self.zero_fill_steps,

            # Interaction/systemic
            "Joint_drawdown_ratio": np.mean(self.drawdown_overlap),
            "Herding_ratio": np.mean(self.quote_similarity),
            "Inventory_divergence": np.std(np.array(self.inventory_B1_list) - np.array(self.inventory_B2_list)),
            "Quote_distance_bid": np.mean(self.quote_distance_bid),
            "Quote_distance_ask": np.mean(self.quote_distance_ask),
            "Same_side_quote_ratio": np.mean(self.same_side_quote),
            "Fill_overlap_ratio": np.mean(self.fill_overlap)
        }


