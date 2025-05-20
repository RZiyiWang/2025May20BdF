import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from adversary_gym_env import AdversaryEnv


def make_train_env(n_envs=4, seed=42):
    envs = DummyVecEnv([lambda: Monitor(AdversaryEnv()) for _ in range(n_envs)])
    envs.seed(seed)
    return VecNormalize(envs, norm_obs=True, norm_reward=True)

def make_eval_env(n_envs=1, seed=42):
    envs = DummyVecEnv([lambda: Monitor(AdversaryEnv()) for _ in range(n_envs)])
    envs.seed(seed)
    return VecNormalize(envs, norm_obs=True, norm_reward=True)


class MarketLoggingCallback(BaseCallback):
    def __init__(self, reward_log, market_log, console_log, verbose=0, print_freq=1000):
        super().__init__(verbose)
        self.reward_log = reward_log
        self.market_log = market_log
        self.console_log = console_log
        self.timestep = 0
        self.print_freq = print_freq

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", [])

        for i, info in enumerate(infos):
            if i != 0:
                continue  

            if self.timestep % self.print_freq != 0:
                self.timestep += 1
                continue  

            state = info.get("market_state", {})
            total_arrivals = info.get("total_arrivals", 0)


            self.reward_log.write(f"{self.timestep}_env0,{rewards[i]:.6f}\n")
            self.market_log.write(f"{self.timestep}_env0," + ",".join(str(state.get(k, 0.0)) for k in [
                "price", "time", "lam", "volatility",
                "inventory_Adversary", "cash_Adversary", "pnl_Adversary",
                "bid_price_Adversary", "ask_price_Adversary",
                "bid_fill_Adversary", "ask_fill_Adversary"
            ]) + f",{total_arrivals}\n")


            line_prefix = f"timestep={self.timestep}_env0 | "
            line = (
                f"price={state.get('price', 0.0):.2f} | "
                f"time={state.get('time', 0.0):.3f} | "
                f"lam={state.get('lam', 0.0):.1f} | "
                f"volatility={state.get('volatility', 0.0):.2f} | "
                f"inventory={state.get('inventory_Adversary', 0.0)} | "
                f"cash={state.get('cash_Adversary', 0.0):.2f} | "
                f"pnl={state.get('pnl_Adversary', 0.0):.2f} | "
                f"bid={state.get('bid_price_Adversary', 0.0):.2f} | "
                f"ask={state.get('ask_price_Adversary', 0.0):.2f} | "
                f"bid_fill={state.get('bid_fill_Adversary', 0.0)} | "
                f"ask_fill={state.get('ask_fill_Adversary', 0.0)} | "
                f"total_arrivals={total_arrivals}"
            )

            print(line)
            self.console_log.write(line_prefix + line + "\n")

            self.timestep += 1

        return True


# === 主训练函数 ===
def run_training(
    project_name: str,         
    run_type: str,             
    train_steps: int,          # Number of timesteps in the current training round
    train_total: int,          # Total training rounds so far (used for naming)
    load_path: str = None,     # Model loading path for resumed training (can be manually specified)
    seed: int = 123,
    eval_freq: int = 6400,     
    n_envs: int = 4,           
    print_freq: int = 1000
):


    continue_training = (run_type == "continue")

    # adv1_continue_10_total30
    model_name = f"{project_name}_{run_type}{train_steps}_total{train_total}"


    base_dir = os.path.join("Train_Adversary", project_name)
    model_dir = os.path.join(base_dir, "models")
    log_dir = os.path.join(base_dir, "logs", model_name)
    tb_dir = os.path.join(base_dir, "tensorboard", project_name)

    for d in (model_dir, log_dir, tb_dir):
        os.makedirs(d, exist_ok=True)


    reward_log = open(os.path.join(log_dir, "reward_log.csv"), "w")
    market_log = open(os.path.join(log_dir, "market_log.csv"), "w")
    console_log = open(os.path.join(log_dir, "market_log_console.txt"), "w")

    reward_log.write("timestep,reward\n")
    market_log.write("timestep," + ",".join([
        "price", "time", "lam", "volatility",
        "inventory_Adversary", "cash_Adversary", "pnl_Adversary",
        "bid_price_Adversary", "ask_price_Adversary",
        "bid_fill_Adversary", "ask_fill_Adversary",
        "total_arrivals"
    ]) + "\n")


    train_env = make_train_env(n_envs=n_envs, seed=seed)
    eval_env = make_eval_env(n_envs=1, seed=seed)


    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(model_dir, f"best_{model_name}"),
        log_path=os.path.join(log_dir, "eval"),
        eval_freq=eval_freq,
        n_eval_episodes=4,
        deterministic=True,
        verbose=1
    )

    logging_cb = MarketLoggingCallback(
        reward_log, market_log, console_log, print_freq=print_freq)


    model_path = os.path.join(model_dir, f"{model_name}.zip")
    if continue_training and load_path and os.path.exists(load_path):
        print(f"[INFO] Loading existing model from {load_path}")
        model = PPO.load(load_path, env=train_env, seed=seed)
    else:
        print("[INFO] Creating new PPO model")
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            seed=seed,
            verbose=1,
            learning_rate=1e-4,
            n_steps=400,
            batch_size=200,
            n_epochs=6,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.02,
            vf_coef=0.7,
            tensorboard_log=tb_dir,
        )


    print(f"[INFO] Training {model_name} for {train_steps} steps...")
    model.learn(
        total_timesteps=train_steps,
        callback=[eval_cb, logging_cb],
        reset_num_timesteps=not continue_training
    )


    model.save(model_path)
    print(f"[INFO] Model saved to {model_path}")

  
    reward_log.close()
    market_log.close()
    console_log.close()



if __name__ == "__main__":
        run_training(
        project_name="adv1",
        run_type="new",
        train_steps=3200,
        train_total=3200,
        load_path=None
    )


