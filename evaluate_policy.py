import os
import pandas as pd
from evaluate_env import (
    MMA_eval_Environment,
    MMB1_eval_Environment,
    MultiAgent_eval_Environment_A_B1,
    MultiAgent_eval_Environment_A_B2,
    MultiAgent_eval_Environment_B1_B2
)

def evaluate_agent(env_class, env_kwargs, env_name, n_episodes=10, max_steps=200):
    all_metrics = []
    for ep in range(n_episodes):
        env = env_class(**env_kwargs, seed=ep)
        env.reset()
        for t in range(max_steps):
            _ = env.step_with_agent(final=(t == max_steps - 1))
        metrics = env.evaluate_metrics()
        metrics["episode"] = ep
        metrics["env"] = env_name
        all_metrics.append(metrics)

    df = pd.DataFrame(all_metrics)
    output_file = f"results_{env_name.replace(' ', '_')}.csv"
    df.to_csv(output_file, index=False)
    print(f"Evaluation complete. Results saved to {output_file}")
    return df

def test_A(env_name, n_episodes, agent_A=None, adversary_model=None):
    return evaluate_agent(
        env_class=MMA_eval_Environment,
        env_kwargs={"agent_A": agent_A, "adversary_model": adversary_model},
        env_name=env_name,
        n_episodes=n_episodes
    )

def test_B1(env_name, n_episodes, agent_B1=None, adversary_model=None):
    return evaluate_agent(
        env_class=MMB1_eval_Environment,
        env_kwargs={"agent_B1": agent_B1, "adversary_model": adversary_model},
        env_name=env_name,
        n_episodes=n_episodes
    )

def test_A_B1(env_name, n_episodes, herding_threshold=0.01, agent_A=None, agent_B1=None, adversary_model=None):
    return evaluate_agent(
        env_class=MultiAgent_eval_Environment_A_B1,
        env_kwargs={"agent_A": agent_A, "agent_B1": agent_B1, "herding_threshold": herding_threshold, "adversary_model": adversary_model},
        env_name=env_name,
        n_episodes=n_episodes
    )

def test_A_B2(env_name, n_episodes, herding_threshold=0.01, agent_A=None, agent_B2=None, adversary_model=None):
    return evaluate_agent(
        env_class=MultiAgent_eval_Environment_A_B2,
        env_kwargs={"agent_A": agent_A, "agent_B2": agent_B2, "herding_threshold": herding_threshold, "adversary_model": adversary_model},
        env_name=env_name,
        n_episodes=n_episodes
    )

def test_B1_B2(env_name, n_episodes, herding_threshold=0.01, agent_B1=None, agent_B2=None, adversary_model=None):
    return evaluate_agent(
        env_class=MultiAgent_eval_Environment_B1_B2,
        env_kwargs={"agent_B1": agent_B1, "agent_B2": agent_B2, "herding_threshold": herding_threshold, "adversary_model": adversary_model},
        env_name=env_name,
        n_episodes=n_episodes
    )

