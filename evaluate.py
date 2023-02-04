from typing import Dict, Type, List
import os

import numpy as np
import pandas as pd
from stable_baselines3.common.base_class import BaseAlgorithm

from utils.utils import ROLES
from utils.wrappers import wrap_action_d_plus_a
from register_envs import register_envs

import argparse
import gym
from stable_baselines3 import DQN, TD3, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from collections import namedtuple

Result = namedtuple("Result", ["algo", "model_dir", "info_scope", "ordering_rule", "role", "scenario", "mean_rewards"])


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-a", "--algo", help="DRL algorithm of the agent. One of [td3, a2c, dqn]", required=True)
    parser.add_argument("-m", "--models-dir", help="Path to directory the saved model(s)", required=True)
    parser.add_argument(
        "-n", "--n-eval-episodes", help="Number of episodes to evaluate per model", required=True, default=100, type=int
    )
    parser.add_argument(
        "-i",
        "--info-scope",
        type=str,
        required=True,
        help="Should be one of 'local', 'global'. Whether to return global info of the entire supply chain in the decentralized setting. This argument is ignored in the centralized setting",
    )
    parser.add_argument("--ordering-rule", type=str, required=True, help="'a' or 'd+a'")
    parser.add_argument(
        "--role",
        type=str,
        required=True,
        help="Should be one of 'Retailer', 'Wholesaler', 'Distributor', 'Manufacturer' or 'MultiFacility' (Centralized control)",
        choices=ROLES,
    )
    parser.add_argument("--scenario", type=str, required=True, help="complex or basic")
    # Read arguments from command line
    args = parser.parse_args()

    if args.scenario == "basic":
        demand_type = "Normal"
        action_range = [0, 20]
    elif args.scenario == "complex":
        demand_type = "Uniform"
        action_range = [0, 16]
    else:
        raise ValueError

    register_envs()

    # If td3, make the continuous environment, otherwise make the discrete environment
    if args.algo == "td3":
        env_name = f"BeerGame{demand_type}{args.role}{'FullInfo' * (args.info_scope=='global')}-v0"
    else:
        env_name = f"BeerGame{demand_type}{args.role}{'FullInfo' * (args.info_scope=='global')}Discrete-v0"
    print("Env id: ", env_name)

    def env_factory() -> gym.Env:
        if args.ordering_rule == "d+a":
            return wrap_action_d_plus_a(
                gym.make(env_name),
                offset=-(action_range[1] - action_range[0]) / 2,
                lb=action_range[0],
                ub=action_range[1],
            )
        elif args.ordering_rule == "a":
            return gym.make(env_name)
        else:
            raise ValueError

    algos: Dict[str, Type[BaseAlgorithm]] = {"td3": TD3, "a2c": A2C, "dqn": DQN}
    sub_dirs: List[str] = next(os.walk(args.models_dir + "/best_models/"))[1]
    experiment_prefix: str = f"_{args.algo.upper()}_{args.role}_{args.scenario}{'_FullInfo'*(args.info_scope=='global')}_{args.ordering_rule}"

    results: List[Result] = []
    mean_rewards = []
    for sub_dir in sub_dirs:
        if experiment_prefix in sub_dir:
            rewards = []
            env = make_vec_env(env_factory, n_envs=1)
            env = VecNormalize.load(f"{args.models_dir+'/best_models'}/{sub_dir}/best_env", env)

            # Check whether the training is complete
            evaluation_log = np.load(f"{args.models_dir+'/logs'}/{sub_dir}/evaluations.npz")
            if evaluation_log["timesteps"].max() < 10_000_000:
                print(f"Warning: Model {sub_dir} trained for {evaluation_log['timesteps'].max()} steps only")

            model = algos[args.algo].load(f"{args.models_dir+'/best_models'}/{sub_dir}/best_model.zip", env=env)

            for i in range(args.n_eval_episodes):
                np.random.seed(i)
                # Calling evaluate_policy() to evaluate one episode at a time so that the seed can be fixed properly
                mean_reward, std_reward = evaluate_policy(
                    model,
                    model.get_env(),
                    n_eval_episodes=1,
                )
                rewards.append(mean_reward)
            # print(f"Rewards: {rewards}")
            results.append(
                Result(
                    algo=args.algo,
                    model_dir=sub_dir,
                    info_scope=args.info_scope,
                    ordering_rule=args.ordering_rule,
                    role=args.role,
                    scenario=args.scenario,
                    mean_rewards=np.mean(rewards),
                )
            )
            print(f"\tModel: {sub_dir}: Mean reward:{np.mean(rewards)} over {len(rewards)} episodes")
            mean_rewards.append(np.mean(rewards))

    print(
        f"\tBest mean rewards {np.max(mean_rewards)}, average mean rewards over {len(mean_rewards)} models: {np.mean(mean_rewards)}"
    )

    df = pd.DataFrame(results)
    df.to_csv(f"./df_{args.algo}.csv", index=False, mode="a")


if __name__ == "__main__":
    main()
