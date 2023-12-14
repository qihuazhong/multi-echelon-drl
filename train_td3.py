import argparse
import yaml
import numpy as np
from utils.wrappers import wrap_action_d_plus_a, wrap_action_a
from utils.heuristics import BaseStockPolicy
from utils.utils import ROLES
from register_envs import register_envs
from hge import HgeTD3
from utils.callbacks import HgeRateCallback, SaveEnvStatsCallback, HParamCallback
import gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import EvalCallback
import time


def main():
    parser = argparse.ArgumentParser()

    # Adding required argument
    parser.add_argument(
        "--cost-type",
        type=str,
        required=True,
        help="Should be one of 'general', 'clark-scarf'",
    )

    parser.add_argument("--state-version", type=str, default="v0", required=False)

    parser.add_argument(
        "-i",
        "--info-scope",
        type=str,
        required=True,
        help="Should be one of 'local', 'global'. Whether to return global info of the entire supply chain in the decentralized setting. This argument is ignored in the centralized setting",
    )
    parser.add_argument(
        "-p",
        "--hyperparameters",
        help="Path to the experiment setup file (.yaml)",
        required=True,
    )
    parser.add_argument(
        "--name",
        help="Name of the experiment. Used as a prefix for saving log files and models to avoid "
        "overwriting previous experiment outputs",
        default="",
        required=False,
    )
    parser.add_argument("--ordering-rule", type=str, required=True, help="'a' or 'd+a'")
    parser.add_argument(
        "--role",
        type=str,
        required=True,
        help="Should be one of 'Retailer', 'Wholesaler', 'Distributor', 'Manufacturer' or 'MultiFacility' (Centralized control)",
        choices=ROLES,
    )
    parser.add_argument(
        "--scenario", type=str, required=True, help="complex, basic or dunnhumby"
    )
    # Read arguments from command line
    args = parser.parse_args()

    with open(args.hyperparameters) as fh:
        setup = yaml.load(fh, Loader=yaml.FullLoader)

    if args.scenario == "basic":
        benchmark_target_stock_level = [48, 43, 41, 30]
        demand_type = "Normal"
        action_range = [0, 20]
    elif args.scenario == "complex":
        benchmark_target_stock_level = [19, 20, 20, 14]
        demand_type = "Uniform"
        action_range = [0, 16]
    elif args.scenario == "dunnhumby":
        benchmark_target_stock_level = [529, 451, 437, 318]
        demand_type = "Dunnhumby"
        action_range = [0, 200]
    else:
        raise ValueError

    params = setup["hyperparameters"]["td3"]
    print(args)
    print(params)

    if params["hge_rate_at_start"] > 0 and args.role != "MultiFacility":
        raise NotImplementedError(
            "For now the TD3 with HGE only supports centralized setting"
        )

    env_name = f"BeerGame{'CSCost' * (args.cost_type == 'clark-scarf')}{demand_type}{args.role}{'FullInfo' * (args.info_scope == 'global')}-{args.state_version}"

    bsp = BaseStockPolicy(
        target_levels=benchmark_target_stock_level,
        state_name_to_index={
            "on_hand": 0,
            "unreceived_pipeline": [3, 4, 5, 6],
            "unfilled_demand": 1,
            "latest_demand": 2,
        },
        state_dim_per_facility=7,
        lb=action_range[0],
        ub=action_range[1],
        rule=args.ordering_rule,
    )

    n_env = 8

    def env_factory() -> gym.Env:
        # Register different versions of the beer game to the Gym Registry, so the environment can be created using gym.make
        register_envs()
        if args.ordering_rule == "d+a":
            return wrap_action_d_plus_a(
                gym.make(env_name),
                offset=-(action_range[1] - action_range[0]) / 2,
                lb=action_range[0],
                ub=action_range[1],
            )
        elif args.ordering_rule == "a":
            # return gym.make(env_name)
            return wrap_action_a(
                gym.make(env_name),
                offset=-(action_range[1] - action_range[0]) / 2,
                lb=action_range[0],
                ub=action_range[1],
            )
        else:
            raise ValueError

    for run in range(setup["runs"]):
        exp_name = f"{args.name}_TD3_{args.role}_{args.scenario}{'_CSCost'*(args.cost_type=='clark-scarf')}{'_FullInfo'*(args.info_scope=='global')}{'_hge'*(params['hge_rate_at_start']>0)}_{args.ordering_rule}_{args.state_version}_{run}_{time.time_ns()}"
        env = VecNormalize(
            make_vec_env(env_factory, n_env, vec_env_cls=SubprocVecEnv),
            clip_obs=100,
            clip_reward=1000,
        )
        # eval_env = VecNormalize(make_vec_env(env_factory, n_env), clip_obs=100, clip_reward=1000)

        policy_kwargs = dict(net_arch=[params["network_width"]] * params["num_layers"])

        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=params["action_noise_std"] * np.ones(n_actions),
        )
        model = HgeTD3(
            "MlpPolicy",
            env,
            batch_size=params["batch_size"],
            learning_rate=params["learning_rate"],
            tau=params["tau"],
            train_freq=params["train_freq"],
            action_noise=action_noise,
            verbose=1,
            gamma=params["gamma"],
            policy_kwargs=policy_kwargs,
            tensorboard_log=f"./tensorboard/",
            hge_rate=params["hge_rate_at_start"],
            heuristic=bsp,
            learning_starts=50000,
        )

        hge_callback = HgeRateCallback(
            mu_start=params["hge_rate_at_start"],
            action_noise_std=params["action_noise_std"],
            action_noise_annealing=True,
        )
        eval_callback = EvalCallback(
            env,
            callback_on_new_best=SaveEnvStatsCallback(
                env_save_path=f"./best_models/{exp_name}/"
            ),
            best_model_save_path=f"./best_models/{exp_name}/",
            log_path=f"./logs/{exp_name}/",
            eval_freq=5000,
            n_eval_episodes=100,
            deterministic=True,
            render=False,
        )
        hparam_callback = HParamCallback(hparam_dict=params)
        print("Start training")
        model.learn(
            total_timesteps=setup["max_time_steps"],
            tb_log_name=exp_name,
            callback=[hge_callback, eval_callback, hparam_callback],
            reset_num_timesteps=True,
        )


if __name__ == "__main__":
    main()
