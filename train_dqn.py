import argparse
import yaml
from utils.wrappers import wrap_action_d_plus_a
from register_envs import register_envs
from utils.callbacks import SaveEnvStatsCallback, HParamCallback
from utils.utils import ROLES
import gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
import time


def main():
    parser = argparse.ArgumentParser()

    # Adding required argument
    parser.add_argument(
        "-g",
        "--global-info",
        type=bool,
        required=True,
        help="Whether to return global info of the entire supply chain in the decentralized setting. This argument is ignored in the centralized setting",
    )
    parser.add_argument("-p", "--hyperparameters", help="Path to the experiment setup file (.yaml)", required=True)
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
    parser.add_argument("--scenario", type=str, required=True, help="complex or basic")

    # Read arguments from command line
    args = parser.parse_args()

    with open(args.hyperparameters) as fh:
        setup = yaml.load(fh, Loader=yaml.FullLoader)

    if args.role == "MultiFacility":
        raise NotImplementedError("For now the implemented DQN only supports the decentralized setting")

    if args.scenario == "basic":
        demand_type = "Normal"
        action_range = [0, 20]
    elif args.scenario == "complex":
        demand_type = "Uniform"
        action_range = [0, 16]
    else:
        raise ValueError

    params = setup["hyperparameters"]["dqn"]
    print(args)
    print(params)
    env_name = f"BeerGame{demand_type}{args.role}{'FullInfo'*args.global_info}Discrete-v0"

    # Register different versions of the beer game to the Gym Registry, so the environment can be created using gym.make
    register_envs()

    n_env = 8

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

    for run in range(setup["runs"]):

        exp_name = f"{args.name}_DQN_{args.role}_{args.scenario}{'_FullInfo'*args.global_info}_{args.ordering_rule}_{run}_{time.time_ns()}"
        env = VecNormalize(make_vec_env(env_factory, n_env), clip_obs=100, clip_reward=1000)

        policy_kwargs = dict(net_arch=[params["network_width"] * params["num_layers"]])

        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=params["learning_rate"],
            train_freq=params["train_freq"],
            verbose=1,
            gamma=params["gamma"],
            policy_kwargs=policy_kwargs,
            exploration_final_eps=0.05,
            exploration_fraction=0.2,
            target_update_interval=params["target_update_interval"],
            gradient_steps=1,
            buffer_size=params["buffer_size"],
            tensorboard_log=f"./tensorboard/",
        )

        eval_callback = EvalCallback(
            env,
            callback_on_new_best=SaveEnvStatsCallback(env_save_path=f"./best_models/{exp_name}/"),
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
            callback=[hparam_callback, eval_callback],
            reset_num_timesteps=True,
        )


if __name__ == "__main__":
    main()
