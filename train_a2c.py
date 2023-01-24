import argparse
import yaml
from utils.wrappers import wrap_action_d_plus_a
from register_envs import register_envs
from utils.callbacks import SaveEnvStatsCallback
import gym
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import EvalCallback


def main():
    parser = argparse.ArgumentParser()

    # Adding required argument
    parser.add_argument("-e", help="Path to the experiment setup file (.yaml)", required=True)
    parser.add_argument(
        "--name",
        help="Name of the experiment. Used as a prefix saving log files and models to avoid "
        "overwriting previous experiment outputs",
        default="",
        required=False,
    )

    # Read arguments from command line
    args = parser.parse_args()

    with open(args.e) as fh:
        setup = yaml.load(fh, Loader=yaml.FullLoader)

    environment = setup["environment"]

    if environment["role"] == "MultiFacility":
        raise NotImplementedError("For now the implemented DQN only supports the decentralized setting")

    if environment["scenario"] == "basic":
        demand_type = "Normal"
        action_range = [0, 20]
    elif environment["scenario"] == "complex":
        demand_type = "Uniform"
        action_range = [0, 16]
    else:
        raise ValueError

    params = setup["hyperparameters"]["a2c"]

    env_name = f"BeerGame{demand_type}{environment['role']}{'FullInfo'*environment['global_info']}Discrete-v0"

    # Register different versions of the beer game to the Gym Registry, so the environment can be created using gym.make
    register_envs()

    n_env = 2

    def env_factory() -> gym.Env:
        if environment["ordering_rule"] == "d+a":
            return wrap_action_d_plus_a(
                gym.make(env_name),
                offset=-(action_range[1] - action_range[0]) / 2,
                lb=action_range[0],
                ub=action_range[1],
            )
        elif environment["ordering_rule"] == "a":
            return gym.make(env_name)
        else:
            raise ValueError

    for run in range(setup["runs"]):

        exp_name = f"{args.name}_A2C_{environment['role']}_{environment['scenario']}_{'FullInfo'*environment['global_info']}_{environment['ordering_rule']}_{run}"
        env = VecNormalize(make_vec_env(env_factory, n_env), clip_obs=100, clip_reward=1000)
        eval_env = VecNormalize(make_vec_env(env_factory, n_env), clip_obs=100, clip_reward=1000)

        policy_kwargs = dict(net_arch=[params["network_width"] * params["num_layers"]])

        model = A2C(
            "MlpPolicy",
            env,
            learning_rate=params["learning_rate"],
            verbose=1,
            gamma=params["gamma"],
            gae_lambda=params["gae_lambda"],
            ent_coef=params["ent_coef"],
            vf_coef=params["vf_coef"],
            max_grad_norm=params["max_grad_norm"],
            policy_kwargs=policy_kwargs,
            tensorboard_log=f"./tensorboard/",
        )
        eval_callback = EvalCallback(
            eval_env,
            callback_on_new_best=SaveEnvStatsCallback(env_save_path=f"./best_models/{exp_name}/"),
            best_model_save_path=f"./best_models/{exp_name}/",
            log_path=f"./logs/{exp_name}/",
            eval_freq=5000,
            n_eval_episodes=100,
            deterministic=True,
            render=False,
        )

        model.learn(
            total_timesteps=setup["max_time_steps"],
            tb_log_name=exp_name,
            callback=eval_callback,
            reset_num_timesteps=True,
        )


if __name__ == "__main__":
    main()
