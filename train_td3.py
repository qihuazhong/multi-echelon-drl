import sys
import yaml
import numpy as np
from utils.wrappers import wrap_action_d_plus_a
from utils.heuristics import BaseStockPolicy
from register_envs import register_envs

from hge import HgeTD3
from utils.callbacks import HgeRateCallback, SaveEnvStatsCallback

import gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import EvalCallback


def main(setup_file: str):
    with open(setup_file) as fh:
        setup = yaml.load(fh, Loader=yaml.FullLoader)

    environment = setup['environment']
    if environment['scenario'] == 'basic':
        benchmark_target_stock_level = [48, 43, 41, 30]
        demand_type = 'Normal'
        action_range = [0, 20]
    elif environment['scenario'] == 'complex':
        benchmark_target_stock_level = [19, 20, 20, 14]
        demand_type = 'Uniform'
        action_range = [0, 16]
    else:
        raise ValueError

    params = setup['hyperparameters']['td3']

    if params['hge_rate_at_start'] > 0 and environment['role'] != 'MultiFacility':
        raise NotImplementedError('For now the TD3 with HGE only supports centralized setting')

    env_name = f"BeerGame{demand_type}{environment['role']}-v0"
    bsp = BaseStockPolicy(target_levels=benchmark_target_stock_level,
                          array_index={'on_hand': 0, 'unreceived_pipeline': [3, 4, 5, 6],
                                       'unfilled_demand': 1, 'latest_demand': 2},
                          state_dim_per_facility=7,
                          lb=action_range[0], ub=action_range[1], rule=environment['ordering_rule'])

    # Register different versions of the beer game to the Gym Registry, so the environment can be created using gym.make
    register_envs()

    n_env = 2

    def env_factory() -> gym.Env:
        if environment['ordering_rule'] == 'd+a':
            return wrap_action_d_plus_a(gym.make(env_name), offset=-(action_range[1] - action_range[0]) / 2,
                                        lb=action_range[0], ub=action_range[1])
        elif environment['ordering_rule'] == 'a':
            return gym.make(env_name)
        else:
            raise ValueError

    for run in range(setup['runs']+1):
        exp_name = f"TD3_{environment['role']}_{environment['scenario']}_{environment['ordering_rule']}_{run}"
        env = VecNormalize(make_vec_env(env_factory, n_env), clip_obs=100, clip_reward=1000)
        eval_env = VecNormalize(make_vec_env(env_factory, n_env), clip_obs=100, clip_reward=1000)

        policy_kwargs = dict(net_arch=[params['network_width'] * 2])

        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions),
                                         sigma=params['action_noise_std'] * np.ones(n_actions))
        model = HgeTD3('MlpPolicy', env,
                       batch_size=params['batch_size'],
                       learning_rate=params['learning_rate'],
                       tau=params['tau'],
                       train_freq=params['train_freq'],
                       action_noise=action_noise,
                       verbose=1,
                       gamma=params['gamma'],
                       policy_kwargs=policy_kwargs,
                       tensorboard_log=f"./tensorboard/",
                       hge_rate=params['hge_rate_at_start'],
                       heuristic=bsp)

        hge_callback = HgeRateCallback(mu_start=params['hge_rate_at_start'])
        eval_callback = EvalCallback(eval_env,
                                     callback_on_new_best=SaveEnvStatsCallback(
                                         env_save_path=f"./best_models/{exp_name}/"),
                                     best_model_save_path=f"./best_models/{exp_name}/",
                                     log_path=f"./logs/{exp_name}/",
                                     eval_freq=5000,
                                     n_eval_episodes=50,
                                     deterministic=True,
                                     render=False)

        model.learn(total_timesteps=setup['max_time_steps'],
                    tb_log_name=exp_name,
                    callback=[hge_callback, eval_callback],
                    reset_num_timesteps=True)

        # save the vectorized environment (along with the normalizing statistics)
        env.save(f'./best_models/{exp_name}/vec_env')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        raise ValueError(f'Need the path to a setup file (YAML) as the first argument')
