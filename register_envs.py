from typing import List
import gym
from envs import make_beer_game_uniform_multi_facility, make_beer_game_normal_multi_facility
from utils.wrappers import wrap_action_d_plus_a


# For td3 hyperparameters tuning
def uniform_multi_facility_dplusa_env_factory():
    env = make_beer_game_uniform_multi_facility(
        agent_managed_facilities=["retailer", "wholesaler", "distributor", "manufacturer"],
        box_action_space=True,
        random_init=True,
    )
    env = wrap_action_d_plus_a(env, offset=-8, lb=0, ub=16)
    return env


# For td3 hyperparameters tuning
def uniform_single_facility_retailer_dplusa_env_factory():
    env = make_beer_game_uniform_multi_facility(
        agent_managed_facilities=["retailer"], box_action_space=True, random_init=True
    )
    env = wrap_action_d_plus_a(env, offset=-8, lb=0, ub=16)
    return env


# For dqn and a2c hyperparameters tuning
def uniform_single_facility_retailer_discrete_dplusa_env_factory():
    env = make_beer_game_uniform_multi_facility(
        agent_managed_facilities=["retailer"], box_action_space=False, random_init=True
    )
    env = wrap_action_d_plus_a(env, offset=-8, lb=0, ub=16)
    return env


def uniform_env_factory(role: List[str], discrete: bool, global_observable: bool):
    def func():
        env = make_beer_game_uniform_multi_facility(
            agent_managed_facilities=role,
            box_action_space=~discrete,
            random_init=True,
            global_observable=global_observable,
        )
        return env

    return func


def normal_env_factory(role: List[str], discrete: bool, global_observable: bool):
    def func():
        env = make_beer_game_normal_multi_facility(
            agent_managed_facilities=role,
            box_action_space=~discrete,
            random_init=True,
            global_observable=global_observable,
        )
        return env

    return func


def normal_multi_facility_dplusa_env_factory():
    env = make_beer_game_normal_multi_facility(
        agent_managed_facilities=["retailer", "wholesaler", "distributor", "manufacturer"],
        box_action_space=True,
        random_init=True,
    )
    env = wrap_action_d_plus_a(env, offset=-10, lb=0, ub=20)
    return env


def register_envs():

    gym.envs.register(
        id="BeerGameUniformMultiFacilityDPlusA-v0",
        entry_point=uniform_multi_facility_dplusa_env_factory,
        max_episode_steps=100,
    )
    gym.envs.register(
        id="BeerGameNormalMultiFacilityDPlusA-v0",
        entry_point=normal_multi_facility_dplusa_env_factory,
        max_episode_steps=100,
    )

    gym.envs.register(
        id="BeerGameUniformRetailerDPlusA-v0",
        entry_point=uniform_single_facility_retailer_dplusa_env_factory,
        max_episode_steps=100,
    )
    gym.envs.register(
        id="BeerGameUniformRetailerDiscreteDPlusA-v0",
        entry_point=uniform_single_facility_retailer_discrete_dplusa_env_factory,
        max_episode_steps=100,
    )

    gym.envs.register(
        id="BeerGameUniformMultiFacilityFullInfo-v0",
        entry_point=uniform_env_factory(
            role=["retailer", "wholesaler", "distributor", "manufacturer"], discrete=False, global_observable=True
        ),
        max_episode_steps=100,
    )

    gym.envs.register(
        id="BeerGameNormalMultiFacilityFullInfo-v0",
        entry_point=normal_env_factory(
            role=["retailer", "wholesaler", "distributor", "manufacturer"], discrete=False, global_observable=True
        ),
        max_episode_steps=100,
    )
    roles = {
        "Retailer": ["retailer"],
        "Wholesaler": ["wholesaler"],
        "Distributor": ["distributor"],
        "Manufacturer": ["manufacturer"],
    }
    for key, role in roles.items():

        # uniform (complex scenario)
        gym.envs.register(
            id=f"BeerGameUniform{key}-v0",
            entry_point=uniform_env_factory(role=role, discrete=False, global_observable=False),
            max_episode_steps=100,
        )

        gym.envs.register(
            id=f"BeerGameUniform{key}Discrete-v0",
            entry_point=uniform_env_factory(role=role, discrete=True, global_observable=False),
            max_episode_steps=100,
        )
        gym.envs.register(
            id=f"BeerGameUniform{key}FullInfo-v0",
            entry_point=uniform_env_factory(role=role, discrete=False, global_observable=True),
            max_episode_steps=100,
        )
        gym.envs.register(
            id=f"BeerGameUniform{key}DiscreteFullInfo-v0",
            entry_point=uniform_env_factory(role=role, discrete=True, global_observable=True),
            max_episode_steps=100,
        )

        # normal (basic scenario)
        gym.envs.register(
            id=f"BeerGameNormal{key}-v0",
            entry_point=normal_env_factory(role=role, discrete=False, global_observable=False),
            max_episode_steps=100,
        )

        gym.envs.register(
            id=f"BeerGameNormal{key}Discrete-v0",
            entry_point=normal_env_factory(role=role, discrete=True, global_observable=False),
            max_episode_steps=100,
        )

        gym.envs.register(
            id=f"BeerGameNormal{key}FullInfo-v0",
            entry_point=normal_env_factory(role=role, discrete=False, global_observable=True),
            max_episode_steps=100,
        )

        gym.envs.register(
            id=f"BeerGameNormal{key}DiscreteFullInfo-v0",
            entry_point=normal_env_factory(role=role, discrete=True, global_observable=True),
            max_episode_steps=100,
        )
