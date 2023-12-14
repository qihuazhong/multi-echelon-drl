from typing import List
import gym
from envs import (
    make_beer_game_uniform_multi_facility,
    make_beer_game_normal_multi_facility,
    make_beer_game_dunnhumby_multi_facility,
)
from utils.wrappers import wrap_action_d_plus_a


# For td3 hyperparameters tuning
def uniform_multi_facility_dplusa_env_factory():
    env = make_beer_game_uniform_multi_facility(
        agent_managed_facilities=[
            "retailer",
            "wholesaler",
            "distributor",
            "manufacturer",
        ],
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


def uniform_env_factory(
    role: List[str],
    discrete: bool,
    global_observable: bool,
    multi_discrete_action_space=False,
    state_version="v0",
):
    def func():
        # TODO
        env = make_beer_game_uniform_multi_facility(
            agent_managed_facilities=role,
            box_action_space=not discrete,
            multi_discrete_action_space=multi_discrete_action_space,
            random_init=True,
            global_observable=global_observable,
            state_version=state_version,
        )
        return env

    return func


def dunnhumby_env_factory(
    role: List[str],
    discrete: bool,
    global_observable: bool,
    multi_discrete_action_space=False,
    state_version="v0",
):
    def func():
        # TODO
        env = make_beer_game_dunnhumby_multi_facility(
            agent_managed_facilities=role,
            box_action_space=not discrete,
            multi_discrete_action_space=multi_discrete_action_space,
            random_init=True,
            global_observable=global_observable,
            state_version=state_version,
        )
        return env

    return func


def normal_env_factory(
    role: List[str],
    discrete: bool,
    global_observable: bool,
    info_leadtime: List[int],
    shipment_leadtime: List[int],
    multi_discrete_action_space=False,
    cost_type="general",
    state_version="v0",
    target_levels=None,
):
    def func():
        env = make_beer_game_normal_multi_facility(
            agent_managed_facilities=role,
            box_action_space=not discrete,
            multi_discrete_action_space=multi_discrete_action_space,
            random_init=True,
            global_observable=global_observable,
            cost_type=cost_type,
            state_version=state_version,
            info_leadtime=info_leadtime,
            shipment_leadtime=shipment_leadtime,
            target_levels=target_levels,
        )
        return env

    return func


def normal_multi_facility_dplusa_env_factory():
    env = make_beer_game_normal_multi_facility(
        agent_managed_facilities=[
            "retailer",
            "wholesaler",
            "distributor",
            "manufacturer",
        ],
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
            role=["retailer", "wholesaler", "distributor", "manufacturer"],
            discrete=False,
            global_observable=True,
        ),
        max_episode_steps=100,
    )

    gym.envs.register(
        id="BeerGameUniformMultiFacilityFullInfo-v1",
        entry_point=uniform_env_factory(
            role=["retailer", "wholesaler", "distributor", "manufacturer"],
            discrete=False,
            global_observable=True,
            state_version="v1",
        ),
        max_episode_steps=100,
    )

    gym.envs.register(
        id="BeerGameUniformMultiFacilityFullInfoDiscrete-v1",
        entry_point=uniform_env_factory(
            role=["retailer", "wholesaler", "distributor", "manufacturer"],
            discrete=True,
            multi_discrete_action_space=True,
            global_observable=True,
            state_version="v1",
        ),
        max_episode_steps=100,
    )

    gym.envs.register(
        id="BeerGameNormalMultiFacilityFullInfo-v0",
        entry_point=normal_env_factory(
            role=["retailer", "wholesaler", "distributor", "manufacturer"],
            discrete=False,
            global_observable=True,
            info_leadtime=[2, 2, 2, 1],
            shipment_leadtime=[2, 2, 2, 2],
        ),
        max_episode_steps=100,
    )

    gym.envs.register(
        id="BeerGameNormalMultiFacilityFullInfo-v1",
        entry_point=normal_env_factory(
            role=["retailer", "wholesaler", "distributor", "manufacturer"],
            discrete=False,
            global_observable=True,
            state_version="v1",
            info_leadtime=[2, 2, 2, 1],
            shipment_leadtime=[2, 2, 2, 2],
        ),
        max_episode_steps=100,
    )

    gym.envs.register(
        id="BeerGameCSCostNormalMultiFacilityFullInfo-v1",
        entry_point=normal_env_factory(
            role=["retailer", "wholesaler", "distributor", "manufacturer"],
            discrete=False,
            global_observable=True,
            state_version="v1",
            cost_type="clark-scarf",
            info_leadtime=[0, 0, 0, 0],
            shipment_leadtime=[4, 4, 4, 3],
        ),
        max_episode_steps=100,
    )

    gym.envs.register(
        id="BeerGameNormalMultiFacilityFullInfoDiscrete-v1",
        entry_point=normal_env_factory(
            role=["retailer", "wholesaler", "distributor", "manufacturer"],
            discrete=True,
            multi_discrete_action_space=True,
            global_observable=True,
            state_version="v1",
            info_leadtime=[2, 2, 2, 1],
            shipment_leadtime=[2, 2, 2, 2],
        ),
        max_episode_steps=100,
    )

    gym.envs.register(
        id="BeerGameCSCostNormalMultiFacilityFullInfoDiscrete-v1",
        entry_point=normal_env_factory(
            role=["retailer", "wholesaler", "distributor", "manufacturer"],
            discrete=True,
            multi_discrete_action_space=True,
            global_observable=True,
            state_version="v1",
            cost_type="clark-scarf",
            info_leadtime=[0, 0, 0, 0],
            shipment_leadtime=[4, 4, 4, 3],
        ),
        max_episode_steps=100,
    )

    gym.envs.register(
        id="BeerGameDunnhumbyMultiFacilityFullInfo-v0",
        entry_point=dunnhumby_env_factory(
            role=["retailer", "wholesaler", "distributor", "manufacturer"],
            discrete=False,
            global_observable=True,
            state_version="v0",
        ),
        max_episode_steps=100,
    )

    gym.envs.register(
        id="BeerGameDunnhumbyMultiFacilityFullInfo-v1",
        entry_point=dunnhumby_env_factory(
            role=["retailer", "wholesaler", "distributor", "manufacturer"],
            discrete=False,
            global_observable=True,
            state_version="v1",
        ),
        max_episode_steps=100,
    )

    gym.envs.register(
        id="BeerGameDunnhumbyMultiFacilityFullInfoDiscrete-v1",
        entry_point=dunnhumby_env_factory(
            role=["retailer", "wholesaler", "distributor", "manufacturer"],
            discrete=True,
            multi_discrete_action_space=True,
            global_observable=True,
            state_version="v1",
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
            entry_point=uniform_env_factory(
                role=role, discrete=False, global_observable=False
            ),
            max_episode_steps=100,
        )

        gym.envs.register(
            id=f"BeerGameUniform{key}-v1",
            entry_point=uniform_env_factory(
                role=role, discrete=False, global_observable=False, state_version="v1"
            ),
            max_episode_steps=100,
        )

        gym.envs.register(
            id=f"BeerGameUniform{key}Discrete-v0",
            entry_point=uniform_env_factory(
                role=role, discrete=True, global_observable=False
            ),
            max_episode_steps=100,
        )
        gym.envs.register(
            id=f"BeerGameUniform{key}Discrete-v1",
            entry_point=uniform_env_factory(
                role=role, discrete=True, global_observable=False, state_version="v1"
            ),
            max_episode_steps=100,
        )

        gym.envs.register(
            id=f"BeerGameUniform{key}FullInfo-v0",
            entry_point=uniform_env_factory(
                role=role, discrete=False, global_observable=True
            ),
            max_episode_steps=100,
        )
        gym.envs.register(
            id=f"BeerGameUniform{key}FullInfo-v1",
            entry_point=uniform_env_factory(
                role=role,
                discrete=False,
                global_observable=True,
                state_version="v1",
            ),
            max_episode_steps=100,
        )

        gym.envs.register(
            id=f"BeerGameUniform{key}FullInfoDiscrete-v0",
            entry_point=uniform_env_factory(
                role=role, discrete=True, global_observable=True
            ),
            max_episode_steps=100,
        )

        gym.envs.register(
            id=f"BeerGameUniform{key}FullInfoDiscrete-v1",
            entry_point=uniform_env_factory(
                role=role,
                discrete=True,
                global_observable=True,
                state_version="v1",
            ),
            max_episode_steps=100,
        )

        # normal (basic scenario)
        gym.envs.register(
            id=f"BeerGameNormal{key}-v0",
            entry_point=normal_env_factory(
                role=role,
                discrete=False,
                global_observable=False,
                info_leadtime=[2, 2, 2, 1],
                shipment_leadtime=[2, 2, 2, 2],
            ),
            max_episode_steps=100,
        )

        gym.envs.register(
            id=f"BeerGameNormal{key}-v1",
            entry_point=normal_env_factory(
                role=role,
                discrete=False,
                global_observable=False,
                info_leadtime=[2, 2, 2, 1],
                shipment_leadtime=[2, 2, 2, 2],
                state_version="v1",
            ),
            max_episode_steps=100,
        )

        gym.envs.register(
            id=f"BeerGameNormal{key}Discrete-v0",
            entry_point=normal_env_factory(
                role=role,
                discrete=True,
                global_observable=False,
                info_leadtime=[2, 2, 2, 1],
                shipment_leadtime=[2, 2, 2, 2],
            ),
            max_episode_steps=100,
        )

        gym.envs.register(
            id=f"BeerGameNormal{key}Discrete-v1",
            entry_point=normal_env_factory(
                role=role,
                discrete=True,
                global_observable=False,
                info_leadtime=[2, 2, 2, 1],
                shipment_leadtime=[2, 2, 2, 2],
                state_version="v1",
            ),
            max_episode_steps=100,
        )

        gym.envs.register(
            id=f"BeerGameNormal{key}FullInfo-v0",
            entry_point=normal_env_factory(
                role=role,
                discrete=False,
                global_observable=True,
                info_leadtime=[2, 2, 2, 1],
                shipment_leadtime=[2, 2, 2, 2],
            ),
            max_episode_steps=100,
        )

        gym.envs.register(
            id=f"BeerGameNormal{key}FullInfo-v1",
            entry_point=normal_env_factory(
                role=role,
                discrete=False,
                global_observable=True,
                state_version="v1",
                info_leadtime=[2, 2, 2, 1],
                shipment_leadtime=[2, 2, 2, 2],
            ),
            max_episode_steps=100,
        )

        gym.envs.register(
            id=f"BeerGameNormal{key}FullInfoDiscrete-v0",
            entry_point=normal_env_factory(
                role=role,
                discrete=True,
                global_observable=True,
                info_leadtime=[2, 2, 2, 1],
                shipment_leadtime=[2, 2, 2, 2],
            ),
            max_episode_steps=100,
        )

        gym.envs.register(
            id=f"BeerGameNormal{key}FullInfoDiscrete-v1",
            entry_point=normal_env_factory(
                role=role,
                discrete=True,
                global_observable=True,
                info_leadtime=[2, 2, 2, 1],
                shipment_leadtime=[2, 2, 2, 2],
                state_version="v1",
            ),
            max_episode_steps=100,
        )

        gym.envs.register(
            id=f"BeerGameCSCostNormal{key}Discrete-v0",
            entry_point=normal_env_factory(
                role=role,
                discrete=True,
                global_observable=False,
                cost_type="clark-scarf",
                info_leadtime=[0, 0, 0, 0],
                shipment_leadtime=[4, 4, 4, 3],
            ),
            max_episode_steps=100,
        )

        gym.envs.register(
            id=f"BeerGameCSCostNormal{key}Discrete-v1",
            entry_point=normal_env_factory(
                role=role,
                discrete=True,
                global_observable=False,
                cost_type="clark-scarf",
                info_leadtime=[0, 0, 0, 0],
                shipment_leadtime=[4, 4, 4, 3],
                state_version="v1",
            ),
            max_episode_steps=100,
        )

        gym.envs.register(
            id=f"BeerGameCSCostNormal{key}FullInfoDiscrete-v0",
            entry_point=normal_env_factory(
                role=role,
                discrete=True,
                global_observable=True,
                cost_type="clark-scarf",
                info_leadtime=[0, 0, 0, 0],
                shipment_leadtime=[4, 4, 4, 3],
            ),
            max_episode_steps=100,
        )

        gym.envs.register(
            id=f"BeerGameCSCostNormal{key}FullInfoDiscrete-v1",
            entry_point=normal_env_factory(
                role=role,
                discrete=True,
                global_observable=True,
                cost_type="clark-scarf",
                info_leadtime=[0, 0, 0, 0],
                shipment_leadtime=[4, 4, 4, 3],
                state_version="v1",
            ),
            max_episode_steps=100,
        )

        gym.envs.register(
            id=f"BeerGameCSCostNormal{key}-v0",
            entry_point=normal_env_factory(
                role=role,
                discrete=False,
                global_observable=False,
                cost_type="clark-scarf",
                info_leadtime=[0, 0, 0, 0],
                shipment_leadtime=[4, 4, 4, 3],
            ),
            max_episode_steps=100,
        )

        gym.envs.register(
            id=f"BeerGameCSCostNormal{key}-v1",
            entry_point=normal_env_factory(
                role=role,
                discrete=False,
                info_leadtime=[0, 0, 0, 0],
                global_observable=False,
                cost_type="clark-scarf",
                shipment_leadtime=[4, 4, 4, 3],
                state_version="v1",
            ),
            max_episode_steps=100,
        )

        gym.envs.register(
            id=f"BeerGameCSCostNormal{key}FullInfo-v0",
            entry_point=normal_env_factory(
                role=role,
                discrete=False,
                info_leadtime=[0, 0, 0, 0],
                global_observable=True,
                cost_type="clark-scarf",
                shipment_leadtime=[4, 4, 4, 3],
            ),
            max_episode_steps=100,
        )

        gym.envs.register(
            id=f"BeerGameCSCostNormal{key}FullInfo-v1",
            entry_point=normal_env_factory(
                role=role,
                discrete=False,
                info_leadtime=[0, 0, 0, 0],
                global_observable=True,
                cost_type="clark-scarf",
                shipment_leadtime=[4, 4, 4, 3],
                state_version="v1",
            ),
            max_episode_steps=100,
        )

        # Dunnhumby
        gym.envs.register(
            id=f"BeerGameDunnhumby{key}-v1",
            entry_point=dunnhumby_env_factory(
                role=role, discrete=False, global_observable=False, state_version="v1"
            ),
            max_episode_steps=100,
        )

        gym.envs.register(
            id=f"BeerGameDunnhumby{key}Discrete-v1",
            entry_point=dunnhumby_env_factory(
                role=role, discrete=True, global_observable=False, state_version="v1"
            ),
            max_episode_steps=100,
        )

        gym.envs.register(
            id=f"BeerGameDunnhumby{key}FullInfo-v1",
            entry_point=dunnhumby_env_factory(
                role=role,
                discrete=False,
                global_observable=True,
                state_version="v1",
            ),
            max_episode_steps=100,
        )

        gym.envs.register(
            id=f"BeerGameDunnhumby{key}FullInfoDiscrete-v1",
            entry_point=dunnhumby_env_factory(
                role=role,
                discrete=True,
                global_observable=True,
                state_version="v1",
            ),
            max_episode_steps=100,
        )
