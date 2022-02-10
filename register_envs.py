# A script to

import gym

from envs import make_beer_game_uniform_multi_facility, make_beer_game_normal_multi_facility
from utils.wrappers import wrap_action_d_plus_a


def uniform_multi_facility_env_factory():
    env = make_beer_game_uniform_multi_facility(agent_managed_facilities=[
        'retailer', 'wholesaler', 'distributor', 'manufacturer'], box_action_space=True, random_init=True)
    return env


def uniform_multi_facility_dplusa_env_factory():
    env = make_beer_game_uniform_multi_facility(agent_managed_facilities=[
        'retailer', 'wholesaler', 'distributor', 'manufacturer'], box_action_space=True, random_init=True)
    env = wrap_action_d_plus_a(env, offset=-8, lb=0, ub=16)
    return env


def uniform_single_facility_retailer_env_factory():
    env = make_beer_game_uniform_multi_facility(agent_managed_facilities=[
        'retailer'], box_action_space=True, random_init=True)
    return env


def uniform_single_facility_wholesaler_env_factory():
    env = make_beer_game_uniform_multi_facility(agent_managed_facilities=[
        'wholesaler'], box_action_space=True, random_init=True)
    #     env = wrap_action_d_plus_a(env, offset=-8)
    return env


def uniform_single_facility_distributor_env_factory():
    env = make_beer_game_uniform_multi_facility(agent_managed_facilities=[
        'distributor'], box_action_space=True, random_init=True)
    #     env = wrap_action_d_plus_a(env, offset=-8)
    return env


def uniform_single_facility_manufacturer_env_factory():
    env = make_beer_game_uniform_multi_facility(agent_managed_facilities=[
        'manufacturer'], box_action_space=True, random_init=True)
    #     env = wrap_action_d_plus_a(env, offset=-8)
    return env


def uniform_single_facility_retailer_discrete_env_factory():
    env = make_beer_game_uniform_multi_facility(agent_managed_facilities=[
        'retailer'], box_action_space=False, random_init=True)
    return env


def uniform_single_facility_wholesaler_discrete_env_factory():
    env = make_beer_game_uniform_multi_facility(agent_managed_facilities=[
        'wholesaler'], box_action_space=False, random_init=True)
    return env


def uniform_single_facility_distributor_discrete_env_factory():
    env = make_beer_game_uniform_multi_facility(agent_managed_facilities=[
        'distributor'], box_action_space=False, random_init=True)
    return env


def uniform_single_facility_manufacturer_discrete_env_factory():
    env = make_beer_game_uniform_multi_facility(agent_managed_facilities=[
        'manufacturer'], box_action_space=False, random_init=True)
    return env


def normal_multi_facility_env_factory():
    env = make_beer_game_normal_multi_facility(agent_managed_facilities=[
        'retailer', 'wholesaler', 'distributor', 'manufacturer'], box_action_space=True, random_init=True)
    #     env = wrap_action_d_plus_a(env, offset=-8)
    return env


def normal_multi_facility_dplusa_env_factory():
    env = make_beer_game_normal_multi_facility(agent_managed_facilities=[
        'retailer', 'wholesaler', 'distributor', 'manufacturer'], box_action_space=True, random_init=True)
    env = wrap_action_d_plus_a(env, offset=-10, lb=0, ub=20)
    return env


def normal_single_facility_retailer_env_factory():
    env = make_beer_game_normal_multi_facility(agent_managed_facilities=[
        'retailer'], box_action_space=True, random_init=True)
    #     env = wrap_action_d_plus_a(env, offset=-8)
    return env


def normal_single_facility_wholesaler_env_factory():
    env = make_beer_game_normal_multi_facility(agent_managed_facilities=[
        'wholesaler'], box_action_space=True, random_init=True)
    #     env = wrap_action_d_plus_a(env, offset=-8)
    return env


def normal_single_facility_distributor_env_factory():
    env = make_beer_game_normal_multi_facility(agent_managed_facilities=[
        'distributor'], box_action_space=True, random_init=True)
    #     env = wrap_action_d_plus_a(env, offset=-8)
    return env


def normal_single_facility_manufacturer_env_factory():
    env = make_beer_game_normal_multi_facility(agent_managed_facilities=[
        'manufacturer'], box_action_space=True, random_init=True)
    #     env = wrap_action_d_plus_a(env, offset=-8)
    return env


def normal_single_facility_retailer_discrete_env_factory():
    env = make_beer_game_normal_multi_facility(agent_managed_facilities=[
        'retailer'], box_action_space=False, random_init=True)
    #     env = wrap_action_d_plus_a(env, offset=-8)
    return env


def normal_single_facility_wholesaler_discrete_env_factory():
    env = make_beer_game_normal_multi_facility(agent_managed_facilities=[
        'wholesaler'], box_action_space=False, random_init=True)
    #     env = wrap_action_d_plus_a(env, offset=-8)
    return env


def normal_single_facility_distributor_discrete_env_factory():
    env = make_beer_game_normal_multi_facility(agent_managed_facilities=[
        'distributor'], box_action_space=False, random_init=True)
    #     env = wrap_action_d_plus_a(env, offset=-8)
    return env


def normal_single_facility_manufacturer_discrete_env_factory():
    env = make_beer_game_normal_multi_facility(agent_managed_facilities=[
        'manufacturer'], box_action_space=False, random_init=True)
    #     env = wrap_action_d_plus_a(env, offset=-8)
    return env


def register_envs() -> None:
    """
    Register different versions of the beer game to the Gym Registry

    """
    gym.envs.register(id='BeerGameUniformMultiFacility-v0',
                      entry_point=uniform_multi_facility_env_factory, max_episode_steps=100)
    gym.envs.register(id='BeerGameUniformMultiFacilityDPlusA-v0',
                      entry_point=uniform_multi_facility_dplusa_env_factory, max_episode_steps=100)

    gym.envs.register(id='BeerGameUniformRetailer-v0',
                      entry_point=uniform_single_facility_retailer_env_factory, max_episode_steps=100)
    gym.envs.register(id='BeerGameUniformWholesaler-v0',
                      entry_point=uniform_single_facility_wholesaler_env_factory, max_episode_steps=100)
    gym.envs.register(id='BeerGameUniformDistributor-v0',
                      entry_point=uniform_single_facility_distributor_env_factory, max_episode_steps=100)
    gym.envs.register(id='BeerGameUniformManufacturer-v0',
                      entry_point=uniform_single_facility_manufacturer_env_factory, max_episode_steps=100)

    gym.envs.register(id='BeerGameUniformRetailerDiscrete-v0',
                      entry_point=uniform_single_facility_retailer_discrete_env_factory, max_episode_steps=100)
    gym.envs.register(id='BeerGameUniformWholesalerDiscrete-v0',
                      entry_point=uniform_single_facility_wholesaler_discrete_env_factory, max_episode_steps=100)
    gym.envs.register(id='BeerGameUniformDistributorDiscrete-v0',
                      entry_point=uniform_single_facility_distributor_discrete_env_factory, max_episode_steps=100)
    gym.envs.register(id='BeerGameUniformManufacturerDiscrete-v0',
                      entry_point=uniform_single_facility_manufacturer_discrete_env_factory, max_episode_steps=100)


    gym.envs.register(id='BeerGameNormalMultiFacility-v0',
                      entry_point=normal_multi_facility_env_factory, max_episode_steps=100)
    gym.envs.register(id='BeerGameNormalMultiFacilityDPlusA-v0',
                      entry_point=normal_multi_facility_dplusa_env_factory, max_episode_steps=100)

    gym.envs.register(id='BeerGameNormalRetailer-v0',
                      entry_point=normal_single_facility_retailer_env_factory, max_episode_steps=100)
    gym.envs.register(id='BeerGameNormalWholesaler-v0',
                      entry_point=normal_single_facility_wholesaler_env_factory, max_episode_steps=100)
    gym.envs.register(id='BeerGameNormalDistributor-v0',
                      entry_point=normal_single_facility_distributor_env_factory, max_episode_steps=100)
    gym.envs.register(id='BeerGameNormalManufacturer-v0',
                      entry_point=normal_single_facility_manufacturer_env_factory, max_episode_steps=100)

    gym.envs.register(id='BeerGameNormalRetailerDiscrete-v0',
                      entry_point=normal_single_facility_retailer_discrete_env_factory, max_episode_steps=100)
    gym.envs.register(id='BeerGameNormalWholesalerDiscrete-v0',
                      entry_point=normal_single_facility_wholesaler_discrete_env_factory, max_episode_steps=100)
    gym.envs.register(id='BeerGameNormalDistributorDiscrete-v0',
                      entry_point=normal_single_facility_distributor_discrete_env_factory, max_episode_steps=100)
    gym.envs.register(id='BeerGameNormalManufacturerDiscrete-v0',
                      entry_point=normal_single_facility_manufacturer_discrete_env_factory, max_episode_steps=100)