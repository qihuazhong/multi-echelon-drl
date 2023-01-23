from unittest import TestCase
import os
import sys
from envs import build_beer_game_uniform_aec
from utils.heuristics import BaseStockPolicy

sys.path.insert(0, os.path.abspath('../'))


class Test(TestCase):

    def setUp(self) -> None:
        super().setUp()


    # def test_build_beer_game_uniform_aec(self):
    #     bg_env = build_beer_game_uniform_aec(agent_managed_facilities=['retailer'], return_dict=True, max_episode_steps=100)
    #
    #     array_index = {'on_hand': 0, 'unreceived_pipeline': [3, 4, 5, 6], 'unfilled_demand': 1}
    #
    #     bs_14 = BaseStockPolicy(target_levels=[14],
    #                             array_index=array_index,
    #                             state_dim_per_facility=7)
    #
    #     predefined_policies = {'distributor': bs_14,
    #                            'wholesaler': bs_14,
    #                            'manufacturer': bs_14}
    #
    #     bg_env.reset()
    #     total_r = 0
    #
    #     for agent in bg_env.agent_iter():
    #
    #         observation, reward, done, info = bg_env.last()
    #         # print(observation, reward, done, info)
    #         # print(agent, bg_env.period, bg_env.rewards)
    #         if agent == 'retailer':
    #             action = 4
    #         else:
    #             action = predefined_policies[agent].get_order_quantity({agent: observation})
    #         bg_env.step(action)
    #
    #         total_r += reward
