import os
import sys
sys.path.insert(0, os.path.abspath("../"))

import unittest
import numpy as np
from supplynetwork import SupplyNetwork
from network_components import Node, Arc
from utils.demands import Demand
from envs import (
    InventoryManagementEnvMultiPlayer,
    make_beer_game,
    make_beer_game_normal_multi_facility,
    make_beer_game_uniform_multi_facility,
)
from stable_baselines3.common.env_checker import check_env


def run_beer_game(env: InventoryManagementEnvMultiPlayer, order_quantities):
    env.reset()
    total_r = 0
    for i in range(0, 20):
        states, reward, terminal, _ = env.step(order_quantities[i])
        total_r += reward
    return total_r


class TestBuildBeerGame(unittest.TestCase):
    def test_classic_beer_game_retailer(self):
        env = make_beer_game(agent_managed_facilities=["retailer"])
        test_quantities_1 = np.array([0] * 20).reshape((20, 1))
        test_quantities_2 = np.array([4] * 20).reshape((20, 1))
        test_quantities_3 = np.array([12] * 20).reshape((20, 1))

        self.assertEqual(run_beer_game(env, test_quantities_1), -1656.0)
        self.assertEqual(run_beer_game(env, test_quantities_2), -798.0)
        self.assertEqual(run_beer_game(env, test_quantities_3), -1260.0)

    def test_classic_beer_game_manufacturer(self):
        env = make_beer_game(agent_managed_facilities=["manufacturer"])
        test_quantities_1 = np.array([0, 0] + [8] * 18).reshape((20, 1))
        test_quantities_2 = np.array([4] * 20).reshape((20, 1))
        test_quantities_3 = np.array([12] * 20).reshape((20, 1))

        self.assertEqual(run_beer_game(env, test_quantities_1), -224)
        self.assertEqual(run_beer_game(env, test_quantities_2), -644)
        self.assertEqual(run_beer_game(env, test_quantities_3), -658)

    def test_classic_beer_game_distributor(self):
        env = make_beer_game(agent_managed_facilities=["distributor"])
        test_quantities_1 = np.array([0, 0] + [8] * 18).reshape((20, 1))
        test_quantities_2 = np.array([4] * 20).reshape((20, 1))
        test_quantities_3 = np.array([12] * 20).reshape((20, 1))

        self.assertEqual(run_beer_game(env, test_quantities_1), -170)
        self.assertEqual(run_beer_game(env, test_quantities_2), -784)
        self.assertEqual(run_beer_game(env, test_quantities_3), -644)

    def test_classic_beer_game_wholesaler(self):
        env = make_beer_game(agent_managed_facilities=["wholesaler"])
        test_quantities_1 = np.array([0, 0] + [8] * 18).reshape((20, 1))
        test_quantities_2 = np.array([4] * 20).reshape((20, 1))
        test_quantities_3 = np.array([12] * 20).reshape((20, 1))

        self.assertEqual(run_beer_game(env, test_quantities_1), -158.0)
        self.assertEqual(run_beer_game(env, test_quantities_2), -864.0)
        self.assertEqual(run_beer_game(env, test_quantities_3), -888.0)

    def test_classic_beer_game_retailer_wholesaler(self):
        env = make_beer_game(agent_managed_facilities=["retailer", "wholesaler"])
        test_quantities_1 = np.array([0, 8] + [0, 4] + [0, 0] * 18).reshape(20, 2)

        self.assertEqual(run_beer_game(env, test_quantities_1), -1656.0)

    def test_classic_beer_game_gym_compatibility(self):
        check_env(make_beer_game(agent_managed_facilities=["retailer"]))
        check_env(make_beer_game(agent_managed_facilities=["wholesaler"]))
        check_env(make_beer_game(agent_managed_facilities=["retailer", "wholesaler"]))
        check_env(
            make_beer_game(
                agent_managed_facilities=[
                    "retailer",
                    "wholesaler",
                    "distributor",
                    "manufacturer",
                ]
            )
        )

    def test_uniform_beer_game_gym_compatibility(self):
        check_env(
            make_beer_game_uniform_multi_facility(
                agent_managed_facilities=["retailer"], box_action_space=True
            )
        )
        check_env(
            make_beer_game_uniform_multi_facility(
                agent_managed_facilities=["wholesaler"], box_action_space=True
            )
        )
        check_env(
            make_beer_game_uniform_multi_facility(
                agent_managed_facilities=["retailer", "wholesaler"],
                box_action_space=True,
            )
        )
        check_env(
            make_beer_game_uniform_multi_facility(
                agent_managed_facilities=[
                    "retailer",
                    "wholesaler",
                    "distributor",
                    "manufacturer",
                ],
                box_action_space=True,
            )
        )

    def test_normal_beer_game_gym_compatibility(self):
        check_env(
            make_beer_game_normal_multi_facility(
                agent_managed_facilities=["retailer"], box_action_space=True
            )
        )
        check_env(
            make_beer_game_normal_multi_facility(
                agent_managed_facilities=["wholesaler"], box_action_space=True
            )
        )
        check_env(
            make_beer_game_normal_multi_facility(
                agent_managed_facilities=["retailer", "wholesaler"],
                box_action_space=True,
            )
        )
        check_env(
            make_beer_game_normal_multi_facility(
                agent_managed_facilities=[
                    "retailer",
                    "wholesaler",
                    "distributor",
                    "manufacturer",
                ],
                box_action_space=True,
            )
        )


class TestSupplyNetwork(unittest.TestCase):
    def test_get_customers(self):
        env = make_beer_game_normal_multi_facility(
            agent_managed_facilities=["retailer"]
        )

        self.assertEqual(env.sn.get_customer_names("retailer"), [])
        self.assertEqual(env.sn.get_customer_names("wholesaler"), ["retailer"])
        self.assertEqual(env.sn.get_customer_names("distributor"), ["wholesaler"])
        self.assertEqual(env.sn.get_customer_names("manufacturer"), ["distributor"])


if __name__ == "__main__":
    unittest.main()
