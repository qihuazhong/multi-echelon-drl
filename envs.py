from abc import ABC
from typing import Union, Tuple, List, Any, Dict
import numpy as np
from numpy import ndarray

from utils.demands import Demand
from utils.heuristics import BaseStockPolicy
from supplynetwork import SupplyNetwork
from network_components import Node, Arc
import gym
from gym.utils import seeding


class InventoryManagementEnvMultiFacility(ABC, gym.Env):

    def __init__(self, supply_network, max_episode_steps: int, action_space: gym.Space,
                 observation_space: gym.Space, return_dict=False):
        """
        Args:
            supply_network:
            return_dict: whether the return states is a numpy array(Default) or a dictionary
        """
        self.sn: SupplyNetwork = supply_network
        self.max_episode_steps = max_episode_steps
        self.action_space = action_space
        self.observation_space = observation_space
        self.return_dict = return_dict
        self.period = 0
        self.terminal = False

        self.seed()

    def reset(self):
        self.terminal = False
        self.sn.reset()
        self.period = 0

        self.sn.before_action(self.period)

        states: Union[np.ndarray, Dict[str, dict]] = {agent: self.sn.get_state(agent)
                                                      for agent in self.sn.agent_managed_facilities}

        if not self.return_dict:
            states = np.array(
                [list(state.values()) for agent, state in states.items()]
            ).flatten()

        return states

    def step(self, quantity) -> Tuple[Union[ndarray, List[dict]], float, bool, dict]:
        """
        return:
            a tuple of state (dict), cost (float), terminal (bool) and info (dict)
        """

        if self.terminal:
            raise ValueError("Cannot take action when the state is terminal.")

        self.sn.agent_action(self.period, quantity)
        self.sn.after_action(self.period)

        cost = self.sn.get_cost()

        self.period += 1

        if self.period >= self.max_episode_steps:
            self.terminal = True
        else:
            self.sn.before_action(self.period)

        states = {agent: self.sn.get_state(agent) for agent in self.sn.agent_managed_facilities}

        if not self.return_dict:
            states = np.array(
                [list(state.values()) for agent, state in states.items()]
            ).flatten()

        return states, cost, self.terminal, {}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


def make_beer_game_basic(agent_managed_facilities=None, max_episode_steps=100, return_dict=False,
                         random_init=False, box_action_space=False):
    if agent_managed_facilities is None:
        agent_managed_facilities = ['retailer']

    if len(agent_managed_facilities) > 1 and not box_action_space:
        raise ValueError('length of agent_managed_facilities >= 1, only box_action_space is allowed. Please specify'
                         'box_action_space=True')

    demand_generator = Demand('normal', mean=10, sd=2, size=max_episode_steps)

    array_index = {'on_hand': 0, 'unreceived_pipeline': [3, 4, 5, 6], 'unfilled_demand': 1, 'latest_demand': 2}

    bsp_48 = BaseStockPolicy(target_levels=[48],
                             array_index=array_index,
                             state_dim_per_facility=7)
    bsp_43 = BaseStockPolicy(target_levels=[43],
                             array_index=array_index,
                             state_dim_per_facility=7)
    bsp_41 = BaseStockPolicy(target_levels=[41],
                             array_index=array_index,
                             state_dim_per_facility=7)
    bsp_30 = BaseStockPolicy(target_levels=[30],
                             array_index=array_index,
                             state_dim_per_facility=7)

    if random_init:
        init_inventory = [0, 21]
        init_shipments = [[0, 21]] * 4
        init_sales_orders = [[0, 21]] * 4
    else:
        init_inventory = 10
        init_shipments = [[10, 0]] * 4
        init_sales_orders = [[10, 0]] * 4

    retailer = Node(name='retailer', initial_inventory=init_inventory, holding_cost=1.0, backorder_cost=10,
                    policy=bsp_48,
                    is_demand_source=True, demands=demand_generator)
    wholesaler = Node(name='wholesaler', initial_inventory=init_inventory, holding_cost=0.75, backorder_cost=0,
                      policy=bsp_43)
    distributor = Node(name='distributor', initial_inventory=init_inventory, holding_cost=0.5, backorder_cost=0,
                       policy=bsp_41)
    manufacturer = Node(name='manufacturer', initial_inventory=init_inventory, holding_cost=0.25, backorder_cost=0,
                        policy=bsp_30)
    supply_source = Node(name='external_supplier', is_external_supplier=True)
    nodes = [retailer, wholesaler, distributor, manufacturer, supply_source]

    arcs = [Arc('external_supplier', 'manufacturer', 1, 2,
                initial_shipments=init_shipments[0],
                initial_sales_orders=init_sales_orders[0],
                random_init=random_init),
            Arc('manufacturer', 'distributor', 2, 2,
                initial_shipments=init_shipments[1],
                initial_sales_orders=init_sales_orders[1],
                random_init=random_init),
            Arc('distributor', 'wholesaler', 2, 2,
                initial_shipments=init_shipments[2],
                initial_sales_orders=init_sales_orders[2],
                random_init=random_init),
            Arc('wholesaler', 'retailer', 2, 2,
                initial_shipments=init_shipments[3],
                initial_sales_orders=init_sales_orders[3],
                random_init=random_init)]

    num_agent_managed_facilities = len(agent_managed_facilities)
    sn = SupplyNetwork(nodes=nodes, arcs=arcs, agent_managed_facilities=agent_managed_facilities)
    if box_action_space:
        action_space = gym.spaces.Box(0, 20, shape=(num_agent_managed_facilities,))
    else:
        action_space = gym.spaces.Discrete(21)
    observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7 * num_agent_managed_facilities,))

    return InventoryManagementEnvMultiFacility(sn, max_episode_steps=max_episode_steps, action_space=action_space,
                                               observation_space=observation_space, return_dict=return_dict)


def make_beer_game_complex(agent_managed_facilities=None, max_episode_steps=100, return_dict=False,
                           random_init=True, box_action_space=False):
    if agent_managed_facilities is None:
        agent_managed_facilities = ['retailer', 'wholesaler', 'distributor', 'manufacturer']

    if len(agent_managed_facilities) > 1 and not box_action_space:
        raise ValueError('length of agent_managed_facilities >= 1, only box_action_space is allowed. Please specify'
                         'box_action_space=True')

    demand_generator = Demand('uniform', low=0, high=8, size=max_episode_steps)

    array_index = {'on_hand': 0, 'unreceived_pipeline': [3, 4, 5, 6], 'unfilled_demand': 1, 'latest_demand': 2}

    bsp_19 = BaseStockPolicy(target_levels=[19],
                             array_index=array_index,
                             state_dim_per_facility=7)

    bsp_20 = BaseStockPolicy(target_levels=[20],
                             array_index=array_index,
                             state_dim_per_facility=7)

    bsp_14 = BaseStockPolicy(target_levels=[14],
                             array_index=array_index,
                             state_dim_per_facility=7)

    if random_init:
        init_inventory = [0, 25]
        init_shipments = [[0, 9]] * 4
        init_sales_orders = [[0, 9]] * 4
    else:
        init_inventory = 12
        init_shipments = [[4, 4]] * 4
        init_sales_orders = [[4, 4]] * 4

    retailer = Node(name='retailer', initial_inventory=init_inventory, holding_cost=0.5, backorder_cost=1,
                    policy=bsp_19,
                    is_demand_source=True, demands=demand_generator)
    wholesaler = Node(name='wholesaler', initial_inventory=init_inventory, holding_cost=0.5, backorder_cost=1,
                      policy=bsp_20)
    distributor = Node(name='distributor', initial_inventory=init_inventory, holding_cost=0.5, backorder_cost=1,
                       policy=bsp_20)
    manufacturer = Node(name='manufacturer', initial_inventory=init_inventory, holding_cost=0.5, backorder_cost=1,
                        policy=bsp_14)
    supply_source = Node(name='external_supplier', is_external_supplier=True)
    nodes = [retailer, wholesaler, distributor, manufacturer, supply_source]

    arcs = [Arc('external_supplier', 'manufacturer', 1, 2,
                initial_shipments=init_shipments[0],
                initial_sales_orders=init_sales_orders[0],
                random_init=random_init),
            Arc('manufacturer', 'distributor', 2, 2,
                initial_shipments=init_shipments[1],
                initial_sales_orders=init_sales_orders[1],
                random_init=random_init),
            Arc('distributor', 'wholesaler', 2, 2,
                initial_shipments=init_shipments[2],
                initial_sales_orders=init_sales_orders[2],
                random_init=random_init),
            Arc('wholesaler', 'retailer', 2, 2,
                initial_shipments=init_shipments[3],
                initial_sales_orders=init_sales_orders[3],
                random_init=random_init)]

    num_agent_managed_facilities = len(agent_managed_facilities)
    sn = SupplyNetwork(nodes=nodes, arcs=arcs, agent_managed_facilities=agent_managed_facilities)

    if box_action_space:
        action_space = gym.spaces.Box(0, 16, shape=(num_agent_managed_facilities,))
    else:
        action_space = gym.spaces.Discrete(17)

    observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7 * num_agent_managed_facilities,))

    return InventoryManagementEnvMultiFacility(sn, max_episode_steps=max_episode_steps, action_space=action_space,
                                               observation_space=observation_space, return_dict=return_dict)
