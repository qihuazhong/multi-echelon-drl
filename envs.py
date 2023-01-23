import logging
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
from gym.spaces import Discrete, Box
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector

# print logging info to stdout
logging.basicConfig(level=logging.WARNING, handlers=[logging.StreamHandler()])


class InventoryManagementAECEnv(AECEnv, ABC):
    """An Agent-Environment-Cycle environment wrapper

    Attributes:
        supply_network (SupplyNetwork): A SupplyNetwork object.
        possible_agents (List[str]): A list of string names of the available agents'
        agent_name_mapping (dict): a dict mapping agent name to agent index
        agent_selection (str): current agent
    """

    def __init__(self, supply_network: SupplyNetwork, max_episode_steps, return_dict: bool = False):
        super().__init__()
        self.agent_selection = None
        self.supply_network = supply_network
        self.possible_agents = self.supply_network.internal_nodes

        self.max_episode_steps = max_episode_steps
        self.period = 0
        self.done = False
        self.agents = self.possible_agents[:]
        self.dones = {agent: False for agent in self.agents}

        self.rewards = {agent: 0 for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}

        # self.possible_agents = self.supply_network.agent_managed_facilities
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))
        self._agent_selector = agent_selector(self.possible_agents)

        # TODO action space and observation space should be specified per environment
        self.action_spaces = None  # {agent: Discrete(30) for agent in self.possible_agents}
        self.observation_spaces = (
            None  # {agent: Box(low=-np.inf, high=np.inf, shape=(7,)) for agent in self.possible_agents}
        )

        self.return_dict = return_dict

    def reset(self):
        """Reset (initialize) the environment

        Different from a parallel environment. The reset() method of an AEC environment does not return the observation.
        """

        self.agents = self.possible_agents[:]

        self.rewards = {agent: 0 for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        # self.terminal = False
        self.agent_selection = self._agent_selector.next()
        self.period = 0
        self.done = False
        self.dones = {agent: False for agent in self.agents}

        self.supply_network.reset(aec=True)

    def step(self, action: Union[dict, list]):
        """
        Args:
            action : The action can either be a scalar value, a dictionary object keyed by the supplier(s), or a
            list/array of values. When list or array of values are provided, the order of the arcs taking the actions
            may not be consistent.
        """
        current_agent = self.agent_selection

        if self.dones[current_agent]:
            return self._was_done_step(None)

        # Check max period
        if self.period >= self.max_episode_steps:
            raise RuntimeError("The number Maximum period has been reached. No more action is allowed.")
        elif self.period >= self.max_episode_steps - 1:
            self.dones[current_agent] = True

        if isinstance(action, dict):
            for supplier, action_value in action.items():
                arc = self.supply_network.arcs[(supplier, current_agent)]
                self.supply_network.nodes[current_agent].place_order(self.observe(current_agent), arc, action_value)
        elif isinstance(action, (list, np.ndarray, int, float, np.generic)):
            action: np.ndarray = np.asarray(action).reshape(
                -1,
            )
            if action.size > 1:
                logging.warning(
                    f"The action is of type {type(action)}, not a dict. "
                    f"The order of arcs taking the actions may not be consistent."
                )
            elif action.size == 1:
                logging.info(
                    f"A scalar value is provided, the order will be placed against the first supplier."
                    f"This may cause issues when there are multiple supplier"
                )

            for i, action_value in enumerate(action):
                supplier = self.supply_network.suppliers[current_agent][i]
                arc = self.supply_network.arcs[(supplier, current_agent)]
                self.supply_network.nodes[current_agent].place_order(self.observe(current_agent), arc, action_value)
        else:
            raise TypeError(f"Action type {type(action)} not supported")

        if self._agent_selector.is_last():
            self.supply_network.transition()
            self.period += 1

            # Cooperative setting, all agents share the same total reward
            total_reward = sum([self.supply_network.get_node_cost(agent) for agent in self.agents])
            self.rewards = {agent: total_reward for agent in self.agents}

        else:
            self._clear_rewards()

        self.agent_selection = self._agent_selector.next()
        self._cumulative_rewards[current_agent] = 0
        self._accumulate_rewards()

        if self.agent_selection not in self.supply_network.agent_managed_facilities:
            # Use predefined policy
            pass
            # print(self.last())  # TODO

    def observe(self, agent):

        states = self.supply_network.get_state(agent)

        if not self.return_dict:
            states = np.array(list(states.values()), dtype=np.float32)
        return states

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    # def state(self):
    #     pass


class InventoryManagementEnv(gym.Env, ABC):
    def __init__(self, supply_chain_network, max_episode_steps: int, visible_states=None, return_dict=False):
        raise NotImplemented

    def reset(self):
        raise NotImplemented

    def step(self, quantity):
        raise NotImplemented


class InventoryManagementEnvMultiPlayer(ABC, gym.Env):
    def __init__(
        self,
        supply_network,
        max_episode_steps: int,
        action_space: gym.Space,
        observation_space: gym.Space,
        visible_states=None,
        return_dict=False,
    ):
        """
        Args:
            supply_network:
            visible_states: A string or a list of strings. Limit the states that are visible to the agent. Return
            all states when not provided.
            return_dict: whether the return states is a numpy array(Default) or a dictionary
        """
        self.sn: SupplyNetwork = supply_network
        self.max_episode_steps = max_episode_steps
        self.action_space = action_space
        self.observation_space = observation_space
        self.visible_states = visible_states
        self.return_dict = return_dict
        self.period = 0
        self.terminal = False

        self.seed()

    def reset(self):
        self.terminal = False
        self.sn.reset()
        self.period = 0

        self.sn.before_action(self.period)

        states: Union[np.ndarray, Dict[str, dict]] = {
            agent: self.sn.get_state(agent) for agent in self.sn.agent_managed_facilities
        }

        if not self.return_dict:
            states = np.array([list(state.values()) for agent, state in states.items()], dtype=np.float32).flatten()

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
            states = np.array([list(state.values()) for agent, state in states.items()], dtype=np.float32).flatten()

        # return states, cost, self.terminal, self.terminal, {}
        return states, cost, self.terminal, {}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


def build_beer_game_uniform_aec(
    agent_managed_facilities=None, max_episode_steps=100, return_dict=False, random_init=False, mode="train", seed=None
) -> InventoryManagementAECEnv:
    if seed:
        np.random.seed(seed)

    if agent_managed_facilities is None:
        agent_managed_facilities = ["retailer", "wholesaler", "distributor", "manufacturer"]

    # TODO: make this an modifiable attribute
    if mode == "test":
        demand_generator = Demand("samples", data_path="data/deepbeerinventory/demandTs0-9.npy", size=max_episode_steps)
    elif mode == "train":
        demand_generator = Demand("uniform", low=0, high=8, size=max_episode_steps)
    else:
        raise ValueError()

    array_index = {"on_hand": 0, "unreceived_pipeline": [3, 4, 5, 6], "unfilled_demand": 1}
    bs_19 = BaseStockPolicy(target_levels=[19], array_index=array_index, state_dim_per_facility=7)

    bs_20 = BaseStockPolicy(target_levels=[20], array_index=array_index, state_dim_per_facility=7)

    bs_14 = BaseStockPolicy(target_levels=[14], array_index=array_index, state_dim_per_facility=7)

    # whether to initialize the state randomly
    if random_init:
        init_inventory = [0, 25]
        init_shipments = [[0, 9]] * 4
        init_sales_orders = [[0, 9]] * 4
    else:
        init_inventory = 12
        init_shipments = [[4, 4]] * 4
        init_sales_orders = [[4, 4]] * 4

    retailer = Node(
        name="retailer",
        initial_inventory=init_inventory,
        holding_cost=0.5,
        backorder_cost=1,
        policy=bs_19,
        is_demand_source=True,
        demands=demand_generator,
    )
    wholesaler = Node(
        name="wholesaler", initial_inventory=init_inventory, holding_cost=0.5, backorder_cost=1, policy=bs_20
    )
    distributor = Node(
        name="distributor", initial_inventory=init_inventory, holding_cost=0.5, backorder_cost=1, policy=bs_20
    )
    manufacturer = Node(
        name="manufacturer", initial_inventory=init_inventory, holding_cost=0.5, backorder_cost=1, policy=bs_14
    )
    supply_source = Node(name="external_supplier", is_external_supplier=True)
    nodes = [retailer, wholesaler, distributor, manufacturer, supply_source]

    arcs = [
        Arc(
            "external_supplier",
            "manufacturer",
            1,
            2,
            initial_shipments=init_shipments[0],
            initial_sales_orders=init_sales_orders[0],
            random_init=random_init,
        ),
        Arc(
            "manufacturer",
            "distributor",
            2,
            2,
            initial_shipments=init_shipments[1],
            initial_sales_orders=init_sales_orders[1],
            random_init=random_init,
        ),
        Arc(
            "distributor",
            "wholesaler",
            2,
            2,
            initial_shipments=init_shipments[2],
            initial_sales_orders=init_sales_orders[2],
            random_init=random_init,
        ),
        Arc(
            "wholesaler",
            "retailer",
            2,
            2,
            initial_shipments=init_shipments[3],
            initial_sales_orders=init_sales_orders[3],
            random_init=random_init,
        ),
    ]

    sn = SupplyNetwork(nodes=nodes, arcs=arcs, agent_managed_facilities=agent_managed_facilities)

    return InventoryManagementAECEnv(sn, return_dict=return_dict, max_episode_steps=max_episode_steps)


def make_beer_game_basic(player="retailer", max_episode_steps=100, return_dict=False, seed=None):
    """
    A basic scenario described in Oroojlooyjadid et al. page 19.
    Args:
        player:
        max_episode_steps:
        return_dict:
        seed:
    Returns:

    """
    if seed:
        np.random.seed(seed)
    # else:
    # make sure the environment is random in parallel processing
    # np.random.seed(int(time.time()))

    demand_generator = Demand("uniform", low=0, high=2, size=max_episode_steps)

    bs_8 = BaseStockPolicy(8)
    bs_0 = BaseStockPolicy(0)

    demand_source = Node(name="is_demand_source", is_demand_source=True, demands=demand_generator)
    retailer = Node(name="retailer", holding_cost=2.0, backorder_cost=2.0, policy=bs_8)
    wholesaler = Node(name="wholesaler", holding_cost=2.0, backorder_cost=0.0, policy=bs_8)
    distributor = Node(name="distributor", holding_cost=2.0, backorder_cost=0.0, policy=bs_0)
    manufacturer = Node(name="manufacturer", holding_cost=2.0, backorder_cost=0.0, policy=bs_0)
    supply_source = Node(name="is_external_supplier", is_external_supplier=True)
    nodes = [demand_source, retailer, wholesaler, distributor, manufacturer, supply_source]

    arcs = [
        Arc("is_external_supplier", "manufacturer", 2, 2, initial_previous_orders=[0, 0, 0, 0]),
        Arc("manufacturer", "distributor", 2, 2, initial_previous_orders=[0, 0, 0, 0]),
        Arc("distributor", "wholesaler", 2, 2, initial_previous_orders=[0, 0, 0, 0]),
        Arc("wholesaler", "retailer", 2, 2, initial_previous_orders=[0, 0, 0, 0]),
        Arc("retailer", "is_demand_source", 0, 0),
    ]

    scn = SupplyNetwork(nodes=nodes, arcs=arcs, agent_managed_facilities=[player])

    return InventoryManagementEnv(scn, return_dict=return_dict, max_episode_steps=max_episode_steps)


def make_beer_game_normal_multi_facility(
    agent_managed_facilities=None,
    max_episode_steps=100,
    return_dict=False,
    random_init=False,
    env_mode="train",
    box_action_space=False,
):
    if agent_managed_facilities is None:
        agent_managed_facilities = ["retailer"]

    if len(agent_managed_facilities) > 1 and not box_action_space:
        raise ValueError(
            "length of agent_managed_facilities >= 1, only box_action_space is allowed. Please specify"
            "box_action_space=True"
        )

    # TODO Make env_mode an attribute of the environment
    if env_mode == "test":
        demand_generator = Demand(
            "samples", data_path="./data/deepbeerinventory/demandTs1-10-2.npy", size=max_episode_steps
        )
    elif env_mode == "train":
        demand_generator = Demand("normal", mean=10, sd=2, size=max_episode_steps)
    else:
        raise ValueError()

    array_index = {"on_hand": 0, "unreceived_pipeline": [3, 4, 5, 6], "unfilled_demand": 1, "latest_demand": 2}

    bsp_48 = BaseStockPolicy(target_levels=[48], array_index=array_index, state_dim_per_facility=7)
    bsp_43 = BaseStockPolicy(target_levels=[43], array_index=array_index, state_dim_per_facility=7)
    bsp_41 = BaseStockPolicy(target_levels=[41], array_index=array_index, state_dim_per_facility=7)
    bsp_30 = BaseStockPolicy(target_levels=[30], array_index=array_index, state_dim_per_facility=7)

    if random_init:
        init_inventory = [0, 21]
        init_shipments = [[0, 21]] * 4
        init_sales_orders = [[0, 21]] * 4
    else:
        init_inventory = 10
        init_shipments = [[10, 0]] * 4
        init_sales_orders = [[10, 0]] * 4

    retailer = Node(
        name="retailer",
        initial_inventory=init_inventory,
        holding_cost=1.0,
        backorder_cost=10,
        policy=bsp_48,
        is_demand_source=True,
        demands=demand_generator,
    )
    wholesaler = Node(
        name="wholesaler", initial_inventory=init_inventory, holding_cost=0.75, backorder_cost=0, policy=bsp_43
    )
    distributor = Node(
        name="distributor", initial_inventory=init_inventory, holding_cost=0.5, backorder_cost=0, policy=bsp_41
    )
    manufacturer = Node(
        name="manufacturer", initial_inventory=init_inventory, holding_cost=0.25, backorder_cost=0, policy=bsp_30
    )
    supply_source = Node(name="external_supplier", is_external_supplier=True)
    nodes = [retailer, wholesaler, distributor, manufacturer, supply_source]

    arcs = [
        Arc(
            "external_supplier",
            "manufacturer",
            1,
            2,
            initial_shipments=init_shipments[0],
            initial_sales_orders=init_sales_orders[0],
            random_init=random_init,
        ),
        Arc(
            "manufacturer",
            "distributor",
            2,
            2,
            initial_shipments=init_shipments[1],
            initial_sales_orders=init_sales_orders[1],
            random_init=random_init,
        ),
        Arc(
            "distributor",
            "wholesaler",
            2,
            2,
            initial_shipments=init_shipments[2],
            initial_sales_orders=init_sales_orders[2],
            random_init=random_init,
        ),
        Arc(
            "wholesaler",
            "retailer",
            2,
            2,
            initial_shipments=init_shipments[3],
            initial_sales_orders=init_sales_orders[3],
            random_init=random_init,
        ),
    ]

    num_agent_managed_facilities = len(agent_managed_facilities)
    sn = SupplyNetwork(nodes=nodes, arcs=arcs, agent_managed_facilities=agent_managed_facilities)
    if box_action_space:
        action_space = gym.spaces.Box(0, 20, shape=(num_agent_managed_facilities,))
    else:
        action_space = gym.spaces.Discrete(21)
    observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7 * num_agent_managed_facilities,))

    return InventoryManagementEnvMultiPlayer(
        sn,
        max_episode_steps=max_episode_steps,
        action_space=action_space,
        observation_space=observation_space,
        return_dict=return_dict,
    )


def make_beer_game_uniform_multi_facility(
    agent_managed_facilities=None,
    max_episode_steps=100,
    return_dict=False,
    random_init=True,
    env_mode="train",
    box_action_space=False,
):
    if agent_managed_facilities is None:
        agent_managed_facilities = ["retailer", "wholesaler", "distributor", "manufacturer"]

    if len(agent_managed_facilities) > 1 and not box_action_space:
        raise ValueError(
            "length of agent_managed_facilities >= 1, only box_action_space is allowed. Please specify"
            "box_action_space=True"
        )

    # TODO Make env_mode an attribute of the environment
    if env_mode == "test":
        demand_generator = Demand(
            "samples", data_path="./data/deepbeerinventory/demandTs0-9.npy", size=max_episode_steps
        )
    elif env_mode == "train":
        demand_generator = Demand("uniform", low=0, high=8, size=max_episode_steps)
    else:
        raise ValueError()

    array_index = {"on_hand": 0, "unreceived_pipeline": [3, 4, 5, 6], "unfilled_demand": 1, "latest_demand": 2}

    bsp_19 = BaseStockPolicy(target_levels=[19], array_index=array_index, state_dim_per_facility=7)

    bsp_20 = BaseStockPolicy(target_levels=[20], array_index=array_index, state_dim_per_facility=7)

    bsp_14 = BaseStockPolicy(target_levels=[14], array_index=array_index, state_dim_per_facility=7)

    if random_init:
        init_inventory = [0, 25]
        init_shipments = [[0, 9]] * 4
        init_sales_orders = [[0, 9]] * 4
    else:
        init_inventory = 12
        init_shipments = [[4, 4]] * 4
        init_sales_orders = [[4, 4]] * 4

    retailer = Node(
        name="retailer",
        initial_inventory=init_inventory,
        holding_cost=0.5,
        backorder_cost=1,
        policy=bsp_19,
        is_demand_source=True,
        demands=demand_generator,
    )
    wholesaler = Node(
        name="wholesaler", initial_inventory=init_inventory, holding_cost=0.5, backorder_cost=1, policy=bsp_20
    )
    distributor = Node(
        name="distributor", initial_inventory=init_inventory, holding_cost=0.5, backorder_cost=1, policy=bsp_20
    )
    manufacturer = Node(
        name="manufacturer", initial_inventory=init_inventory, holding_cost=0.5, backorder_cost=1, policy=bsp_14
    )
    supply_source = Node(name="external_supplier", is_external_supplier=True)
    nodes = [retailer, wholesaler, distributor, manufacturer, supply_source]

    arcs = [
        Arc(
            "external_supplier",
            "manufacturer",
            1,
            2,
            initial_shipments=init_shipments[0],
            initial_sales_orders=init_sales_orders[0],
            random_init=random_init,
        ),
        Arc(
            "manufacturer",
            "distributor",
            2,
            2,
            initial_shipments=init_shipments[1],
            initial_sales_orders=init_sales_orders[1],
            random_init=random_init,
        ),
        Arc(
            "distributor",
            "wholesaler",
            2,
            2,
            initial_shipments=init_shipments[2],
            initial_sales_orders=init_sales_orders[2],
            random_init=random_init,
        ),
        Arc(
            "wholesaler",
            "retailer",
            2,
            2,
            initial_shipments=init_shipments[3],
            initial_sales_orders=init_sales_orders[3],
            random_init=random_init,
        ),
    ]

    num_agent_managed_facilities = len(agent_managed_facilities)
    sn = SupplyNetwork(nodes=nodes, arcs=arcs, agent_managed_facilities=agent_managed_facilities)

    if box_action_space:
        action_space = gym.spaces.Box(0, 16, shape=(num_agent_managed_facilities,))
    else:
        action_space = gym.spaces.Discrete(17)

    observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7 * num_agent_managed_facilities,))

    return InventoryManagementEnvMultiPlayer(
        sn,
        max_episode_steps=max_episode_steps,
        action_space=action_space,
        observation_space=observation_space,
        return_dict=return_dict,
    )


def make_beer_game_normal(player="retailer", max_period=100, return_dict=False, random_init=False, seed=None):
    if seed:
        np.random.seed(seed)
    # else:
    # make sure the environment is random in parallel processing
    # np.random.seed(int(time.time()))

    demand_generator = Demand("normal", mean=10, sd=2, size=max_period)

    bsp_48 = BaseStockPolicy(48)
    bsp_43 = BaseStockPolicy(43)
    bsp_41 = BaseStockPolicy(41)
    bsp_30 = BaseStockPolicy(30)

    if random_init:
        init_inventory = [0, 21]
        init_shipments = [[0, 21]] * 4
        init_sales_orders = [[0, 21]] * 4
    else:
        init_inventory = 10
        init_shipments = [[10, 10]] * 4
        init_sales_orders = [[10, 10]] * 4

    demand_source = Node(name="is_demand_source", is_demand_source=True, demands=demand_generator)
    retailer = Node(
        name="retailer", initial_inventory=init_inventory, holding_cost=1.0, backorder_cost=10, policy=bsp_48
    )
    wholesaler = Node(
        name="wholesaler", initial_inventory=init_inventory, holding_cost=0.75, backorder_cost=0, policy=bsp_43
    )
    distributor = Node(
        name="distributor", initial_inventory=init_inventory, holding_cost=0.5, backorder_cost=0, policy=bsp_41
    )
    manufacturer = Node(
        name="manufacturer", initial_inventory=init_inventory, holding_cost=0.25, backorder_cost=0, policy=bsp_30
    )
    supply_source = Node(name="is_external_supplier", is_external_supplier=True)
    nodes = [demand_source, retailer, wholesaler, distributor, manufacturer, supply_source]

    arcs = [
        Arc(
            "is_external_supplier",
            "manufacturer",
            1,
            2,
            initial_shipments=init_shipments[0],
            initial_sales_orders=init_sales_orders[0],
            random_init=random_init,
        ),
        Arc(
            "manufacturer",
            "distributor",
            2,
            2,
            initial_shipments=init_shipments[1],
            initial_sales_orders=init_sales_orders[1],
            random_init=random_init,
        ),
        Arc(
            "distributor",
            "wholesaler",
            2,
            2,
            initial_shipments=init_shipments[2],
            initial_sales_orders=init_sales_orders[2],
            random_init=random_init,
        ),
        Arc(
            "wholesaler",
            "retailer",
            2,
            2,
            initial_shipments=init_shipments[3],
            initial_sales_orders=init_sales_orders[3],
            random_init=random_init,
        ),
        Arc("retailer", "is_demand_source", 0, 0),
    ]

    scn = SupplyNetwork(nodes=nodes, arcs=arcs, agent_managed_facilities=[player])

    return InventoryManagementEnv(scn, max_episode_steps=max_period, return_dict=return_dict)


def make_beer_game(
    agent_managed_facilities=None,
    demand_type="classic_beer_game",
    max_period=35,
    return_dict=False,
    random_init=False,
    seed=None,
) -> InventoryManagementEnvMultiPlayer:
    if agent_managed_facilities is None:
        agent_managed_facilities = ["retailer"]
    if seed:
        np.random.seed(seed)
    # else:
    # make sure the environment is random in parallel processing
    # np.random.seed(int(time.time()))

    if demand_type == "classic_beer_game":
        demand_generator = Demand("classic_beer_game")
    elif demand_type == "deterministic_random":
        demand_generator = Demand((8 + 2 * np.random.randn(max_period)).astype(int))
    elif demand_type == "normal":
        demand_generator = Demand("normal", mean=10, sd=2, size=max_period)
    elif demand_type == "uniform":
        demand_generator = Demand("uniform", low=0, high=2, size=max_period)
    else:
        raise ValueError()

    bs_32 = BaseStockPolicy(32)
    bs_24 = BaseStockPolicy(24)

    if random_init:
        init_inventory = [0, 25]
        init_shipments = [[0, 9]] * 4
        init_sales_orders = [[0, 9]] * 4
    else:
        init_inventory = 12
        init_shipments = [[4, 4]] * 4
        init_sales_orders = [[4, 4]] * 4

    retailer = Node(
        name="retailer", initial_inventory=init_inventory, policy=bs_32, is_demand_source=True, demands=demand_generator
    )
    wholesaler = Node(name="wholesaler", initial_inventory=init_inventory, policy=bs_32)
    distributor = Node(name="distributor", initial_inventory=init_inventory, policy=bs_32)
    manufacturer = Node(name="manufacturer", initial_inventory=init_inventory, policy=bs_24)
    supply_source = Node(name="external_supplier", is_external_supplier=True)
    nodes = [retailer, wholesaler, distributor, manufacturer, supply_source]

    arcs = [
        Arc(
            "external_supplier",
            "manufacturer",
            1,
            2,
            initial_shipments=init_shipments[0],
            initial_sales_orders=init_sales_orders[0],
        ),
        Arc(
            "manufacturer",
            "distributor",
            2,
            2,
            initial_shipments=init_shipments[1],
            initial_sales_orders=init_sales_orders[1],
        ),
        Arc(
            "distributor",
            "wholesaler",
            2,
            2,
            initial_shipments=init_shipments[2],
            initial_sales_orders=init_sales_orders[2],
        ),
        Arc(
            "wholesaler",
            "retailer",
            2,
            2,
            initial_shipments=init_shipments[3],
            initial_sales_orders=init_sales_orders[3],
        ),
    ]

    num_agent_managed_facilities = len(agent_managed_facilities)
    sn = SupplyNetwork(nodes=nodes, arcs=arcs, agent_managed_facilities=agent_managed_facilities)

    action_space = gym.spaces.MultiDiscrete([17] * num_agent_managed_facilities)
    observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7 * num_agent_managed_facilities,))

    return InventoryManagementEnvMultiPlayer(
        sn,
        max_episode_steps=max_period,
        action_space=action_space,
        observation_space=observation_space,
        return_dict=return_dict,
    )
