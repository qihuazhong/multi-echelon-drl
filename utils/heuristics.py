from typing import Union, Optional, List
import numpy as np


class InventoryPolicy:

    def get_order_quantity(self, observation: Union[dict, np.ndarray]):
        raise NotImplementedError


class BaseStockPolicy(InventoryPolicy):


class BaseStockPolicy(InventoryPolicy):
    def __init__(
        self,
        target_levels: Union[int, float, List, np.ndarray],
        array_index: Optional[dict] = None,
        state_dim_per_facility=7,
        lb=0,
        ub=np.inf,
        rule: str = "a",
    ):
        """

        Args:
            target_levels:
            array_index:
            state_dim_per_facility:
        """
        if not isinstance(target_levels, np.ndarray):
            target_levels = np.array(target_levels)

        self.target_levels = target_levels.reshape(-1)
        self.array_index = array_index
        self.state_dim_per_facility = state_dim_per_facility

        self.lb = lb
        self.ub = ub
        self.rule = rule

    def get_order_quantity(self, observation: Union[dict, np.ndarray]):
        """

        Args:
            observation:
        Returns:

        """
        if isinstance(observation, np.ndarray):
            step = self.state_dim_per_facility

            agent_states = observation.reshape(-1)
            state_len = agent_states.shape[0]

            on_hands = agent_states[np.arange(self.array_index["on_hand"], state_len, step)]
            unfilled_demands = agent_states[np.arange(self.array_index["unfilled_demand"], state_len, step)]
            on_orders = np.sum(
                [agent_states[np.arange(i, state_len, step)] for i in self.array_index["unreceived_pipeline"]], axis=0
            )
            if self.rule == "d+a":
                latest_demand = agent_states[np.arange(self.array_index["latest_demand"], state_len, step)]
                quantities = self.target_levels - (on_hands + on_orders - unfilled_demands) - latest_demand
            else:
                quantities = self.target_levels - (on_hands + on_orders - unfilled_demands)

        # TODO write test to make sure the two types of input would give the same result
        elif isinstance(observation, dict):
            quantities = []
            for agent, agent_state in observation.items():
                on_hands = agent_state["on_hand"]
                unfilled_demands = agent_state["unfilled_demand"]

                on_orders = sum([agent_state[f"unreceived_pipeline_{i}"] for i in range(4)])
                # unfilled_demand = agent_state['unfilled_demand']
                quantity = self.target_levels - (on_hands + on_orders - unfilled_demands)
                if self.rule == "d+a":
                    quantity -= agent_state["latest_demand"]
                quantities.append(quantity)
        else:
            raise TypeError

        return np.clip(quantities, self.lb, self.ub)
