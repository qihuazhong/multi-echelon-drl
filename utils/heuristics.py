from typing import Union, Optional, List
import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecNormalize


class InventoryPolicy:
    def get_order_quantity(self, observation: Union[dict, np.ndarray]):
        raise NotImplementedError


class DRLPolicy(InventoryPolicy):
    """Deep reinforcement learning agent policy that maps observations to order quantities."""

    def __init__(
        self, model: BaseAlgorithm, vec_normalize: Optional[VecNormalize] = None
    ):
        self.model: BaseAlgorithm = model
        self.vec_normalize: Optional[VecNormalize] = vec_normalize

    def update_model(self, target_model: BaseAlgorithm) -> None:
        self.model.policy.load_state_dict(target_model.policy.state_dict())

    def update_vec_norm(self, target_vec_normalize: VecNormalize) -> None:
        self.vec_normalize.__setstate__(target_vec_normalize.__getstate__())
        self.vec_normalize.num_envs = 1

    def get_order_quantity(self, observation: Union[dict, np.ndarray]):
        if isinstance(observation, np.ndarray):
            raise NotImplementedError(
                f"{type(observation)} observation type not supported yet"
            )
            # return self.model.predict(observation)

        elif isinstance(observation, dict):
            observation = np.array(
                [list(state.values()) for agent, state in observation.items()],
                dtype=np.float32,
            ).flatten()

            actions, _ = self.model.predict(
                self.vec_normalize.normalize_obs(observation), deterministic=True
            )
            # print(observation, actions)
            return actions - 8 + observation[2]


class BaseStockPolicy(InventoryPolicy):
    def __init__(
        self,
        target_levels: Union[int, float, List, np.ndarray],
        state_name_to_index: Optional[dict] = None,
        state_dim_per_facility=7,
        lb=-np.inf,
        ub=np.inf,
        rule: str = "a",
    ):
        """

        Args:
            target_levels:
            state_name_to_index:
            state_dim_per_facility:
        """
        if not isinstance(target_levels, np.ndarray):
            target_levels = np.array(target_levels)

        self.target_levels = target_levels.reshape(-1)
        self.array_index = state_name_to_index
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

            on_hands = agent_states[
                np.arange(self.array_index["on_hand"], state_len, step)
            ]
            unfilled_demands = agent_states[
                np.arange(self.array_index["unfilled_demand"], state_len, step)
            ]

            if "on_order" in self.array_index.keys():
                on_orders = agent_states[
                    np.arange(self.array_index["on_order"], state_len, step)
                ]
            else:
                on_orders = np.sum(
                    [
                        agent_states[np.arange(i, state_len, step)]
                        for i in self.array_index["unreceived_pipeline"]
                    ],
                    axis=0,
                )

            if self.rule == "d+a":
                latest_demand = agent_states[
                    np.arange(self.array_index["latest_demand"], state_len, step)
                ]
                quantities = (
                    self.target_levels
                    - (on_hands + on_orders - unfilled_demands)
                    - latest_demand
                )
            else:
                quantities = self.target_levels - (
                    on_hands + on_orders - unfilled_demands
                )

        # TODO write test to make sure the two types of input would give the same result
        elif isinstance(observation, dict):
            quantities = []
            for agent, agent_state in observation.items():
                on_hands = agent_state["on_hand"]
                unfilled_demands = agent_state["unfilled_demand"]

                if "on_order" in self.array_index.keys():
                    on_orders = agent_state["on_order"]
                else:
                    on_orders = sum(
                        [agent_state[f"unreceived_pipeline_{i}"] for i in range(4)]
                    )

                # unfilled_demand = agent_state['unfilled_demand']
                quantity = self.target_levels - (
                    on_hands + on_orders - unfilled_demands
                )
                if self.rule == "d+a":
                    quantity -= agent_state["latest_demand"]
                quantities.append(quantity)
        else:
            raise TypeError

        return np.clip(quantities, self.lb, self.ub)
