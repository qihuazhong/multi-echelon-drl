from typing import Union, List
import types
import numpy as np


def wrap_action_d_plus_a(env, offset=-8, action_multiplier: int = 1, lb: int = 0, ub: int = 16):
    """A wrapper function that modify the step() function of the given environment.

    After wrapping the environment, the action values provided to the step() function will be interpreted as the 'a' in
    'd+a' order rule, where 'd' is the latest demand and 'a' is the action value. The quantity d+a will be ordered
    instead of a.

    Args:
        env:
        offset:
        lb:
        ub:
    Returns:
        A wrapped environment with modified step() function.
    """
    env.wrappee_step = env.step

    def wrapped_step(self, action):
        states = [self.unwrapped.sn.get_state(facility) for facility in self.unwrapped.sn.agent_managed_facilities]
        modified_action = np.array([state["latest_demand"] for state in states]) + action * action_multiplier + offset

        return self.wrappee_step(np.clip(modified_action, lb, ub))

    env.step = types.MethodType(wrapped_step, env)

    return env


def wrap_action_a(env, offset=-8, action_multiplier: int = 1, lb: int = 0, ub: int = 16):
    """A wrapper function that modify the step() function of the given environment.

    After wrapping the environment, the action values provided to the step() function will be interpreted as the 'a' in
    'd+a' order rule, where 'd' is the latest demand and 'a' is the action value. The quantity d+a will be ordered
    instead of a.

    Args:
        env:
        offset:
        lb:
        ub:
    Returns:
        A wrapped environment with modified step() function.
    """
    env.wrappee_step = env.step

    def wrapped_step(self, action):
        states = [self.unwrapped.sn.get_state(facility) for facility in self.unwrapped.sn.agent_managed_facilities]
        # modified_action = np.array([state['latest_demand'] for state in states]) + action + offset
        modified_action = np.array(action * action_multiplier).reshape(-1)
        return self.wrappee_step(np.clip(modified_action, lb, ub))

    env.step = types.MethodType(wrapped_step, env)

    return env


# def d_plus_a(states: Union[List[dict], np.ndarray], action: np.ndarray, offset: int = 0,
#              state_dim: int = None,
#              demand_index: int = None):
#     if isinstance(states, list):
#         modified_action = np.array([state['latest_demand'] for state in states]) + action + offset
#     elif isinstance(states, np.ndarray):
#         modified_action = np.array([states[i] for i in range(demand_index, state_dim, states.size)]) + action + offset
#     else:
#         raise TypeError(f'states must be type of list of dict or numpy.ndarray')
#
#     return np.maximum(modified_action, 0)
