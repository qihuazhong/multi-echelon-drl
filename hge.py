import sys
import os
from abc import ABC
from typing import Optional, Tuple, Union, Type, Dict, Any

import numpy as np
import torch as th
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.preprocessing import maybe_transpose
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.utils import is_vectorized_observation
from stable_baselines3.td3.policies import TD3Policy

sys.path.append(os.path.abspath(os.path.join("..")) + "/snim")


# from policies import BaseStockPolicy
from utils.heuristics import BaseStockPolicy, InventoryPolicy

import gymnasium as gym
from stable_baselines3 import DQN, PPO, TD3, A2C
from stable_baselines3.common.noise import NormalActionNoise, ActionNoise
from stable_baselines3.common.callbacks import BaseCallback


class HgeRateCallback(BaseCallback):
    def __init__(self, verbose: int = 0, mu_start=0.5, mu_end=0.0, decay_periods: int = 1000):
        super().__init__(verbose)

        self.mu_start = mu_start
        self.mu_end = mu_end
        self.decay_periods = decay_periods

    def _on_step(self) -> bool:
        return True

    def on_step(self) -> bool:
        super().on_step()

        self.model.hge_rate = self.mu_start - (self.mu_start - self.mu_end) * min(
            1.0, self.num_timesteps / 100 / self.decay_periods
        )

        return self._on_step()


class HgeTD3(TD3):
    """
    Include the heuristic-guided exploration rate
    """

    def __init__(
        self,
        policy: Union[str, Type[TD3Policy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-3,
        buffer_size: int = 1_000_000,
        learning_starts: int = 100,
        batch_size: int = 100,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = (1, "episode"),
        gradient_steps: int = -1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[ReplayBuffer] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        policy_delay: int = 2,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        hge_rate: float = 0.0,
        heuristic: InventoryPolicy = None,
    ):

        self.hge_rate = hge_rate
        self.heuristic = heuristic

        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            replay_buffer_class,
            replay_buffer_kwargs,
            optimize_memory_usage,
            policy_delay,
            target_policy_noise,
            target_noise_clip,
            tensorboard_log,
            policy_kwargs,
            verbose,
            seed,
            device,
            _init_setup_model,
        )

    def _sample_action(
        self, learning_starts: int, action_noise: Optional[ActionNoise] = None, n_envs: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Override the off_policy_algorithm _sample_action() to pass both normalized and original observation to
        the predict() method
        """
        # Select action randomly or according to policy
        if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
            # Warmup phase
            unscaled_action = np.array([self.action_space.sample() for _ in range(n_envs)])
        else:
            # Note: when using continuous actions,
            # we assume that the policy uses tanh to scale the action
            # We use non-deterministic action in the case of SAC, for TD3, it does not matter
            unscaled_action, _ = self.predict(self._last_obs, deterministic=False, original_obs=self._last_original_obs)

        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.action_space, gym.spaces.Box):
            scaled_action = self.policy.scale_action(unscaled_action)

            # Add noise to the action (improve exploration)
            if action_noise is not None:
                scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

            # We store the scaled action in the buffer
            buffer_action = scaled_action
            action = self.policy.unscale_action(scaled_action)
        else:
            # Discrete case, no need to normalize or clip
            buffer_action = unscaled_action
            action = buffer_action
        return action, buffer_action

    def predict(
        self,
        observation: np.ndarray,
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
        original_obs: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:

        """
        Overrides the base_class predict function to include heuristic-guided exploration.

        """
        if not deterministic and np.random.rand() < self.hge_rate**2:
            if is_vectorized_observation(maybe_transpose(observation, self.observation_space), self.observation_space):
                if isinstance(self.observation_space, gym.spaces.Dict):
                    n_envs = observation[list(observation.keys())[0]].shape[0]
                else:
                    n_envs = observation.shape[0]
                # action = np.array([self.action_space.sample() for _ in range(n_batch)])
                action = np.array([self.heuristic.get_order_quantity(original_obs[i]) for i in range(n_envs)])
                # action = self.heuristic.get_order_quantity(original_obs, n_envs=n_envs)
            else:
                # get action from the heuristic
                action = self.heuristic.get_order_quantity(original_obs)
                # print(observation, action)
                # action = np.array(self.action_space.sample())
        else:
            action, state = self.policy.predict(observation, state, episode_start, deterministic)
        return action, state

        # return super().predict(observation, state, episode_start, deterministic)
