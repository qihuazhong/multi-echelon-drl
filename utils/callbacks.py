import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam
from stable_baselines3.common.noise import NormalActionNoise


class HgeRateCallback(BaseCallback):
    def __init__(self, verbose: int = 0, mu_start=0.5, mu_end=0.0, decay_periods: int = 1000, action_noise_annealing=False):
        super().__init__(verbose)

        self.mu_start = mu_start
        self.mu_end = mu_end
        self.decay_periods = decay_periods

        self.action_noise_std = 0.4
        self.action_noise_annealing = action_noise_annealing

    def _on_step(self) -> bool:
        return True

    def on_step(self) -> bool:
        super().on_step()

        self.model.hge_rate = self.mu_start - (self.mu_start - self.mu_end) * min(
            1.0, self.num_timesteps / 100 / self.decay_periods
        )

        if self.action_noise_annealing:
            self.model.action_noise = NormalActionNoise(
                mean=np.zeros(4),
                sigma=self.action_noise_std * (((10_000_000 - self.num_timesteps) / 10_000_000) ** 2) * np.ones(4),
            )

        return self._on_step()


class SaveEnvStatsCallback(BaseCallback):
    def __init__(self, verbose: int = 0, env_save_path: str = None):
        super().__init__(verbose)
        self.env_save_path = env_save_path

    def _on_step(self) -> bool:
        return True

    def on_step(self) -> bool:
        super().on_step()
        self.training_env.save(f"{self.env_save_path}best_env")
        return True


class HParamCallback(BaseCallback):
    """
    Saves the hyperparameters and metrics at the start of the training, and logs them to TensorBoard.
    """

    def __init__(self, hparam_dict: dict, verbose: int = 0):
        super().__init__(verbose)
        self.hparam_dict = hparam_dict

    def _on_training_start(self) -> None:
        self.hparam_dict["algorithm"] = self.model.__class__.__name__

        # define the metrics that will appear in the `HPARAMS` Tensorboard tab by referencing their tag
        # Tensorbaord will find & display metrics from the `SCALARS` tab
        metric_dict = {
            "eval/mean_reward": 0,
            "train/loss": 0.0,
            "time/fps": 0,
        }
        self.logger.record(
            "hparams",
            HParam(self.hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self) -> bool:
        return True
