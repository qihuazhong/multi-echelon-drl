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

        self.model.hge_rate = \
            self.mu_start - (self.mu_start - self.mu_end) * min(1.0, self.num_timesteps/100/self.decay_periods)

        return self._on_step()

class SaveEnvStatsCallback(BaseCallback):

    def __init__(self, verbose: int = 0, env_save_path: str = None):
        super().__init__(verbose)
        self.env_save_path = env_save_path
        
    def _on_step(self) -> bool:
        return True

    def on_step(self) -> bool:
        super().on_step()
        self.training_env.save(f'{self.env_save_path}best_env')
