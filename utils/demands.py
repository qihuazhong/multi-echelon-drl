import logging
import numpy as np
import pandas as pd


class Demand:
    def __init__(
        self,
        demand_pattern="classic_beer_game",
        data_path=None,
        size=100,
        low=None,
        high=None,
        mean=None,
        sd=None,
        n=None,
        p=None,
    ):
        """
        Args:
            demand_pattern: can be a string ('normal', 'uniform', 'classic_beer_game') to specify a stochastic demand
                pattern or a list of numbers to specify deterministic _demands.
            size: number of periods in an episode
            low: inclusive, must be provided when uniform pattern is specified
            high: inclusive, must be provided when uniform pattern is specified
            mean: mean of a normal distribution, must be provided when normal pattern is specified
            sd: standard deviation of a normal distribution, must be provided when normal pattern is specified
            n: parameter for negative binomial distribution
            p: parameter for negative binomial distribution
        """
        self.demands_pattern = demand_pattern
        self._demands: np.ndarray = None

        self.low = low
        self.high = high
        self.mean = mean
        self.sd = sd
        self.n = n
        self.p = p
        if data_path is not None:
            self.samples = pd.read_csv(data_path, header=None).values.reshape(-1)
            self.samples_length = self.samples.shape[0]
            self.sample_pointer = 0
            self.size = self.samples_length
        else:
            self.samples = None
            self.samples_length = None
            self.sample_pointer = None
            self.size = size + 3  # TODO

        self.reset()

    def reset(self):
        if isinstance(self.demands_pattern, (np.ndarray, list)):
            self._demands = np.asarray(self.demands_pattern)

        elif self.demands_pattern == "uniform":
            if (self.low is None) or (self.high is None):
                raise ValueError('"low" and "high" need to be provided when uniform distribution pattern is specified')
            self._demands = np.random.randint(self.low, self.high + 1, self.size)

        elif self.demands_pattern == "normal":
            self._demands = np.round(np.maximum(self.mean + self.sd * np.random.randn(self.size), 0)).astype(int)
            if (self.mean is None) or (self.sd is None):
                raise ValueError('"mean" and "sd" need to be provided when normal distribution pattern is specified')

        elif self.demands_pattern == "classic_beer_game":
            self._demands = np.array([4] * 4 + [8] * (self.size - 4))

        elif self.demands_pattern == "negative_binomial":
            self._demands = np.random.negative_binomial(self.n, self.p, size=104)

        elif self.demands_pattern == "bucketized_negative_binomial":
            nb = np.random.negative_binomial(self.n, self.p, size=104)
            self._demands = np.round(nb.clip(min=0, max=399) / 10).astype(int)

        elif self.demands_pattern == "samples":
            self._demands = self.samples
            self.sample_pointer = 0

        else:
            raise ValueError("Demand pattern not recognized")

    def generator(self):
        # Implemented as a generator, so that the argument 'period' is not needed
        period = 0
        while period <= self.size:
            logging.info(f"Demand generated: {self._demands[period % self.size].item()}")
            yield self._demands[period % self.size].item()
            period += 1
