import numpy as np
# import unittest
from utils.demands import Demand


def test_uniform_demand(low=0, high=2):
    demand = Demand('uniform', low=low, high=high, size=100).generator()

    realized_low = +999999
    realized_high = -999999

    for t in range(100):
        d = next(demand)

        if d < realized_low:
            realized_low = d
        if d > realized_high:
            realized_high = d

    return realized_low, realized_high


assert test_uniform_demand(low=0, high=2) == (0, 2)
assert test_uniform_demand(low=0, high=8) == (0, 8)


def test_normal_demand(mean=10, sd=4, size=100000):
    demand = Demand('normal', mean=mean, sd=sd, size=size).generator()
    demands = []
    for t in range(size):
        demands.append(next(demand))

    demands = np.array(demands)

    realized_mean = np.mean(demands)
    realized_std = np.std(demands)
    realized_low = np.min(demands)

    return realized_mean, realized_std, realized_low


# if __name__ == '__main__':
#     unittest.main()

mean, std, low = test_normal_demand(mean=10, sd=4)

assert abs(mean - 10) < 0.1
assert abs(std - 4) < 0.04
assert low == 0
