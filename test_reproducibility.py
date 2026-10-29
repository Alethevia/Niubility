import numpy as np
from ultra_trading_env import UltraTradingEnv

def test_fixed_seed_replays_every_market_component():
    a, b = UltraTradingEnv(), UltraTradingEnv()
    oa, _ = a.reset(seed=17)
    ob, _ = b.reset(seed=17)
    np.testing.assert_array_equal(oa, ob)
    for _ in range(20):
        xa, xb = a.step([0.5, 0.2]), b.step([0.5, 0.2])
        np.testing.assert_array_equal(xa[0], xb[0])
        assert xa[1:4] == xb[1:4]
