import numpy as np
import scipy.stats as st

import sys
sys.path.append("../")
from superstatistics import (estimate_long_time, volatility,
    find_best_distribution)

def test_distribution_fiting():
    for alpha in [1.9, 1.8, 1.7]:

        # Timeseries
        timeseries = st.levy_stable.rvs(1.9, 0, size=100000)
        lag = np.linspace(1, 50, 50, dtype=int)

        # Testing wrong-shape array
        try:
            estimate_long_time(np.stack((X,X), axis=1))
        except Exception:
            pass

        # Obtain a long time T for volatility
        T = estimate_long_time(timeseries)

        # Obtain a volatility
        beta = volatility(timeseries, T)

        # Changing brackets
        KL, d = find_best_distribution(beta)
        assert isinstance(KL, dict)
        assert isinstance(d, dict)
