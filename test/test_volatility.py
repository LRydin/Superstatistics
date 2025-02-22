import numpy as np
import scipy.stats as st

import sys
sys.path.append("../")
from superstatistics import volatility, estimate_long_time

def test_volatility():
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

        # Basic call
        beta = volatility(timeseries, T)
        assert isinstance(beta, np.ndarray)

        # Changing brackets
        beta = volatility(timeseries, T, bracket=[3, 8])
        assert isinstance(beta, np.ndarray)
