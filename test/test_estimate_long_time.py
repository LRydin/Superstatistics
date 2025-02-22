import numpy as np
import scipy.stats as st

import sys
sys.path.append("../")
from superstatistics import estimate_long_time

def test_estimate_long_time():
    for alpha in [1.9, 1.8, 1.7]:

        # Timeseries
        timeseries = st.levy_stable.rvs(1.9, 0, size=100000)
        lag = np.linspace(1, 50, 50, dtype=int)

        # Testing wrong-shape array
        try:
            estimate_long_time(np.stack((X,X), axis=1))
        except Exception:
            pass

        # Testing simple estimation using root solver
        T = estimate_long_time(timeseries)
        assert isinstance(T, int)

        # Testing use of lag
        T = estimate_long_time(timeseries, lag=lag)
        assert isinstance(T, np.ndarray)

        # Testing use of lag and quantiles
        T = estimate_long_time(timeseries, lag=lag, quantiles=[0.1,0.5,0.9])
        assert isinstance(T, tuple)
        assert isinstance(T[0], np.ndarray)
        assert isinstance(T[1], list)

        # Testing root solver with assigned threshold and moment
        T = estimate_long_time(timeseries, lag=lag, threshold=0., moment='skew')

        # Testing missing threshold
        try:
            estimate_long_time(timeseries, moment='std')
        except Exception:
            pass
