import numpy as np
import scipy.stats as st

import sys
sys.path.append("../")
from superstatistics import estimate_long_time

def test_functions():
    for alpha in [1.9, 1.8, 1.7]:

        timeseries = st.levy_stable.rvs(1.9, 0, size=100000)
        lag = np.linspace(1, 50, 50, dtype=int)

        # Testing wrong-shape array
        try:
            estimate_long_time(np.stack((X,X), axis=1))
        except Exception:
            pass

        # Testing simple estiamtions
        T = estimate_long_time(timeseries)

        assert isinstance(T, int)
