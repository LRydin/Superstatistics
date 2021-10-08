# This is the main module of superstatistics.

import warnings

import numpy as np
from scipy import stats
from scipy.optimize import root_scalar

def estimate_long_time(timeseries: np.array, lag: np.array=None,
                       threshold: float=3, moment: str='kurtosis',
                       tol: float=0.05, full: bool=False) -> np.array:
    """
    To find the long superstatistical time :math:`T` one needs to examine the
    distribution of the segments of the data.

    Parameters
    ----------
    timeseries: np.ndarray
        A 1-dimensional timeseries of length `N`.

    lag: np.ndarray of ints (default `None`)
        An array with the segments lengths to examine. If `None` or not declared,
        it will try to find the long superstatistical time via a Newton–Raphson
        method. Else needs a list or array of integers `>1` on which to segment
        the data.

    threshold: float (default `3`)
        The threshold expected to be crossed to estimate the long
        superstatistical time :math:`T`. Defaults to `3`, as the most common
        superstatistics analysis relies on Gaussian increments, which have a
        kurtosis of `3`.

    moment: str (default `'kurtosis'`)
        The moment under examination in finding the long superstatistical time.
        Should be a function of ``scipy.stats`` such as `['skew', 'kurtosis']`
        or ``numpy``'s `['mean','var']`. Defaults to `'kurtosis'`, as this is
        the common central statistical moment under examination.

    tol: float (default `0.05`)
        The percentage error acceptable to find the long time. Should be a
        positive value between `0` and `1`.

    full: bool (default `False`)
        If `full` is `True` and a `lag` is given, returns
        the full scan over `T` for the given lag. Note that the lag should be
        an array of list of a set of integers `>1`.

    Returns
    -------
    T: int, np.array
        The long superstatistical time (or an set of values, in case more than)
        one is found. While using the standard Newton–Raphson method, only one
        value is returned. If `full` is `True` and a `lag` is given, returns
        the full scan over `T`. Useful for plotting `T` vs `lag`. Suggested to
        first run without a single time to locate `T` and they adjust `lag`
        accordingly

    Notes
    -----

     - `Wikipedia: Newton Method <https://en.wikipedia.org/wiki/Newton%27s_method>`_

    References
    ----------
    .. [Beck2003] Beck, C., & Cohen, E. G. (2003). Superstatistics. Physica A:
        Statistical mechanics and its applications, 322, 267--275.
        doi: 10.1016/S0378-4371(03)00019-0
    """

    # length of the timeseries
    N = timeseries.shape[0]

    # check if the selected moment is either in numpy or scipy
    try:
        selected_moment = getattr(np, moment)
    except:
        try:
            selected_moment = getattr(stats, moment)
        except:
            raise Exception("The moment you have selected is "
                            "not part of numpy or scipy.")

    # Check if lag is a sequence of integers and bounded
    if isinstance(lag, (np.ndarray, list)):
        if not all(isinstance(x, (np.integer, int)) for x in lag):
            raise ValueError("lag must be a sequence of integers.")
        if min(lag) < 2 or lag[lag > N // 2 + 1 ].size:
            raise ValueError("Any element of lag must be >1 and smaller "
                             "than half the size of the timeseries.")

    kwargs_for_scipy = {}
    if moment == 'kurtosis':
        kwargs_for_scipy['fisher'] = False

    # function to either run through or find root
    def fun(j):
        if j < 3:
            j = 3
        if j > (N // 2):
            j = N // 2
        print(j)
        j = int(j)
        x = np.mean(selected_moment(
                        timeseries[:N - N % j].reshape((N - N % j) // j, j),
                        axis=1,
                        **kwargs_for_scipy)
                    )
        return x - threshold

    # bulk of the operation
    # if a lag is given
    if isinstance(lag, (np.ndarray, list)):
        part_T = np.zeros(lag.shape[0])
        for i, ele in enumerate(lag):
            part_T[i] = fun(ele)

    if lag is None:
        # catch warning, as this use root_scalar with integers, which can
        # result in warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            T = root_scalar(fun, x0=10, x1=30)

    if lag is None:
        if T.converged:
            return int(T.root)
        else:
            warnings.warn("The root solver did not converge to a "
                          "long time T.", RuntimeWarning)

    if isinstance(lag, (np.ndarray, list)):
        if full:
            return part_T
        else:
            # find the lag that minimises the problem

            if np.min(np.abs(np.array(part_T) - 0)) > tol:
                raise ValueError("The given lag does not cover the transition "
                                 "wherein the long time T can be found.")

            T = lag[np.argmin(np.abs(np.array(part_T) - 0))]
            return T
