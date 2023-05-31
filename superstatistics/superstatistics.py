# This is the main module of superstatistics.

import warnings

import numpy as np
from scipy import stats
from scipy.optimize import root_scalar
from functools import partial

def estimate_long_time(timeseries: np.array, lag: np.array=None,
                       threshold: float=None, moment: str='kurtosis',
                       tol: float=0.05, quantiles: list=[False]) -> np.array:
    """
    To find the long superstatistical time :math:`T` one needs to examine the
    distribution of the segments of the data.

    Parameters
    ----------
    timeseries: np.ndarray
        A 1-dimensional timeseries of length `N`.

    lag: np.ndarray of ints (default `None`)
        An array with the segments lengths to examine. If `None` or not
        declared, it will try to find the long superstatistical time via
        scalar-root finding method. It is advised to first not use a lag, to
        find the typical long superstatistical time with a root finder, and then
        after adjust the lag for plotting purposes. If given, `lag > 1`.

    threshold: float (default `3.`)
        The threshold expected to be crossed to estimate the long
        superstatistical time :math:`T`. Defaults to `3.`, as the most common
        superstatistics analyses rely on Gaussian increments, which have a
        kurtosis of `3`. If `'skew'` or `'mean'` from scipy are selected and
        threshold is `None`, then threshold defaults to `0.`.

    moment: str (default `'kurtosis'`)
        The moment under examination in finding the long superstatistical time.
        Should be a function of `scipy.stats` such as `['skew', 'kurtosis']`
        or `numpy`'s `['mean','var']`. Defaults to `'kurtosis'`, as this is
        the common central statistical moment under examination.

    tol: float (default `0.05`) - Not implemented
        The percentage error acceptable to find the long time. Should be a
        positive value between `0` and `1`.

    quantiles: list (default `[False]`)
        If values are given, returns the quantiles, accordingly, around the
        estimated long superstatistical time (if lag is given).

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
     - `SciPy scalar root finder <https://docs.scipy.org/doc/scipy/reference\
    /generated/scipy.optimize.root_scalar.html#scipy.optimize.root_scalar>`_

    References
    ----------
    .. [Beck2003] Beck, C., & Cohen, E. G. (2003). Superstatistics. Physica A:
        Statistical mechanics and its applications, 322, 267--275.
        doi: 10.1016/S0378-4371(03)00019-0
    """

    # Force lag to be ints if given, ensure lag > 1
    if lag is not None:
        lag = lag[lag > 1]
        lag = lag.astype(int)

    # Assert if timeseries is 1 dimensional
    if timeseries.ndim > 1:
        assert timeseries.shape[1] == 1, "Timeseries needs to be 1 dimensional"

    # length of the timeseries
    N = timeseries.shape[0]

    # check if the selected moment is either in numpy or scipy
    try:
        selected_moment = getattr(stats, moment)
    except:
        try:
            selected_moment = getattr(np, moment)
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

    # kwargs to pass to scipy
    kwargs_for_scipy = {}
    kwargs_for_scipy['nan_policy'] = 'omit'
    if moment == 'kurtosis':
        kwargs_for_scipy['fisher'] = False
        if not threshold:
            threshold = 3.

    if moment in ['skew', 'mean', 'nanmean']:
        if not threshold:
            threshold = 0.

    if threshold is None:
        raise ValueError("A threshold (float) must be given.")

    # function to either run through or find root
    def fun(j, operation):
        # limiting conditions
        j = 3 if j < 3 else int(j)
        j = int(N // 2) if j > (N // 2) else int(j)
        x = selected_moment(
                        timeseries[:N - N % j].reshape((N - N % j) // j, j),
                        axis=1,
                        **kwargs_for_scipy)
        return operation(x[x>0]) - threshold

    # bulk of the operation
    def given_lag(operation):
        part_T = np.zeros(lag.shape[0])
        for i, ele in enumerate(lag):
            part_T[i] = fun(ele, operation=operation) + threshold
        return part_T

    if isinstance(lag, (np.ndarray, list)):
        part_T = given_lag(np.nanmean)
        if quantiles[0]:
            part_T_quantiles = [given_lag(
                            partial(np.quantile, q=q)
                          ) for q in quantiles]
    if lag is None:
        # catch warning, as this use root_scalar with integers, which can
        # result in warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            T = root_scalar(fun, args=(np.nanmean), x0=10, x1=30)

    if lag is None:
        if T.converged and T.root > 2 and T.root < N // 2 - 1:
            return int(T.root)
        else:
            raise ValueError("The root solver did not converge to a "
                             "long time T.")

    if isinstance(lag, (np.ndarray, list)):
        # find the lag that minimises the problem

        # seems to not work. A warning to increase lag would be nice
        # if np.min(np.abs(np.array(part_T) - 0)) > tol:
        #     warnings.warn("The given lag does not cover the transition "
        #                   "wherein the long time T can be found.",
        #                   RuntimeWarning)

        # T = lag[np.argmin(np.abs(np.array(part_T) - 0))]
        if quantiles[0]:
            return part_T, part_T_quantiles
        else:
            return part_T


def volatility(timeseries: np.array, T: int, bracket: list=[5, 5]) -> np.array:
    """
    Extract the (inverse) volatility β of a timeseries given a long
    superstatistical time :math:`T`. The (inverse) volatility β is given by

    .. math::

        β = \frac{1}{\langle x(t)^2 \rangle_{T} - \langle x(t) \rangle_{T}^2}.

    where :math:`x(t)` is the timeseries and :math:` \langle\cdot\rangle` is the
    expected value. The (inverse) volatility β is bounded between
    :math:`(0,\infty)` and it is the central variable that is studied in
    superstatistics.

    Parameters
    ----------
    timeseries: np.ndarray
        A 1-dimensional timeseries of length `N`.

    T: int
        The long superstatistical time :math:`T`. It can be obtained with
        `estimate_long_time()`.

    bracket: list of 2 float (default `[5,5]`)
        Determines the bounds in standard devations around the mean that is kept
        after each moment estimation to remove extreme values.

    Returns
    -------
    beta: np.array
        The (inverse) volatilities β.

    References
    ----------
    .. [Beck2003] Beck, C., & Cohen, E. G. (2003). Superstatistics. Physica A:
        Statistical mechanics and its applications, 322, 267--275.
        doi: 10.1016/S0378-4371(03)00019-0
    """
    # length of the timeseries
    N = timeseries.shape[0]

    beta = np.nanvar(timeseries[:N - N % T].reshape((N - N % T) // T, T),
                     axis=1)

    beta = 1/beta[beta>0]

    beta = beta[(beta > np.mean(beta) - bracket[0] * np.std(beta)) &
                (beta < np.mean(beta) + bracket[1] * np.std(beta))]

    return beta

def _fit_distributions(beta: np.array, dists: list=None,
    lim: list=[None, None]) -> dict:

    if not isinstance(lim, list):
        raise ValueError("lim must be a list with a numerical lower and upper "
                         "bound. 'None' can be use for no bound.")
    if not all(isinstance(x, (float, int, type(None))) for x in lim):
        raise ValueError("lim must be a list with a numerical lower and upper "
                         "bound. `None` can be use for no bound.")

    dists = ['lognorm', 'gengamma', 'invgamma', 'f']

    kwargs_for_scipy = {}
    for dist in dists:
        kwargs_for_scipy[dist] = {}
        if dist == 'gengamma':
            kwargs_for_scipy['gengamma'] = {'fc': 1}

    # ensure beta is positive and apply bounds
    beta = beta[beta>0]
    if lim[0]:
        beta = beta[(beta>lim[0])]
    if lim[1]:
        beta = beta[(beta<lim[1])]

    dictionary_of_fits = {dist:
        getattr(stats, dist).fit(beta, **kwargs_for_scipy[dist])
        for dist in dists}

    return dictionary_of_fits

def find_best_distribution(beta: np.array, bins=300, dists: list=None,
    lim: list=[None, None]) -> np.array:
    """
    Estimates the Kullback–Leibler divergence between the distribution of the
    volatility beta and four standard distributions commonly used in
    superstatistics: lognormal, gamma, inverse-gamma, and the F distribution.

    Parameters
    ----------
    beta: np.array
        The (inverse) volatilities β.

    ... unfinished

    Returns
    -------
    KL: dict
        A dictionary with the scipy distributions used to fit the data and the
        resulting Kullback–Leibler divergence between the distribution of the
        volatility beta and the fitted distributions. Note that different
        distributions have different number of parameters. For the four standard
        distributions commonly used in superstatistics, lognormal distribution
        has 2 parameters, gamma and inverse-gamma have 3 parameters, and the F
        distribution has 4 parameters.

    """

    if not dists:
        dists = ['lognorm', 'gengamma', 'invgamma', 'f']

    d = _fit_distributions(beta=beta, dists=dists, lim=lim)

    hist, _  = np.histogram(beta, bins=bins, density=True)
    edge = (_[1:] + _[:-1]) / 2

    KL = {dist: stats.entropy(hist,
            getattr(stats, dist).pdf(edge, *d[dist])
            ) for dist in dists}

    return KL
