# This is the main module of superstatistics.

import warnings
import logging

import numpy as np
from scipy import stats
from scipy.optimize import root_scalar
from functools import partial

__all__ = [
    'estimate_long_time',
    'volatility',
    'find_best_distribution'
]

def estimate_long_time(timeseries: np.ndarray, lag: np.ndarray=None,
                       threshold: float=None, moment: str='kurtosis',
                       tol: float=0.05, quantiles: list=[False],
                       rtol: float=None) -> int | np.ndarray:
    r"""
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

    threshold: scalar (default `3`)
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

    quantiles: list (default `[False]`)
        If values are given, returns the quantiles, accordingly, around the
        estimated long superstatistical time (if lag is given).

    rtol: scalar (default `None`)
        Tolerance for `scipy`'s `root_scalar` method. Default is `None`, which
        runs a procedural decreasing tolerance as well as a counter check with
        various root-scalar methods.

    Returns
    -------
    T: int, np.array
        The long superstatistical time T. While using the standard root-finding
        method, only one value is returned. If `full` is `True` and a `lag` is
        given, returns the full scan over `T`. Useful for plotting `T` vs `lag`.
        Suggested to first run without a single time to locate `T` and they
        adjust `lag` accordingly.

    Notes
    -----
     - `SciPy scalar root finder <https://docs.scipy.org/doc/scipy/reference\
     /generated/scipy.optimize.root_scalar.html>`_

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
        raise ValueError("A scalar threshold must be given.")

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

        # methods to use for root_scalar
        methods = ['secant', 'bisect', 'brentq', 'brenth', 'ridder']

        # catch warning, as this use root_scalar with integers, which can
        # result in warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            # If no rtol is given, try the following, else use give
            if rtol is None:
                rtol = [1e0, 5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3]
            else:
                rtol = [rtol]

            # Store results here
            T_list = {}
            for rtol_ in rtol:
                try:
                    method = 'secant'
                    _ = root_scalar(fun, args=(np.nanmean),
                        method=method, x0=30, x1=50, rtol=rtol_)
                    if _.converged:
                        logging.info(
                            f"Converged with {method} to a "
                            f"floating long time T={_.root:.2f}"
                        )
                        T_list[method] = _
                    else:
                        logging.info(
                            f"Failed to converge with rtol of "
                            "{rtol_:.3f}"
                        )
                except:
                    logging.info(
                        f"Failed to converge with rtol of {rtol_:.3f}"
                    )
                    if rtol_ == 1.:
                        raise ValueError("The root solver did not "
                            "converge to a long time T")
                    else:
                        pass
            logging.info(
                f"Conferring result with bracketed root_scalar methods"
            )
            for method in methods[1:]:
                try:
                    T_list[method] = root_scalar(fun, args=(np.nanmean),
                        method=method, bracket=[3, T_list['secant'].root*3])
                    logging.info(
                        f"Converged with bracketed method {method} to a "
                        f"floating long time T={T_list[method].root:.2f}"
                    )
                except:
                    logging.info(
                        f"Failed to converge with bracketed method: {method}"
                    )


        # Check if all the methods that converged agree on the estimated long time
        roots_dict = {method: round(T_list[method].root) for method in methods
            if T_list[method].converged}
        roots = [x for x in roots_dict.values() if x is not None]
        if all(x==roots[0] for x in roots):
            logging.info(
                f"All converging root_scalar methods agree"
            )
            root = roots[0]
        else:
            warn_method = ["   {:<6}: {:}".format(k, roots_dict[k]) + '\n'
                for k in roots_dict.keys()]

            logging.warning(
                "Not all root_scalar methods agreed on the long time: \n" +
                "".join(warn_method) +
                "Retuning the mode (most common answer)."
            )

            root = max(set(roots), key=roots.count)

    if lag is None:
        # if (T_list['secant'].converged and T_list['secant'].root > 2 and
        #     T_list['secant'].root) < N // 2 - 1:
        if root > 2 and root < N // 2 - 1:
            return root
        else:
            raise ValueError("The root solver did not converge to a "
                             "long time T.")

    if isinstance(lag, (np.ndarray, list)):
        if quantiles[0]:
            return part_T, part_T_quantiles
        else:
            return part_T


def volatility(timeseries: np.array, T: int, bracket: list=[5, 7]) -> np.array:
    r"""
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

    bracket: list of 2 scalars (default `[5, 7]`)
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

def find_best_distribution(beta: np.array, bins=100, dists: list=None,
    lim: list=[None, None]) -> (dict, dict):
    r"""
    Estimates the Kullback–Leibler divergence between the distribution of the
    volatility beta and four standard distributions commonly used in
    superstatistics: lognormal, gamma, inverse-gamma, and the F distribution.

    Parameters
    ----------
    beta: np.array
        The (inverse) volatilities β.

    bins: int or sequence of scalars or str, optional (default `100`)
        Bins, following `numpy`'s `histogram`.

    dist: list (default are scipy's `['lognorm', 'gamma', 'invgamma', 'f']` )
        List of distributions to fit. If none given (default), the four commonly
        employed superstatistical distributions are used: a lognormal
        distribution, a Gamma (or continuous χ²) distribution, and inverse Gamma
        distribution, and an f distribution. Note that generally f distributions
        will fit better than any other due to their flexibiltiy (f
        distributions have 4 parameters, in comparison with all others, which
        only have 3 parameters).

    lim: list of 2 scalars (default `[None, None]`)
        Left and right lims to apply to the (inverse) volatilities β.

    Notes
    -----
     - `NumPy's histogram <https://numpy.org/doc/stable/reference/generated\
     /numpy.histogram.html>`_

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
        dists = ['lognorm', 'gamma', 'invgamma', 'f']

    # ensure beta is positive and apply bounds
    beta = beta[beta>0]
    if lim[0]:
        beta = beta[(beta>lim[0])]
    if lim[1]:
        beta = beta[(beta<lim[1])]

    d = _fit_distributions(beta=beta, dists=dists, lim=lim)

    hist, _  = np.histogram(beta, bins=bins, density=True)
    edge = (_[1:] + _[:-1]) / 2

    KL = {dist: stats.entropy(hist,
            getattr(stats, dist).pdf(edge, *d[dist])
            ) for dist in dists}

    return KL, d

def _fit_distributions(beta: np.array, dists: list=None,
    lim: list=[None, None], kwargs_for_scipy: dict={}) -> dict:

    if not isinstance(lim, list):
        raise ValueError("lim must be a list with a numerical lower and upper "
                         "bound. 'None' can be use for no bound.")
    if not all(isinstance(x, (float, int, type(None))) for x in lim):
        raise ValueError("lim must be a list with a numerical lower and upper "
                         "bound. `None` can be use for no bound.")

    if not dists:
        dists = ['lognorm', 'gamma', 'invgamma', 'f']

    kwds = {ele: {} for ele in dists}
    kwds.update(kwargs_for_scipy)

    dictionary_of_fits = {dist:
        getattr(stats, dist).fit(beta, **kwds[dist])
        for dist in dists}

    return dictionary_of_fits
