from astropy.coordinates import Angle
from astropy.coordinates.angle_utilities import angular_separation
import astropy.units as u
from fact.analysis import li_ma_significance
import numpy as np
from scipy.optimize import minimize_scalar


def find_relative_sensitivity(n_signal, n_background, t_signal, t_background, alpha=0.2, target_sigma=5, N=300):
    '''
    Given number of signal events and background events, both weighted and unweighted, calculates the 
    factor by which to scale the number of signals to reach the required detection significance. Also returns 
    lower and upper (5, 95) percentile.

    relative_sensitivity, lower, upper = find_relative_sensitivity(n_signal, n_background, t_signal, t_background):

    Parameters
    ----------
    n_signal : float
        number of signal events (gammas in on region) weighted with apropriate spectrum
    n_background : float
        number of background events weighted with apropriate spectrum
    t_signal : int
        number of signal events (gammas in on region)
    t_background : int
        number of background events
    alpha : float, optional
        exposure ration between on and off regions (the default is 0.2)
    target_sigma : int, optional
        Target detection level to reach (the default is 5)
    N : int, optional
        Number of repititions for error calculation (the default is 300)

    Returns
    -------
    np.array
        Shape 3 array with relative sensitvity and 95 percentile

    '''

    right_bound = 100

    def target(scaling_factor, n_signal, n_background, alpha=0.2, sigma=5):
        n_on = n_background * alpha + n_signal * scaling_factor
        n_off = n_background

        significance = li_ma_significance(n_on, n_off, alpha=alpha)
        return (sigma - significance) ** 2

    n_signal = np.random.poisson(t_signal, size=N) * n_signal / t_signal
    n_background = np.random.poisson(t_background, size=N) * n_background / t_background

    hs = []
    for signal, background in zip(n_signal, n_background):
        if background == 0:
            hs.append(np.nan)
        else:
            result = minimize_scalar(
                target, args=(signal, background, alpha), bounds=(0, right_bound), method='bounded'
            ).x
            if np.allclose(result, right_bound):
                result = np.nan
            hs.append(result)
    return np.nanpercentile(np.array(hs), (50, 5, 95))
