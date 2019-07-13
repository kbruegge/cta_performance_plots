from io import BytesIO
from pkg_resources import resource_string

import pandas as pd
import numpy as np
from fact.analysis import li_ma_significance
from scipy.optimize import minimize_scalar
from tqdm import tqdm


def load_effective_area_reference(site='paranal', cuts_applied=False):
    if cuts_applied:
        path = 'ascii/CTA-Performance-prod3b-v1-South-20deg-50h-EffArea.txt'
    else:
        path = 'ascii/CTA-Performance-prod3b-v1-South-20deg-50h-EffAreaNoDirectionCut.txt'
    r = resource_string('cta_plots.resources', path)
    df = pd.read_csv(
        BytesIO(r), delimiter='\t\t', skiprows=11, names=['energy', 'effective_area'], engine='python'
    )
    return df


def load_sensitivity_reference():
    path = '/ascii/CTA-Performance-prod3b-v1-South-20deg-50h-DiffSens.txt'
    r = resource_string('cta_plots.resources', path)
    df = pd.read_csv(
        BytesIO(r), delimiter='\t\t', skiprows=10, names=['e_min', 'e_max', 'sensitivity'], engine='python'
    )
    return df


def load_sensitivity_requirement():
    path = 'sensitivity_requirement_south_50.txt'
    r = resource_string('cta_plots.resources', path)
    df = pd.read_csv(
        BytesIO(r),
        delim_whitespace=True,
        names=['log_energy', 'sensitivity'],
        index_col=False,
        engine='python',
    )
    df['energy'] = 10 ** df.log_energy
    return df



def check_validity(n_signal, n_off, total_bkg_counts, alpha=0.2, silent=True):
    n_on = n_signal + alpha * n_off


    enough_bkg_counts = total_bkg_counts >= 20  # unweighted background counts
    # if not silent:
    # enough_bkg_counts = enough_bkg_counts & (n_off >= 1)
    enough_signal_counts = n_signal >= 10
    # https://forge.in2p3.fr/projects/cta_analysis-and-simulations/repository/changes/DOC/InternalReports/IRFReports/released/v1.1/cta-aswg-IRFreport.pdf
    systematic = n_signal > (n_off * 0.05 / alpha)
    # must be higher than 5 times the assumed bkg systematic uncertainty of 1 percent. (See ASWG irf report)

    required_excess = n_on > (alpha * n_off + 10)
    if not silent:
        print(f'bkg: {enough_bkg_counts}')
        # print(f'bkg: {enough_bkg_counts}')
        print(f'signal: {enough_signal_counts}')
        print(f'sys: {systematic}')
        print(f'excess: {required_excess}')
    return enough_bkg_counts & enough_signal_counts & systematic & required_excess



def calculate_n_signal(signal_events, theta_cut):
    m = signal_events.theta <= theta_cut
    n_signal = signal_events[m].weight.sum()
    counts = m.sum()
    return n_signal, counts


def calculate_n_on_n_off(signal_events, background_events, theta_cut, alpha=0.2):
    n_off, n_off_counts, _ = calculate_n_off(background_events, theta_cut, alpha=alpha)
    n_signal, n_signal_counts = calculate_n_signal(signal_events, theta_cut)

    n_on = n_signal + alpha * n_off
    n_on_counts = n_signal_counts + alpha * n_off_counts
    return n_on, n_on_counts, n_off, n_off_counts


def calculate_n_off(background_events, theta_cut, alpha=0.2,):
    m = background_events.theta <= 1.0
    n_off = (background_events[m].weight * (theta_cut**2 / alpha)).sum()
    n_off_counts = m.sum() * (theta_cut**2 / alpha)
    total_counts = m.sum()
    return n_off, n_off_counts, total_counts


def calculate_significance(signal_events, background_events, theta_cut, alpha=0.2):
    n_on, _, n_off, _ = calculate_n_on_n_off(signal_events, background_events, theta_cut, alpha=alpha)
    return li_ma_significance(n_on, n_off, alpha=alpha)


def calculate_relative_sensitivity(signal_events, background_events, theta_cut, alpha=0.2):
    '''
    Calculates the relative sensitivity for the given signal and background. 
    Return np.inf in case check_validity() returns false for the input.
    
    Parameters
    ----------
    signal_events : pd.DataFrame
        A dataframe containing energies and weights for the signal
    background_events : pd.DataFrame
        A dataframe containing energies and weights for the background (protons + electrons)
    theta_cut : array
        signal regions to iterate over
    alpha : float, optional
        assumed ratio between signal and background region
    Returns
    -------
    float
        relative sensitivity
    '''

    n_signal, n_signal_count = calculate_n_signal(signal_events, theta_cut)
    n_off, n_off_count, total_bkg_counts = calculate_n_off(background_events, theta_cut, alpha=alpha)

    
    # valid = check_validity(n_signal_count, n_off_count, total_bkg_counts, alpha=alpha)
    valid = check_validity(n_signal, n_off, total_bkg_counts, alpha=alpha)

    if valid:
        relative_sensitivity = find_relative_sensitivity(n_signal, n_off, alpha=alpha)
        return relative_sensitivity
    else:
        return np.inf



def find_cuts_for_best_sensitivity(
    theta_cuts,
    prediction_cuts,
    multiplicities,
    signal_events,
    background_events,
    alpha=0.2,
    silent=False
):
    '''
    Find best the combination of theta_cuts, predicitons_cuts and multiplicity_cut for which 
    the relative sensitivity is the smallest. 
    
    Parameters
    ----------
    theta_cuts : array
        signal regions to iterate over
    prediction_cuts : array
        prediction cuts to iterate over
    multiplicities : array
        multiplicity cuts to iterate over
    signal_events : pd.DataFrame
        A dataframe containing energies and weights for the signal
    background_events : pd.DataFrame
        A dataframe containing energies and weights for the background (protons + electrons)
    alpha : float, optional
        assumed ratio between signal and background region
    silent : bool, optional
        whether to create a bunch of progressbars
    
    Returns
    -------
    tuple
        best_sensitivity, best_prediction_cut, best_theta_cut, best_significance, best_mult
    '''

    rs = []
    for mult in tqdm(multiplicities, disable=silent):
        for pc in tqdm(prediction_cuts, disable=silent):
            m = (signal_events.gamma_prediction_mean >= pc) & (signal_events.num_triggered_telescopes >= mult)
            selected_signal = signal_events[m]
            
            m = (background_events.gamma_prediction_mean >= pc) & (background_events.num_triggered_telescopes >= mult)
            selected_background = background_events[m]
            for tc in tqdm(theta_cuts, disable=silent):
                significance = calculate_significance(
                    selected_signal,
                    selected_background,
                    tc,
                    alpha=alpha
                )
                relative_sensitivity = calculate_relative_sensitivity(
                    selected_signal,
                    selected_background,
                    tc,
                    alpha=alpha,
                )
                rs.append([relative_sensitivity, significance, tc, pc, mult])

    relative_sensitivities = np.array([r[0] for r in rs])
    significances = np.array([r[1] for r in rs])
    if (significances == 0).all():
        return np.nan, np.nan, np.nan, np.nan, np.nan

    max_index = np.nanargmin(relative_sensitivities)
    best_sensitivity, best_significance, best_theta_cut, best_prediction_cut, best_mult = rs[max_index]

    return best_sensitivity, best_prediction_cut, best_theta_cut, best_significance, best_mult


def _target(scaling_factor, n_signal, n_background, alpha=0.2, sigma=5):
    n_on = n_background * alpha + n_signal * scaling_factor
    n_off = n_background

    significance = li_ma_significance(n_on, n_off, alpha=alpha)
    return (sigma - significance) ** 2


def find_relative_sensitivity(n_signal, n_background, alpha=0.2, target_sigma=5):
    '''
    Given number of signal events and background events calculates the 
    factor by which to scale the number of signals to reach the required detection significance. 

    Parameters
    ----------
    n_signal : float
        number of signal events (gammas in on region) weighted with apropriate spectrum
    n_background : float
        number of background events weighted with apropriate spectrum
    alpha : float, optional
        exposure ration between on and off regions (the default is 0.2)
    target_sigma : int, optional
        Target detection level to reach (the default is 5)

    Returns
    -------
    np.float
        relative sensitvity

    '''

    right_bound = 100
    result = minimize_scalar(
        _target, args=(n_signal, n_background, alpha, target_sigma), bounds=(0, right_bound), method='bounded'
    ).x
    if np.allclose(result, right_bound):
        result = np.nan
    return result


def find_relative_sensitivity_poisson(n_signal, n_background, t_signal, t_background, alpha=0.2, target_sigma=5, N=300):
    '''
    Given number of signal events and background events, both weighted and unweighted, calculates the 
    factor by which to scale the number of signals to reach the required detection significance. 
    This repeats the calculation N times by sampling n_signal and n_background according to a 
    poisson distribution.
    Returns the (50, 5, 95) percentiles.

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

    n_signal = np.random.poisson(n_signal, size=N)
    n_background = np.random.poisson(n_background, size=N) 

    hs = []
    for signal, background in zip(n_signal, n_background):
        if background == 0:
            hs.append(np.nan)
        else:
            result = minimize_scalar(
                _target, args=(signal, background, alpha, target_sigma), bounds=(0, right_bound), method='bounded'
            ).x
            if np.allclose(result, right_bound):
                result = np.nan
            hs.append(result)
    return np.nanpercentile(np.array(hs), (50, 5, 95))
