from tqdm import tqdm
import numpy as np
from . import calculate_relative_sensitivity, calculate_significance
from joblib import Parallel, delayed


def _optimize_prediction_cuts(signal_events, background_events, prediction_cuts, theta_cuts, multiplicity, alpha=0.2, ):
    rs = []
    for pc in tqdm(prediction_cuts, disable=True):
        m = (signal_events.gamma_prediction_mean >= pc)
        selected_signal = signal_events[m]

        m = (background_events.gamma_prediction_mean >= pc)
        selected_background = background_events[m]
        for tc in tqdm(theta_cuts, disable=True):
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
            rs.append([relative_sensitivity, significance, tc, pc, multiplicity])

    return rs


def find_best_cuts(
    theta_cuts,
    prediction_cuts,
    multiplicities,
    signal_events,
    background_events,
    alpha=0.2,
    n_jobs=4,
    criterion='sensitivity'
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
    op = delayed(_optimize_prediction_cuts)

    frames = []
    for mult in multiplicities:
        m = (signal_events.num_triggered_telescopes >= mult)
        selected_signal = signal_events[m]

        m = (background_events.num_triggered_telescopes >= mult)
        selected_background = background_events[m]
        frames.append((selected_signal, selected_background, mult))

    rs = Parallel(n_jobs=n_jobs)(op(s, b, prediction_cuts, theta_cuts, multiplicity=m, alpha=alpha) for (s, b, m) in frames)
    rs = np.array(rs).reshape(-1, 5)

    relative_sensitivities = np.array([r[0] for r in rs])
    significances = np.array([r[1] for r in rs])
    if (significances == 0).all():
        return np.nan, np.nan, np.nan, np.nan, np.nan

    if criterion == 'sensitivity':
        max_index = np.nanargmin(relative_sensitivities)
    elif criterion == 'significance':
        max_index = np.nanargmax(significances)

    best_sensitivity, best_significance, best_theta_cut, best_prediction_cut, best_mult = rs[max_index]

    return best_sensitivity, best_prediction_cut, best_theta_cut, best_significance, best_mult
