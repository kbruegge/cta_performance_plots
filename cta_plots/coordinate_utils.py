from astropy.coordinates import Angle
from astropy.coordinates.angle_utilities import angular_separation
import astropy.units as u
from cta_plots.mc.spectrum import MCSpectrum, CrabSpectrum, CosmicRaySpectrum, CTAElectronSpectrum
from fact.io import read_data
from fact.analysis import li_ma_significance
import pandas as pd
import numpy as np
from tqdm import tqdm
from colorama import Fore


ELECTRON_TYPE = 1
PROTON_TYPE = 0
MAX_THETA_SQUARE_CUT = 1

def calculate_distance_to_true_source_position(df):
    source_az = Angle(df.mc_az.values * u.deg).wrap_at(180 * u.deg)
    source_alt = Angle(df.mc_alt.values * u.deg)

    az = Angle(df.az.values*u.deg).wrap_at(180 * u.deg)
    alt = Angle(df.alt.values*u.deg)

    distance = angular_separation(source_az, source_alt, az, alt).to(u.deg)
    return distance


def calculate_distance_to_point_source(df, source_alt, source_az):
    source_az = Angle(source_az).wrap_at(180 * u.deg)
    source_alt = Angle(source_alt)

    az = Angle(df.az.values*u.deg).wrap_at(180 * u.deg)
    alt = Angle(df.alt.values*u.deg)

    distance = angular_separation(source_az, source_alt, az, alt).to(u.deg)
    return distance


def find_best_detection_significance(theta_square_cuts, prediction_cuts, signal_events, background_events, alpha=1, silent=False):

    # print(signal_events.gamma_energy_prediction_mean.mean())
    m = (signal_events.gamma_prediction_mean >= prediction_cuts.min())
    signal_events = signal_events.copy()[m]
    # print(f'Energy mean after selection: {signal_events.gamma_energy_prediction_mean.mean()}')
    m = (background_events.gamma_prediction_mean >= prediction_cuts.min())
    background_events = background_events.copy()[m]

    rs = []
    for mult in tqdm([4], disable=silent):
        for pc in tqdm(prediction_cuts, disable=silent):
            m = (signal_events.gamma_prediction_mean >= pc) & (signal_events.num_triggered_telescopes >= mult)
            selected_signal = signal_events[m]
            
            m = (background_events.gamma_prediction_mean >= pc) & (background_events.num_triggered_telescopes >= mult)
            selected_background = background_events[m]
            for tc in tqdm(theta_square_cuts, disable=silent):
                significance = calculate_significance(selected_signal, selected_background, tc, alpha=alpha)
                rs.append([significance, tc, pc, mult])
        
    # print(signal_events.gamma_energy_prediction_mean.mean())
    # if (signal_events.gamma_energy_prediction_mean.mean() > 50) and (signal_events.gamma_energy_prediction_mean.mean() < 70):
    #     from IPython import embed; embed()
    significances = np.array([r[0] for r in rs])
    if (significances == 0).all():
        print(Fore.YELLOW +  ' All significances are zero.')
        print(Fore.RESET)
    max_index = np.argmax(significances)
    best_significance, best_theta_square_cut, best_prediction_cut, best_mult = rs[max_index]

    # if True:
    #     # print()
    #     # print(f'Best prediction cut: {best_prediction_cut}')
    #     # print(f'Best theta square cut: {best_theta_square_cut}')
    #     m = (signal_events.gamma_prediction_mean >= best_prediction_cut)
    #     selected_signal = signal_events[m]
    #     m = (background_events.gamma_prediction_mean >= best_prediction_cut)
    #     selected_background = background_events[m]
    #     significance = calculate_significance(selected_signal, selected_background, best_theta_square_cut, alpha=alpha, verbose=True)



    return best_prediction_cut, best_theta_square_cut, best_significance, best_mult


def calculate_n_signal(signal_events, theta_square_cut, return_unweighted=False):
    m = signal_events.theta**2 <= theta_square_cut
    n_signal =  signal_events[m].weight.sum()
    
    if return_unweighted:
        counts = m.sum()
        return n_signal, counts
    
    return n_signal

def calculate_n_on_n_off(signal_events, background_events, theta_square_cut, alpha=1):
    n_off = calculate_n_off(background_events, theta_square_cut, alpha=alpha)
    n_signal = calculate_n_signal(signal_events, theta_square_cut)
    
    n_on = n_signal + alpha*n_off

    return n_on, n_off

def calculate_n_off(background_events, theta_square_cut, alpha=1, return_unweighted=False):
    m = background_events.theta <= 1.0
    n_off = background_events[m].weight.sum() * (theta_square_cut / alpha)
    if return_unweighted:
        counts = m.sum()
        return n_off, counts
    
    return n_off

    ########
    # m = background_events.theta <= (np.sqrt(theta_square_cut / alpha))
    # n_off =  background_events[m].weight.sum()
    
    # if return_unweighted:
    #     counts = m.sum()
    #     return n_off, counts
    
    # return n_off

    ########
    # bins = np.arange(0, 1, theta_square_cut)

    # h, _ = np.histogram(
    #     background_events['theta'] ** 2, bins=bins, weights=background_events.weight
    # )
    # n_off = h.mean()/alpha

    # if return_unweighted:
    #     h, _ = np.histogram(
    #        background_events['theta'] ** 2, bins=bins,
    #     )
    #     counts = h.mean()/alpha

    #     return n_off, counts

    # return n_off


def calculate_significance(signal_events, background_events, theta_square_cut, alpha=1, check_validity=True, verbose=False):
    is_valid = True
    n_on, n_off = calculate_n_on_n_off(signal_events, background_events, theta_square_cut, alpha=alpha)
    
    if False:
        bins = np.arange(0, 1, theta_square_cut)
        h, _ = np.histogram(background_events['theta'] ** 2, bins=bins)

        is_valid = (h == 0).sum() < len(h)//2 # less than half of the bins have to be nonzero 
        is_valid &= (h.sum() > 2 * len(h))
        # is_valid &= h.sum() > 10
        # is_valid &= n_on > n_off + 10
        if verbose:
            print(f'Is valid: {is_valid}')
            print(f'Number of zeros in histogram: {(h == 0).sum()}')
            print(f'Number bins histogram: {h.shape}')
            print(f'Number entries in histogram: {h.sum()}')
            print(f'Theta square cut: {theta_square_cut}')
            if len(h) < 10:
                print(h)
                
            # print(f'Number of zeros in histogram: {(h == 0).sum()}')
        if not is_valid:
            # print('not valid')
            return 0
    
    return li_ma_significance(n_on, n_off, alpha=alpha)
