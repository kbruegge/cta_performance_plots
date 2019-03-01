from io import BytesIO

import astropy.units as u
import numpy as np
import pandas as pd
from colorama import Fore
from fact.io import read_data
from pkg_resources import resource_string
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

from cta_plots.coordinate_utils import calculate_distance_to_point_source

from .mc import spectrum


def load_background_reference():
    path = 'ascii/CTA-Performance-prod3b-v1-South-20deg-50h-BackgroundSqdeg.txt'
    r = resource_string('cta_plots.resources', path)
    df = pd.read_csv(
        BytesIO(r), delimiter='\t\t', skiprows=11, names=['e_min', 'e_max', 'rate'], engine='python'
    )
    return df

def load_energy_resolution_requirement(site='paranal'):
    path = 'ascii/CTA-Performance-prod3b-v1-South-20deg-50h-Eres.txt'
    r = resource_string('cta_plots.resources', path)
    df = pd.read_csv(
        BytesIO(r), delimiter='\t\t', skiprows=11, names=['energy', 'resolution'], engine='python'
    )
    return df


def load_angular_resolution_requirement(site='paranal'):
    path = 'ascii/CTA-Performance-prod3b-v1-South-20deg-50h-Angres.txt'
    r = resource_string('cta_plots.resources', path)
    df = pd.read_csv(
        BytesIO(r), delimiter='\t\t', skiprows=11, names=['energy', 'resolution'], engine='python'
    )
    return df


def load_effective_area_requirement(site='paranal'):
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


def load_angular_resolution_function(angular_resolution_path, sigma=1):
    df = pd.read_csv(angular_resolution_path)
    f = create_interpolated_function(df.energy.values, df.resolution, sigma=sigma)
    return f

def create_interpolated_function(energies, values, sigma=1):
    m  = ~np.isnan(values) # do not use nan values
    r = gaussian_filter1d(values[m], sigma=sigma)
    f = interp1d(energies[m], r, kind='cubic', bounds_error=False, fill_value='extrapolate')
    return f


def load_energy_bias_function(energy_bias_path, sigma=1):
    ''' 
    Creates a the energy bias function 
    f(e_reco) = (e_reco - e_true)/e_reco
    from the data given by the path to the csv file contianing the bias table.
    Parameters
    ----------
    energy_bias_path : str
        path to csv file created by the energy_bias script.
    sigma : int, optional
        amount of smoothing to perform
    
    Returns
    -------
    function
        function of e_reco returning the bias
    '''

    df = pd.read_csv(energy_bias_path)
    f = create_interpolated_function(df.energy_prediction, df['bias'])
    return f


def apply_cuts(df, cuts_path, sigma=1, theta_cuts=True, prediction_cuts=True, multiplicity_cuts=True):
    cuts = pd.read_csv(cuts_path)
    bin_center = np.sqrt(cuts.e_min * cuts.e_max)

    m = np.ones(len(df)).astype(np.bool)
    if theta_cuts:
        source_az = df.mc_az.values * u.deg
        source_alt = df.mc_alt.values * u.deg

        df['theta'] = (calculate_distance_to_point_source(df, source_alt=source_alt, source_az=source_az).to_value(u.deg))

        f_theta =  create_interpolated_function(bin_center, cuts.theta_cut)
        m &= df.theta < f_theta(df.gamma_energy_prediction_mean)

    if prediction_cuts: 
        f_prediction = create_interpolated_function(bin_center, cuts.prediction_cut)
        m &= df.gamma_prediction_mean >= f_prediction(df.gamma_energy_prediction_mean)

    if multiplicity_cuts:
        multiplicity = cuts.multiplicity[0]
        print('multi', multiplicity)
        m &= df.num_triggered_telescopes >= multiplicity
    
    return df[m]





def make_energy_bins(energies=None, e_min=None, e_max=None, bins=10, centering='linear'):
    if energies is not None and len(energies) >= 2:
        e_min = min(energies)
        e_max = max(energies)

    unit = e_min.unit

    low = np.log10(e_min.value)
    high = np.log10(e_max.value)
    bin_edges = np.logspace(low, high, endpoint=True, num=bins + 1) * unit

    if centering == 'log':
        bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    else:
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    bin_widths = np.diff(bin_edges)

    return bin_edges, bin_centers, bin_widths


def make_default_cta_binning(e_min=0.02 * u.TeV, e_max=200 * u.TeV, centering='log', overflow=False):

    bin_edges = np.logspace(np.log10(0.002), np.log10(2000), 31)
    idx = np.searchsorted(bin_edges, [e_min.to_value(u.TeV), e_max.to_value(u.TeV)])
    max_idx = min(idx[1]+1, len(bin_edges) - 1)
    bin_edges = bin_edges[idx[0]:max_idx]
    if overflow:
        bin_edges = np.append(bin_edges, 10000)
        bin_edges = np.append(0, bin_edges)
    
    if centering == 'log':
        bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    else:
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_widths = np.diff(bin_edges)
    
    return bin_edges * u.TeV, bin_centers * u.TeV, bin_widths * u.TeV


def make_energy_bins_per_decade(e_min, e_max, n_bins_per_decade=10, overflow=False, centering='linear'):
    bin_edges = np.logspace(-3, 3, (6 * n_bins_per_decade) + 1)
    
    idx = np.searchsorted(bin_edges, [e_min.to_value(u.TeV), e_max.to_value(u.TeV)])
    bin_edges = bin_edges[idx[0]:idx[1]]
    if overflow:
        bin_edges = np.append(bin_edges, 10000)
        bin_edges = np.append(0, bin_edges)

    bin_edges *= u.TeV

    if centering == 'log':
        bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    else:
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_widths = np.diff(bin_edges)
    
    return bin_edges, bin_centers, bin_widths


DEFAULT_COLUMNS = [
    'mc_energy',
    'gamma_prediction_mean',
    'gamma_energy_prediction_mean',
    'mc_alt',
    'mc_az',
    'alt',
    'az',
    'num_triggered_telescopes',
    'total_intensity',
]

def load_signal_events(gammas_path, assumed_obs_time=30 * u.min, columns=DEFAULT_COLUMNS, calculate_weights=True):
    crab_spectrum = spectrum.CrabSpectrum()

    gamma_runs = read_data(gammas_path, key='runs')

    if (gamma_runs.mc_diffuse.std() != 0).any():
        print(Fore.RED + f'Data given at {gammas_path} contains mix of diffuse and pointlike gammas.')
        print(Fore.RESET)
        raise ValueError
    
    is_diffuse = (gamma_runs.mc_diffuse == 1).all()
    gammas = read_data(gammas_path, key='array_events', columns=columns)
    mc_production_gamma = spectrum.MCSpectrum.from_cta_runs(gamma_runs)

    if (gamma_runs.mc_diffuse == 1).all():
        source_az = gammas.mc_az.values * u.deg
        source_alt = gammas.mc_alt.values * u.deg
    else:
        source_az = gammas.mc_az.iloc[0] * u.deg
        source_alt = gammas.mc_alt.iloc[0] * u.deg

    gammas['theta'] = (
        calculate_distance_to_point_source(gammas, source_alt=source_alt, source_az=source_az).to(u.deg).value
    )

    if calculate_weights:
        if is_diffuse:
            print(Fore.RED + f'Data given at {gammas_path} is diffuse cannot calcualte weights according to crab spectrum')
            print(Fore.RESET)
            raise ValueError

        gammas['weight'] = mc_production_gamma.reweigh_to_other_spectrum(
            crab_spectrum, gammas.mc_energy.values * u.TeV, t_assumed_obs=assumed_obs_time
        )

    return gammas, source_alt, source_az


# define these constants to identify electrons and protons in background data
ELECTRON_TYPE = 1
PROTON_TYPE = 0


def load_background_events(protons_path, electrons_path, source_alt, source_az, assumed_obs_time=30 * u.min, columns=DEFAULT_COLUMNS):
    cosmic_ray_spectrum = spectrum.CosmicRaySpectrum()
    electron_spectrum = spectrum.CTAElectronSpectrum()

    protons = read_data(protons_path, key='array_events', columns=columns)
    protons['theta'] = (
        calculate_distance_to_point_source(protons, source_alt=source_alt, source_az=source_az)
        .to(u.deg)
        .value
    )
    proton_runs = read_data(protons_path, key='runs')
    mc_production_proton = spectrum.MCSpectrum.from_cta_runs(proton_runs)
    protons['weight'] = mc_production_proton.reweigh_to_other_spectrum(
        cosmic_ray_spectrum, protons.mc_energy.values * u.TeV, t_assumed_obs=assumed_obs_time
    )
    protons['type'] = PROTON_TYPE
    electron_runs = read_data(electrons_path, key='runs')
    mc_production_electrons = spectrum.MCSpectrum.from_cta_runs(electron_runs)
    electrons = read_data(electrons_path, key='array_events', columns=columns)
    electrons['weight'] = mc_production_electrons.reweigh_to_other_spectrum(
        electron_spectrum, electrons.mc_energy.values * u.TeV, t_assumed_obs=assumed_obs_time
    )
    electrons['theta'] = (
        calculate_distance_to_point_source(electrons, source_alt=source_alt, source_az=source_az)
        .to(u.deg)
        .value
    )
    electrons['type'] = ELECTRON_TYPE
    background = pd.concat([protons, electrons], sort=False)
    return background
