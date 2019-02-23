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
    r = gaussian_filter1d(df.resolution, sigma=sigma)
    f = interp1d(df.energy, r, kind='cubic', bounds_error=False, fill_value='extrapolate')
    return f


def load_sensitivity_reference():
    path = '/ascii/CTA-Performance-prod3b-v1-South-20deg-50h-DiffSens.txt'
    r = resource_string('cta_plots.resources', path)
    df = pd.read_csv(
        BytesIO(r), delimiter='\t\t', skiprows=10, names=['e_min', 'e_max', 'sensitivity'], engine='python'
    )
    return df


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


def load_signal_events(gammas_path, assumed_obs_time=30 * u.min):
    crab_spectrum = spectrum.CrabSpectrum()

    gamma_runs = read_data(gammas_path, key='runs')

    if (gamma_runs.mc_diffuse == 1).any():
        print(Fore.RED + 'Data given at {} contains diffuse gammas.')
        print(Fore.RED + 'Need point-like gammas to do theta square plot')
        print(Fore.RESET)
        raise ValueError

    gammas = read_data(gammas_path, key='array_events')

    mc_production_gamma = spectrum.MCSpectrum.from_cta_runs(gamma_runs)

    source_az = gammas.mc_az.iloc[0] * u.deg
    source_alt = gammas.mc_alt.iloc[0] * u.deg

    gammas['theta'] = (
        calculate_distance_to_point_source(gammas, source_alt=source_alt, source_az=source_az).to(u.deg).value
    )

    gammas['weight'] = mc_production_gamma.reweigh_to_other_spectrum(
        crab_spectrum, gammas.mc_energy.values * u.TeV, t_assumed_obs=assumed_obs_time
    )

    return gammas, source_alt, source_az


# define these constants to identify electrons and protons in background data
ELECTRON_TYPE = 1
PROTON_TYPE = 0


def load_background_events(protons_path, electrons_path, source_alt, source_az, assumed_obs_time=30 * u.min):
    cosmic_ray_spectrum = spectrum.CosmicRaySpectrum()
    electron_spectrum = spectrum.CTAElectronSpectrum()

    protons = read_data(protons_path, key='array_events')
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
    electrons = read_data(electrons_path, key='array_events')
    electrons['weight'] = mc_production_electrons.reweigh_to_other_spectrum(
        electron_spectrum, electrons.mc_energy.values * u.TeV, t_assumed_obs=assumed_obs_time
    )
    electrons['theta'] = (
        calculate_distance_to_point_source(electrons, source_alt=source_alt, source_az=source_az)
        .to(u.deg)
        .value
    )
    electrons['type'] = ELECTRON_TYPE

    return pd.concat([protons, electrons], sort=False)
