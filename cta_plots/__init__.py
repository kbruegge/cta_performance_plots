import astropy.units as u
import numpy as np
import pandas as pd
from colorama import Fore
from fact.io import read_data
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

from mpl_toolkits.axes_grid1 import make_axes_locatable


from cta_plots.coordinate_utils import calculate_distance_to_point_source

from . import spectrum

# define these constants to identify electrons and protons in background data
ELECTRON_TYPE = 1
PROTON_TYPE = 0


def load_angular_resolution_function(angular_resolution_path, sigma=1):
    df = pd.read_csv(angular_resolution_path)
    f = create_interpolated_function(df.energy.values, df.resolution, sigma=sigma)
    return f


def create_interpolated_function(energies, values, sigma=1):
    m = ~np.isnan(values)  # do not use nan values
    if sigma > 0:
        r = gaussian_filter1d(values[m], sigma=sigma)
    else:
        r = values[m]
    f = interp1d(energies[m], r, kind='linear', bounds_error=False, fill_value='extrapolate')
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
    f = create_interpolated_function(df.energy_prediction, df['bias'], sigma=sigma)
    return f


def apply_cuts(df, cuts_path, sigma=1, theta_cuts=True, prediction_cuts=True, multiplicity_cuts=True):
    cuts = pd.read_csv(cuts_path)
    bin_center = np.sqrt(cuts.e_min * cuts.e_max)

    m = np.ones(len(df)).astype(np.bool)
    if theta_cuts:
        source_az = df.mc_az.values * u.deg
        source_alt = df.mc_alt.values * u.deg

        df['theta'] = (calculate_distance_to_point_source(df, source_alt=source_alt, source_az=source_az).to_value(u.deg))

        f_theta = create_interpolated_function(bin_center, cuts.theta_cut)
        m &= df.theta < f_theta(df.gamma_energy_prediction_mean)

    if prediction_cuts: 
        f_prediction = create_interpolated_function(bin_center, cuts.prediction_cut)
        m &= df.gamma_prediction_mean >= f_prediction(df.gamma_energy_prediction_mean)

    if multiplicity_cuts:
        x0 = cuts.multiplicity.iloc[0]
        x1 = cuts.multiplicity.iloc[-1]
        f_mult = interp1d(cuts.e_min, cuts.multiplicity, kind='previous', bounds_error=False, fill_value=(x0, x1))
        m &= df.num_triggered_telescopes >= f_mult(df.gamma_energy_prediction_mean)
    
    return df[m]


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
    # crab_spectrum = spectrum.CrabLogParabola()

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
            print(Fore.RED + f'Data given at {gammas_path} is diffuse. Cannot calcualte weights according to crab spectrum which is pointlike')
            print(Fore.RESET)
            raise ValueError

        gammas['weight'] = mc_production_gamma.reweigh_to_other_spectrum(
            crab_spectrum, gammas.mc_energy.values * u.TeV, t_assumed_obs=assumed_obs_time
        )

    return gammas, source_alt, source_az


def load_background_events(protons_path, electrons_path, source_alt, source_az, assumed_obs_time=50 * u.h, columns=DEFAULT_COLUMNS):
    cosmic_ray_spectrum = spectrum.CosmicRaySpectrum()
    electron_spectrum = spectrum.CTAElectronSpectrum()

    protons = read_data(protons_path, key='array_events', columns=columns)
    proton_runs = read_data(protons_path, key='runs')
    
    mc_production_proton = spectrum.MCSpectrum.from_cta_runs(proton_runs)
    protons['weight'] = mc_production_proton.reweigh_to_other_spectrum(
        cosmic_ray_spectrum, protons.mc_energy.values * u.TeV, t_assumed_obs=assumed_obs_time
    )
    protons['theta'] = (
        calculate_distance_to_point_source(protons, source_alt=source_alt, source_az=source_az)
        .to(u.deg)
        .value
    )
    protons['type'] = PROTON_TYPE
    
    electrons = read_data(electrons_path, key='array_events', columns=columns)
    electron_runs = read_data(electrons_path, key='runs')
    
    mc_production_electrons = spectrum.MCSpectrum.from_cta_runs(electron_runs)
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


def add_colorbar_to_figure(im, fig, ax, label=None):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical', label=label)
