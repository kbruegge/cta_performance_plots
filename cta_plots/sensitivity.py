import astropy.units as u
from .spectrum import CosmicRaySpectrum, CrabSpectrum, MCSpectrum
from . import make_energy_bins
import fact.io
import click

from astropy.coordinates import Angle
from astropy.coordinates.angle_utilities import angular_separation

from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fact.analysis import li_ma_significance
from scipy.optimize import brute
from scipy.optimize import minimize_scalar
from tqdm import tqdm
from . import load_sensitivity_reference, load_sensitivity_requirement


crab = CrabSpectrum()
cosmic_proton = CosmicRaySpectrum()


def count_events_in_region(df, theta2=0.03, prediction_threshold=0.5):
    m = ((df.theta**2 <= theta2) & (df.gamma_prediction_mean >= prediction_threshold))
    return df[m].weight.sum(), m.sum()


def extrapolate_off_events(df, theta2=0.03, prediction_threshold=0.5, sigma=1):
    if prediction_threshold > 1:
        return 0
    df = df.query('theta < 1.0')

    c_bins = np.linspace(0, 1, 30)
    c_bin_center = (c_bins[0:-1] + c_bins[1:]) / 2
    c_bin_width = np.diff(c_bins)
    nodes = c_bin_center + c_bin_width

    h, _ = np.histogram(df.gamma_prediction_mean, bins=c_bins, weights=df.weight)
    h = gaussian_filter(h, sigma=sigma)
    h = np.sum(h) - np.cumsum(h)
    background = interp1d(nodes, h, kind=1, fill_value='extrapolate', bounds_error=False)


    h, _ = np.histogram(df.gamma_prediction_mean, bins=c_bins)
    h = gaussian_filter(h, sigma=sigma)
    h = np.sum(h) - np.cumsum(h)
    counts = interp1d(nodes, h, kind=1, fill_value='extrapolate', bounds_error=False)

    return background(prediction_threshold) * theta2, counts(prediction_threshold) * theta2



def count_off_events_in_region(df, theta2=0.03, prediction_threshold=0.5):
    df = df.query('theta < 1.0')
    m = df.gamma_prediction_mean >= prediction_threshold
    return df[m].weight.sum() * theta2, m.sum() * theta2


def select_events_in_energy_range(signal, background, e_low, e_high, use_true_energy=False):

    column = 'mc_energy' if use_true_energy else 'gamma_energy_prediction_mean'
    m = ((signal[column] > e_low) & (signal[column] < e_high))
    s = signal[m]

    m = ((background[column] > e_low) & (background[column] < e_high))
    b = background[m]
    return s, b


def scaling_factor(n_signal, n_background, t_signal, t_background, alpha=1, N=200):
    right_bound = 100

    def target(scaling_factor, n_signal, n_background, alpha=1, sigma=5):
        n_on = n_background * alpha + n_signal * scaling_factor
        n_off = n_background

        significance = li_ma_significance(n_on, n_off, alpha=alpha)
        return (5 - significance)**2

#     print(t_background, n_background, '---------', t_signal, n_signal)
    n_signal = np.random.poisson(t_signal, size=N) * n_signal / t_signal
    n_background = np.random.poisson(t_background, size=N) * n_background / t_background


    hs = []
    for signal, background in zip(n_signal, n_background):
        if background == 0:
            hs.append(np.nan)
        else:
            result = minimize_scalar(target, args=(signal, background, alpha), bounds=(0, right_bound), method='bounded').x
            if np.allclose(result, right_bound):
                result = np.nan
            hs.append(result)
    return np.nanpercentile(np.array(hs), (50, 5, 95))


def find_best_cuts(signal, background, alpha, regions=slice(0.0025, 0.08, 0.01), thresholds=slice(0.4, 1, 0.05), method='simple'):

    def significance_target(cuts, signal, background, alpha):
        theta2, p_cut = cuts
        n_signal, t_signal = count_events_in_region(signal, theta2=theta2, prediction_threshold=p_cut)

        if method == 'exact':
            n_background, t_background = count_events_in_region(background, theta2=theta2 / alpha, prediction_threshold=p_cut)

            if t_background < 10:
                print(f'{cuts} not enough background')
                return 0

        elif method == 'simple':
            n_background, t_background = count_off_events_in_region(background, theta2=theta2 / alpha, prediction_threshold=p_cut)

            if t_background / alpha < 10:
                print(f'{cuts} not enough background')
                return 0

        elif method == 'extrapolate':
            n_background, t_background = extrapolate_off_events(background, theta2=theta2/alpha, prediction_threshold=p_cut)


#         if t_background/alpha < 1:
#             print(f'{cuts} not enough background')
#             return 0

        if t_signal <= t_background * alpha + 10:
            print('counts not large enough')
            return 0


        if t_signal <= t_background * alpha + 10:
            print('signal not large enough')
            return 0
        if n_signal*5 < n_background * 0.01:
            print('sys problem')
            return 0


        n_on = n_signal + alpha * n_background
        n_off = n_background
        return -li_ma_significance(n_on, n_off, alpha=alpha)

    result = brute(significance_target, ranges=[regions, thresholds], args=(signal, background, alpha), finish=None)
    print(result)
    return result


def calc_relative_sensitivity(signal, background, bin_edges, alpha=1, use_true_energy=False, method='simple'):
    relative_sensitivities = []
    thresholds = []
    thetas = []
    for e_low, e_high in tqdm(zip(bin_edges[:-1], bin_edges[1:])):
        s, b = select_events_in_energy_range(signal, background, e_low, e_high, use_true_energy=use_true_energy)

        theta2, cut = find_best_cuts(s, b, alpha=alpha, method=method)

        n_signal, t_signal = count_events_in_region(s, theta2=theta2, prediction_threshold=cut)

        if method == 'simple':
            n_background, t_background = count_off_events_in_region(b, theta2=theta2 / alpha, prediction_threshold=cut)
#             extra = extrapolate_off_events(b, theta2=theta2/alpha, prediction_threshold=cut)
#             print(f'Extrapolated: {extra}, Simple{(n_background, t_background)}')
        elif method == 'exact':
            n_background, t_background = count_events_in_region(b, theta2=theta2 / alpha, prediction_threshold=cut)
        elif method == 'extrapolate':
            n_background, t_background = extrapolate_off_events(b, theta2=theta2 / alpha, prediction_threshold=cut)

        print(t_background, t_signal)
        rs = scaling_factor(n_signal, n_background, t_signal, t_background, alpha=alpha)
        relative_sensitivities.append(rs)
        thresholds.append(cut)
        thetas.append(np.sqrt(theta2))



    m, l, h = np.array(relative_sensitivities).T
    d = {'sensitivity': m, 'sensitivity_low': l, 'sensitivity_high': h, 'threshold':thresholds, 'theta':thetas, 'e_min': bin_edges[:-1], 'e_max': bin_edges[1:]}
    return pd.DataFrame(d)


def calculate_theta(df, source_alt=70 * u.deg, source_az=0 * u.deg):
    source_az = Angle(source_az).wrap_at(180 * u.deg)
    source_alt = Angle(source_alt)

    az = Angle(df.az_prediction.values, unit=u.rad).wrap_at(180*u.deg)
    alt = Angle(df.alt_prediction.values, unit=u.rad)

    return angular_separation(source_az, source_alt, az, alt).to(u.deg).value


def load_data(gamma_input, proton_input, t_obs=50 * u.h):
    columns = ['gamma_prediction_mean', 'gamma_energy_prediction_mean', 'az_prediction', 'alt_prediction', 'mc_alt', 'mc_az', 'mc_energy', 'num_triggered_telescopes']

    gammas = fact.io.read_data(gamma_input, key='array_events', columns=columns)
    gammas = gammas.dropna()
    gamma_runs = fact.io.read_data(gamma_input, key='runs')
    mc_production_gamma = MCSpectrum.from_cta_runs(gamma_runs)

    protons = fact.io.read_data(proton_input, key='array_events', columns=columns)
    protons = protons.dropna()
    proton_runs = fact.io.read_data(proton_input, key='runs')
    mc_production_proton = MCSpectrum.from_cta_runs(proton_runs)

    gammas['weight'] = mc_production_gamma.reweigh_to_other_spectrum(crab, gammas.mc_energy.values * u.TeV, t_assumed_obs=t_obs)
    protons['weight'] = mc_production_proton.reweigh_to_other_spectrum(cosmic_proton, protons.mc_energy.values * u.TeV, t_assumed_obs=t_obs)

    return gammas, protons


def plot_sensitivity(rs, bin_edges, bin_center, color='blue', ax=None, **kwargs):
    sensitivity = rs.sensitivity.values * (crab.flux(bin_center) * bin_center**2).to(u.erg / (u.s * u.cm**2))
    sensitivity_low = rs.sensitivity_low.values * (crab.flux(bin_center) * bin_center**2).to(u.erg / (u.s * u.cm**2))
    sensitivity_high = rs.sensitivity_high.values * (crab.flux(bin_center) * bin_center**2).to(u.erg / (u.s * u.cm**2))
    xerr = [np.abs(bin_edges[:-1] - bin_center).value, np.abs(bin_edges[1:] - bin_center).value]
    yerr = [np.abs(sensitivity - sensitivity_low).value, np.abs(sensitivity - sensitivity_high).value]

    if not ax:
        ax = plt.gca()
    ax.errorbar(bin_center.value, sensitivity.value, xerr=xerr, yerr=yerr, linestyle='', ecolor=color, **kwargs)
    return ax


def plot_crab_flux(bin_edges, ax=None):
    if not ax:
        ax = plt.gca()
    ax.plot(bin_edges, crab.flux(bin_edges) * bin_edges**2, ls=':', color='#a3a3a3', label='Crab Flux')
    return ax


def plot_requirement(ax=None):
    df = load_sensitivity_requirement()
    if not ax:
        ax = plt.gca()
    ax.plot(df.energy, df.sensitivity, color='#888888', lw=1, label='Requirement')
    ax.plot(df.energy, df.sensitivity * 3, color='#888888', lw=0.5)
    return ax


def plot_refrence(ax=None):
    df = load_sensitivity_reference()
    bin_edges = sorted(list(set(df.e_min) | set(df.e_max))) * u.TeV
    bin_center = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    sensitivity = df.sensitivity.values * u.erg / (u.cm**2 * u.s)

    if not ax:
        ax = plt.gca()

    xerr = [np.abs(bin_edges[:-1] - bin_center).value, np.abs(bin_edges[1:] - bin_center).value]
    ax.errorbar(bin_center.value, sensitivity.value, xerr=xerr, linestyle='', color='#3e3e3e', label='Reference')
    return ax


@click.command()
@click.argument('gamma_input', type=click.Path(exists=True))
@click.argument('proton_input', type=click.Path(exists=True))
@click.option('-o', '--output', type=click.Path(exists=False))
@click.option('-m', '--multiplicity', default=2)
@click.option('-t', '--t_obs', default=50)
@click.option('-c', '--color', default='xkcd:green')
@click.option('--reference/--no-reference', default=False)
@click.option('--requirement/--no-requirement', default=False)
@click.option('--flux/--no-flux', default=True)
def main(gamma_input, proton_input, output, multiplicity, t_obs, color, reference, requirement, flux):
    t_obs *= u.h

    n_bins = 20
    e_min, e_max = 0.02 * u.TeV, 200 * u.TeV
    bin_edges, bin_center, bin_width = make_energy_bins(e_min=e_min, e_max=e_max, bins=n_bins, centering='log')

    gammas, protons = load_data(gamma_input, proton_input, t_obs=t_obs)
    gammas['theta'] = calculate_theta(gammas)
    protons['theta'] = calculate_theta(protons)

    if multiplicity > 2:
        gammas = gammas.query(f'num_triggered_telescopes >= {multiplicity}')
        protons = protons.query(f'num_triggered_telescopes >= {multiplicity}')
        label = f'This Analysis. Multiplicity > {multiplicity}'
    else:
        label = 'This Analysis'

    rs_mult_extrapolate = calc_relative_sensitivity(gammas, protons, bin_edges, method='extrapolate', alpha=0.2)
    # rs_mult_extrapolate.to_csv('sensi.csv', index=False)

    ax = plot_sensitivity(rs_mult_extrapolate, bin_edges, bin_center, color=color, label=label)

    if reference:
        plot_refrence(ax)
    if requirement:
        plot_requirement(ax)
    if flux:
        plot_crab_flux(bin_edges, ax)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim([1E-2, 10**(2.5)])
    ax.set_ylabel(r'$ E^2 \cdot \quad \mathrm{erg} /( \mathrm{s} \quad  \mathrm{cm}^2$ )  in ' + str(t_obs.to('h')))
    ax.set_xlabel(r'$E /  \mathrm{TeV}$')
    ax.legend()

    if output:
        plt.savefig(output)
    else:
        plt.show()


if __name__ == '__main__':
    main()
