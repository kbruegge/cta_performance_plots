import astropy.units as u
from cta_plots.mc.spectrum import CrabSpectrum

from cta_plots import make_energy_bins
from fact.analysis import li_ma_significance

import click

from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from scipy.optimize import brute
from scipy.optimize import minimize_scalar
from tqdm import tqdm
from cta_plots import load_sensitivity_reference, load_sensitivity_requirement
from cta_plots.coordinate_utils import find_best_detection_significance, calculate_n_signal, calculate_n_off
from cta_plots.coordinate_utils import load_signal_events, load_background_events

from colorama import Fore

crab = CrabSpectrum()
# cosmic_proton = CosmicRaySpectrum()
# cta_electron_spectrum = CTAElectronSpectrum()


def scaling_factor(n_signal, n_background, t_signal, t_background, alpha=1, N=200):
    right_bound = 100

    def target(scaling_factor, n_signal, n_background, alpha=0.2, sigma=5):
        n_on = n_background * alpha + n_signal * scaling_factor
        n_off = n_background

        significance = li_ma_significance(n_on, n_off, alpha=alpha)
        return (sigma - significance) ** 2

    #     print(t_background, n_background, '---------', t_signal, n_signal)
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


def calc_relative_sensitivity(gammas, background, bin_edges, t_obs=50 * u.h, alpha=0.2):
    relative_sensitivities = []
    thresholds = []
    thetas = []

    theta_square_cuts = np.arange(0.01, 0.5, 0.01)
    prediction_cuts = np.arange(0.2, 1, 0.05)

    groups = pd.cut(gammas.gamma_energy_prediction_mean, bins=bin_edges)
    g = gammas.groupby(groups)

    groups = pd.cut(background.gamma_energy_prediction_mean, bins=bin_edges)
    b = background.groupby(groups)

    for (_, signal_in_range), (_, background_in_range) in tqdm(zip(g, b), total=len(bin_edges) - 1):
        best_prediction_cut, best_theta_square_cut, best_significance = find_best_detection_significance(
            theta_square_cuts, prediction_cuts, signal_in_range, background_in_range, alpha=alpha
        )
        gammas_gammalike = signal_in_range[
            signal_in_range.gamma_prediction_mean >= best_prediction_cut
        ]

        background_gammalike = background_in_range[
            background_in_range.gamma_prediction_mean >= best_prediction_cut
        ]

        n_signal, n_signal_counts = calculate_n_signal(
            gammas_gammalike, best_theta_square_cut, return_unweighted=True
        )
        n_off, n_off_counts = calculate_n_off(
            background_gammalike, best_theta_square_cut, alpha=alpha, return_unweighted=True
        )

        rs = scaling_factor(n_signal, n_off, n_signal_counts, n_off_counts, alpha=alpha)

        relative_sensitivities.append(rs)
        thresholds.append(best_prediction_cut)
        thetas.append(np.sqrt(best_theta_square_cut))

    m, l, h = np.array(relative_sensitivities).T
    d = {
        'sensitivity': m,
        'sensitivity_low': l,
        'sensitivity_high': h,
        'threshold': thresholds,
        'theta': thetas,
        'e_min': bin_edges[:-1],
        'e_max': bin_edges[1:],
    }
    return pd.DataFrame(d)


def plot_sensitivity(rs, bin_edges, bin_center, color='blue', ax=None, **kwargs):
    sensitivity = rs.sensitivity.values * (crab.flux(bin_center) * bin_center ** 2).to(
        u.erg / (u.s * u.cm ** 2)
    )
    sensitivity_low = rs.sensitivity_low.values * (crab.flux(bin_center) * bin_center ** 2).to(
        u.erg / (u.s * u.cm ** 2)
    )
    sensitivity_high = rs.sensitivity_high.values * (crab.flux(bin_center) * bin_center ** 2).to(
        u.erg / (u.s * u.cm ** 2)
    )
    xerr = [np.abs(bin_edges[:-1] - bin_center).value, np.abs(bin_edges[1:] - bin_center).value]
    yerr = [np.abs(sensitivity - sensitivity_low).value, np.abs(sensitivity - sensitivity_high).value]

    if not ax:
        ax = plt.gca()
    ax.errorbar(
        bin_center.value, sensitivity.value, xerr=xerr, yerr=yerr, linestyle='', ecolor=color, **kwargs
    )
    return ax


def plot_crab_flux(bin_edges, ax=None):
    if not ax:
        ax = plt.gca()
    ax.plot(
        bin_edges, crab.flux(bin_edges) * bin_edges ** 2, ls=':', lw=1, color='#a3a3a3', label='Crab Flux'
    )
    return ax


def plot_requirement(ax=None):
    df = load_sensitivity_requirement()
    if not ax:
        ax = plt.gca()
    ax.plot(df.energy, df.sensitivity, color='#888888', lw=1.2, label='Requirement Offline')
    ax.plot(df.energy, df.sensitivity * 3, color='#bebebe', lw=0.5, label='Requirement Real Time')
    return ax


def plot_refrence(ax=None):
    df = load_sensitivity_reference()
    bin_edges = sorted(list(set(df.e_min) | set(df.e_max))) * u.TeV
    bin_center = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    sensitivity = df.sensitivity.values * u.erg / (u.cm ** 2 * u.s)

    if not ax:
        ax = plt.gca()

    xerr = [np.abs(bin_edges[:-1] - bin_center).value, np.abs(bin_edges[1:] - bin_center).value]
    ax.errorbar(
        bin_center.value, sensitivity.value, xerr=xerr, linestyle='', color='#3e3e3e', label='Reference'
    )
    return ax


@click.command()
@click.argument('gammas_path', type=click.Path(exists=True))
@click.argument('protons_path', type=click.Path(exists=True))
@click.argument('electrons_path', type=click.Path(exists=True))
@click.option('-o', '--output', type=click.Path(exists=False))
@click.option('-m', '--multiplicity', default=2)
@click.option('-t', '--t_obs', default=50)
@click.option('-c', '--color', default='xkcd:green')
@click.option('--reference/--no-reference', default=False)
@click.option('--requirement/--no-requirement', default=False)
@click.option('--flux/--no-flux', default=True)
def main(
    gammas_path,
    protons_path,
    electrons_path,
    output,
    multiplicity,
    t_obs,
    color,
    reference,
    requirement,
    flux,
):
    t_obs *= u.h

    gammas, source_alt, source_az = load_signal_events(gammas_path, assumed_obs_time=t_obs)
    background = load_background_events(
        protons_path, electrons_path, source_alt, source_az, assumed_obs_time=t_obs
    )

    if multiplicity > 2:
        gammas = gammas.query(f'num_triggered_telescopes >= {multiplicity}')
        background = background.query(f'num_triggered_telescopes >= {multiplicity}')
        label = f'This Analysis. Multiplicity > {multiplicity}'
    else:
        label = 'This Analysis'

    n_bins = 20
    e_min, e_max = 0.02 * u.TeV, 200 * u.TeV
    bin_edges, bin_center, _ = make_energy_bins(e_min=e_min, e_max=e_max, bins=n_bins, centering='log')

    df_sensitivity = calc_relative_sensitivity(gammas, background, bin_edges, t_obs=t_obs)

    # rs_mult_extrapolate = calc_relative_sensitivity(gammas, background, bin_edges, method=method, alpha=0.2)

    ax = plot_sensitivity(df_sensitivity, bin_edges, bin_center, color=color, label=label)

    if reference:
        plot_refrence(ax)
    if requirement:
        plot_requirement(ax)
    if flux:
        plot_crab_flux(bin_edges, ax)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim([1e-2, 10 ** (2.5)])
    ax.set_ylabel(r'$ E^2 \cdot \quad \mathrm{erg} /( \mathrm{s} \quad  \mathrm{cm}^2$)')
    ax.set_xlabel(r'$E_{Reco} /  \mathrm{TeV}$')
    ax.legend()
    plt.title('Point source sensitivity (Prod3b, HB9, Paranal) in ' + str(t_obs.to('h')))

    if output:
        plt.savefig(output)
    else:
        plt.show()


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()
