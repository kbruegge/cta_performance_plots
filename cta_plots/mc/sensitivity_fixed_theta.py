import os
import astropy.units as u
import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from colorama import Fore
from fact.analysis import li_ma_significance
from tqdm import tqdm

from cta_plots import (load_sensitivity_reference,
                       load_sensitivity_requirement,
                       make_energy_bins,
                       load_angular_resolution_function,
                       load_background_events,
                       load_signal_events,
                       load_energy_bias_function,
                       )

from cta_plots.sensitivity_utils import find_relative_sensitivity
from cta_plots.mc.spectrum import CrabSpectrum

crab = CrabSpectrum()


def calculate_num_signal_events(events, angular_resolution):
    m = events.theta <= angular_resolution(events.gamma_energy_prediction_mean)
    n = events[m].weight.sum()
    counts = m.sum()
    return n, counts


def calculate_num_off_events(events, angular_resolution, alpha):
    m = events.theta <= 1.0
    theta_cut = angular_resolution(events[m].gamma_energy_prediction_mean)
    n_off = (events[m].weight * (theta_cut ** 2 / alpha)).sum()
    counts = m.sum()
    return n_off, counts


def find_best_prediction_cut(
    prediction_cuts, signal_events, background_events, angular_resolution, alpha=1, silent=False
):
    rs = []
    for pc in tqdm(prediction_cuts, disable=silent):
        m = signal_events.gamma_prediction_mean >= pc
        selected_signal = signal_events[m]
        m = background_events.gamma_prediction_mean >= pc
        selected_background = background_events[m]

        n_off, n_off_count = calculate_num_off_events(selected_background, angular_resolution, alpha)
        n_signal, n_signal_count = calculate_num_signal_events(selected_signal, angular_resolution)

        n_on = n_signal + alpha * n_off
        n_on_count = n_signal_count + alpha * n_off_count

        significance = li_ma_significance(n_on, n_off, alpha=alpha)
        if n_off_count < 50:
            print(f'not enough bakground {n_off_count, pc}')
            significance = 0
        # if n_signal_count <= 10:
        #     print('not enough signal')
        #     significance = 0

        # # must be higher than 5 times the assumed bkg systematic uncertainty of 1 percent. (See aswg irf report)
        # # https://forge.in2p3.fr/projects/cta_analysis-and-simulations/repository/changes/DOC/InternalReports/IRFReports/released/v1.1/cta-aswg-IRFreport.pdf
        # # print(pc, '---->', n_on, 5*((n_off/alpha) * 0.01))
        if n_on <= 5*((n_off/alpha) * 0.01) :
            print(f'not enough signal, pc: {pc}, on: {n_on_count}, off: {n_off_count}')
            print(f'weights pc: {pc}, on: {n_on}, off: {n_off}')
            significance = 0
        if n_on <= alpha * n_off + 10:
            print(f'not enough signal compared tp bkg pc: {pc}, on: {n_on_count}, off: {n_off_count}')
            print(f'weights pc: {pc}, on: {n_on}, off: {n_off}')
            significance = 0
        rs.append([significance, pc])

    significances = np.array([r[0] for r in rs])
    if (significances == 0).all():
        print(Fore.YELLOW + ' All significances are zero.')
        print(Fore.RESET)
        return np.nan, np.nan

    max_index = np.argmax(significances)
    best_significance, best_prediction_cut = rs[max_index]

    return best_prediction_cut, best_significance


def calc_relative_sensitivity(gammas, background, bin_edges, angular_resolution, alpha=0.2):
    relative_sensitivities = []
    thresholds = []
    significances = []

    prediction_cuts = np.arange(0.0, 1, 0.025)

    groups = pd.cut(gammas.gamma_energy_prediction_mean, bins=bin_edges)
    g = gammas.groupby(groups)

    groups = pd.cut(background.gamma_energy_prediction_mean, bins=bin_edges)
    b = background.groupby(groups)

    for (_, signal_in_range), (_, background_in_range) in tqdm(zip(g, b), total=len(bin_edges) - 1):
        best_prediction_cut, best_significance = find_best_prediction_cut(
            prediction_cuts, signal_in_range, background_in_range, angular_resolution, alpha=alpha
        )
        gammas_gammalike = signal_in_range[signal_in_range.gamma_prediction_mean >= best_prediction_cut]

        background_gammalike = background_in_range[
            background_in_range.gamma_prediction_mean >= best_prediction_cut
        ]

        n_signal, n_signal_counts = calculate_num_signal_events(gammas_gammalike, angular_resolution)
        n_off, n_off_counts = calculate_num_off_events(background_gammalike, angular_resolution, alpha)

        relative_sensitivity = find_relative_sensitivity(n_signal, n_off, n_signal_counts, n_off_counts, alpha=alpha)

        relative_sensitivities.append(relative_sensitivity)
        thresholds.append(best_prediction_cut)
        significances.append(best_significance)

    m, l, h = np.array(relative_sensitivities).T
    d = {
        'sensitivity': m,
        'sensitivity_low': l,
        'sensitivity_high': h,
        'prediction_cut': thresholds,
        'significance': significances,
        'theta_cut': angular_resolution(np.sqrt(bin_edges[:-1] * bin_edges[1:])),
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
@click.argument('angular_resolution_path', type=click.Path(exists=True))
@click.option('-e', '--energy_bias_path', type=click.Path(exists=True))
@click.option('-o', '--output', type=click.Path(exists=False))
@click.option('-t', '--t_obs', default=50)
@click.option('-c', '--color', default='xkcd:red')
@click.option('--reference/--no-reference', default=False)
@click.option('--requirement/--no-requirement', default=False)
@click.option('--flux/--no-flux', default=True)
def main(
    gammas_path,
    protons_path,
    electrons_path,
    angular_resolution_path,
    energy_bias_path,
    output,
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

    n_bins = 20
    e_min, e_max = 0.02 * u.TeV, 200 * u.TeV
    bin_edges, bin_center, _ = make_energy_bins(e_min=e_min, e_max=e_max, bins=n_bins, centering='log')
    alpha = 0.2

    multiplicity = pd.read_csv(angular_resolution_path)['multiplicity'][0]
    if multiplicity > 2:
        gammas = gammas.query(f'num_triggered_telescopes >= {multiplicity}')
        background = background.query(f'num_triggered_telescopes >= {multiplicity}')
        label = f'This Analysis. Multiplicity > {multiplicity}'
    else:
        label = 'This Analysis'

    angular_resolution = load_angular_resolution_function(angular_resolution_path)
    if energy_bias_path:
        energy_bias = load_energy_bias_function(energy_bias_path)
        e_reco = gammas.gamma_energy_prediction_mean
        e_corrected = e_reco - e_reco*energy_bias(e_reco)
        gammas.gamma_energy_prediction_mean = e_corrected

        e_reco = background.gamma_energy_prediction_mean
        e_corrected = e_reco - e_reco*energy_bias(e_reco)
        background.gamma_energy_prediction_mean = e_corrected


    df_sensitivity = calc_relative_sensitivity(gammas, background, bin_edges, angular_resolution, alpha=alpha)
    print(df_sensitivity)

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
        n, _ = os.path.splitext(output)
        print(f"writing csv to {n + '.csv'}")
        df_sensitivity['multiplicity'] = multiplicity
        df_sensitivity.to_csv(n + '.csv', index=False)
        plt.savefig(output)
    else:
        plt.show()


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()
