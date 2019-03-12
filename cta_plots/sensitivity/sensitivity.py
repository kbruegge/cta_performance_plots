import os

import astropy.units as u
import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from colorama import Fore

from tqdm import tqdm

from cta_plots import load_signal_events, load_energy_bias_function, load_background_events

from cta_plots.binning import make_default_cta_binning
from cta_plots.sensitivity.plotting import plot_crab_flux, plot_reference, plot_requirement, plot_sensitivity
from cta_plots.sensitivity import calculate_n_off, calculate_n_signal, find_cuts_for_best_sensitivity
from cta_plots.spectrum import CrabSpectrum
from cta_plots.sensitivity import find_relative_sensitivity_poisson

crab = CrabSpectrum()


def calc_relative_sensitivity(gammas, background, bin_edges, alpha=0.2):
    results = []

    theta_cuts = np.arange(0.01, 0.38, 0.01)
    prediction_cuts = np.arange(0.0, 1, 0.01)
    multiplicities = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    groups = pd.cut(gammas.gamma_energy_prediction_mean, bins=bin_edges)
    g = gammas.groupby(groups)

    groups = pd.cut(background.gamma_energy_prediction_mean, bins=bin_edges)
    b = background.groupby(groups)

    for (_, signal_in_range), (_, background_in_range) in tqdm(zip(g, b), total=len(bin_edges) - 1):
        best_sensitivity, best_prediction_cut, best_theta_cut, best_significance, best_mult = find_cuts_for_best_sensitivity(
            theta_cuts, prediction_cuts, multiplicities, signal_in_range, background_in_range, alpha=alpha
        )
        gammas_gammalike = signal_in_range[
            (signal_in_range.gamma_prediction_mean >= best_prediction_cut)
            &
            (signal_in_range.num_triggered_telescopes >= best_mult)
        ]
        
        background_gammalike = background_in_range[
            (background_in_range.gamma_prediction_mean >= best_prediction_cut)
            &
            (background_in_range.num_triggered_telescopes >= best_mult)
        ]
        n_signal, n_signal_counts = calculate_n_signal(
            gammas_gammalike, best_theta_cut,
        )
        n_off, n_off_counts = calculate_n_off(
            background_gammalike, best_theta_cut, alpha=alpha
        )

        rs = find_relative_sensitivity_poisson(n_signal, n_off, n_signal_counts, n_off_counts, alpha=alpha)

        m, l, h = rs
        d = {
            'sensitivity': m,
            'sensitivity_low': l,
            'sensitivity_high': h,
            'prediction_cut': best_prediction_cut,
            'significance': best_significance,
            'signal_counts': n_signal_counts,
            'background_counts': n_off_counts,
            'weighted_signal_counts': n_signal,
            'weighted_background_counts': n_off,
            'theta_cut': best_theta_cut,
            'multiplicity': best_mult,
        }
        results.append(d)

    results_df = pd.DataFrame(results)
    results_df['e_min'] = bin_edges[:-1]
    results_df['e_max'] = bin_edges[1:]
    return results_df


@click.command()
@click.argument('gammas_path', type=click.Path(exists=True))
@click.argument('protons_path', type=click.Path(exists=True))
@click.argument('electrons_path', type=click.Path(exists=True))
@click.option('-e', '--energy_bias_path', type=click.Path(exists=True))
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
    energy_bias_path,
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

    if energy_bias_path:
        energy_bias = load_energy_bias_function(energy_bias_path, sigma=0)
        e_reco = gammas.gamma_energy_prediction_mean
        e_corrected = -(e_reco - e_reco * energy_bias(e_reco))
        gammas.gamma_energy_prediction_mean = e_corrected

        e_reco = background.gamma_energy_prediction_mean
        e_corrected = -(e_reco - e_reco * energy_bias(e_reco))
        background.gamma_energy_prediction_mean = e_corrected
    else:
        print(Fore.YELLOW + 'Not correcting for energy bias' + Fore.RESET)

    e_min, e_max = 0.005 * u.TeV, 350 * u.TeV
    bin_edges, bin_center, _ = make_default_cta_binning(e_min=e_min, e_max=e_max)

    df_sensitivity = calc_relative_sensitivity(gammas, background, bin_edges, alpha=0.2)
    print(df_sensitivity)

    ax = plot_sensitivity(df_sensitivity, bin_edges, bin_center, color=color)

    if reference:
        plot_reference(ax)
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
        df_sensitivity.to_csv(n + '.csv', index=False, na_rep='NaN')
        plt.savefig(output)
    else:
        plt.show()


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()
