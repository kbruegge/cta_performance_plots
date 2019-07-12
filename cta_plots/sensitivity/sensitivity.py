import os

import astropy.units as u
import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 500)
from colorama import Fore

from tqdm import tqdm

from cta_plots import load_signal_events, load_background_events

from cta_plots.binning import make_default_cta_binning
from cta_plots.sensitivity.plotting import plot_crab_flux, plot_reference, plot_requirement, plot_sensitivity
from cta_plots.sensitivity import calculate_n_off, calculate_n_signal
from cta_plots.sensitivity.optimize import find_best_cuts
from cta_plots.coordinate_utils import calculate_distance_to_true_source_position

from cta_plots.spectrum import CrabSpectrum
from cta_plots.sensitivity import find_relative_sensitivity_poisson, check_validity

from scipy.ndimage import gaussian_filter1d

crab = CrabSpectrum()


def optimize_event_selection_fixed_theta(gammas, background, bin_edges, alpha=0.2, n_jobs=4):
    results = []


    groups = pd.cut(gammas.gamma_energy_prediction_mean, bins=bin_edges)
    g = gammas.groupby(groups)

    groups = pd.cut(background.gamma_energy_prediction_mean, bins=bin_edges)
    b = background.groupby(groups)

    for (_, signal_in_range), (_, background_in_range) in tqdm(zip(g, b), total=len(bin_edges) - 1):
        distance = calculate_distance_to_true_source_position(signal_in_range)
        theta_cuts = np.array([np.nanpercentile(distance, 50)])
        best_sensitivity, best_prediction_cut, best_theta_cut, best_significance, best_mult = find_best_cuts(
            theta_cuts, PREDICTION_CUTS, MULTIPLICITIES, signal_in_range, background_in_range, alpha=alpha, n_jobs=n_jobs
        )

        d = {
            'prediction_cut': best_prediction_cut,
            'significance': best_significance,
            'theta_cut': best_theta_cut,
            'multiplicity': best_mult,
        }
        results.append(d)

    results_df = pd.DataFrame(results)
    results_df['e_min'] = bin_edges[:-1]
    results_df['e_max'] = bin_edges[1:]
    return results_df


def optimize_event_selection(gammas, background, bin_edges, alpha=0.2, n_jobs=4):
    results = []

    # theta_cuts = np.arange(0.01, 0.18, 0.01)
    # prediction_cuts = np.arange(0.0, 1.05, 0.05)
    # multiplicities = np.arange(2, 10)


    groups = pd.cut(gammas.gamma_energy_prediction_mean, bins=bin_edges)
    g = gammas.groupby(groups)

    groups = pd.cut(background.gamma_energy_prediction_mean, bins=bin_edges)
    b = background.groupby(groups)

    for (_, signal_in_range), (_, background_in_range) in tqdm(zip(g, b), total=len(bin_edges) - 1):
        best_sensitivity, best_prediction_cut, best_theta_cut, best_significance, best_mult = find_best_cuts(
            THETA_CUTS, PREDICTION_CUTS, MULTIPLICITIES, signal_in_range, background_in_range, alpha=alpha, n_jobs=n_jobs
        )

        d = {
            'prediction_cut': best_prediction_cut,
            'significance': best_significance,
            'theta_cut': best_theta_cut,
            'multiplicity': best_mult,
        }
        results.append(d)

    results_df = pd.DataFrame(results)
    results_df['e_min'] = bin_edges[:-1]
    results_df['e_max'] = bin_edges[1:]
    return results_df


def calc_relative_sensitivity(gammas, background, cuts, alpha, sigma=0):
    bin_edges = list(cuts['e_min']) + [cuts['e_max'].iloc[-1]]

    results = []

    if sigma > 0:
        cuts.prediction_cut = gaussian_filter1d(cuts.prediction_cut, sigma=sigma)
        cuts.theta_cut = gaussian_filter1d(cuts.theta_cut, sigma=sigma)
        cuts.multiplicity = gaussian_filter1d(cuts.multiplicity, sigma=sigma)

    groups = pd.cut(gammas.gamma_energy_prediction_mean, bins=bin_edges)
    g = gammas.groupby(groups)

    groups = pd.cut(background.gamma_energy_prediction_mean, bins=bin_edges)
    b = background.groupby(groups)

    for (_, signal_in_range), (_, background_in_range), (_, r) in tqdm(zip(g, b, cuts.iterrows()), total=len(bin_edges) - 1):
        best_mult = r.multiplicity
        best_prediction_cut = r.prediction_cut
        best_theta_cut = r.theta_cut
        best_significance = r.significance

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
        n_off, n_off_counts, total_bkg_counts = calculate_n_off(
            background_gammalike, best_theta_cut, alpha=alpha
        )

        # print('----------------')
        # valid = check_validity(n_signal_counts, n_off_counts, total_bkg_counts, alpha=alpha, silent=True)
        # print('----------------')
        valid = check_validity(n_signal, n_off, total_bkg_counts, alpha=alpha, silent=False)
        # print('----------------')
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
            'total_bkg_counts': total_bkg_counts,
            'valid': valid,
        }
        results.append(d)

    results_df = pd.DataFrame(results)
    results_df['e_min'] = bin_edges[:-1]
    results_df['e_max'] = bin_edges[1:]
    return results_df


THETA_CUTS = np.arange(0.01, 0.30, 0.01)
PREDICTION_CUTS = np.arange(0.0, 1.05, 0.05)
MULTIPLICITIES = np.arange(2, 12)


@click.command()
@click.argument('gammas_path', type=click.Path(exists=True))
@click.argument('protons_path', type=click.Path(exists=True))
@click.argument('electrons_path', type=click.Path(exists=True))
@click.option('-o', '--output', type=click.Path(exists=False))
@click.option('-m', '--multiplicity', default=2)
@click.option('-t', '--t_obs', default=50)
@click.option('-c', '--color', default='xkcd:purple')
@click.option('--n_jobs', default=4)
@click.option('--landscape/--no-landscape', default=False)
@click.option('--reference/--no-reference', default=False)
@click.option('--fix_theta/--no-fix_theta', default=False)
@click.option('--correct_bias/--no-correct_bias', default=True)
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
    n_jobs,
    landscape,
    reference,
    fix_theta,
    correct_bias,
    requirement,
    flux,
):
    t_obs *= u.h

    gammas, source_alt, source_az = load_signal_events(gammas_path, assumed_obs_time=t_obs)
    background = load_background_events(
        protons_path, electrons_path, source_alt, source_az, assumed_obs_time=t_obs
    )

    e_min, e_max = 0.02 * u.TeV, 200 * u.TeV
    bin_edges, bin_center, _ = make_default_cta_binning(e_min=e_min, e_max=e_max)
    SIGMA = 1
    if correct_bias:
        from scipy.stats import binned_statistic
        from cta_plots import create_interpolated_function

        e_reco = gammas.gamma_energy_prediction_mean
        e_true = gammas.mc_energy
        resolution = (e_reco - e_true) / e_true

        median, _, _ = binned_statistic(e_reco, resolution, statistic=np.nanmedian, bins=bin_edges)
        energy_bias = create_interpolated_function(bin_center, median, sigma=SIGMA)

        e_corrected = e_reco / (energy_bias(e_reco) + 1)
        gammas.gamma_energy_prediction_mean = e_corrected

        e_reco = background.gamma_energy_prediction_mean
        e_corrected = e_reco / (energy_bias(e_reco) + 1)
        background.gamma_energy_prediction_mean = e_corrected
        
    else:
        print(Fore.YELLOW + 'Not correcting for energy bias' + Fore.RESET)

    if fix_theta:
        print('Not optimizing theta!')
        df_cuts = optimize_event_selection_fixed_theta(gammas, background, bin_edges, alpha=0.2, n_jobs=n_jobs)
    else:
        df_cuts = optimize_event_selection(gammas, background, bin_edges, alpha=0.2, n_jobs=n_jobs)
    
    df_sensitivity = calc_relative_sensitivity(gammas, background, df_cuts, alpha=0.2, sigma=SIGMA)

    print(df_sensitivity)
    if landscape:
        size = plt.gcf().get_size_inches()
        plt.figure(figsize=(8.24, size[0] * 0.9))
    
    ax = plot_sensitivity(df_sensitivity, bin_edges, bin_center, color=color, lw=2)

    if reference:
        plot_reference(ax)
    if requirement:
        plot_requirement(ax)
    if flux:
        plot_crab_flux(bin_edges, ax)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim([1e-2, 10 ** (2.5)])
    ax.set_ylim([4e-14, 3E-10])

    ylabel = '$\\text{E}^2 \\frac{\\text{dN}}{\\text{dE}} / \\text{erg}\;\\text{cm}^{-2}\\text{s}^{-1}$'
    ax.set_ylabel(ylabel)
    ax.set_xlabel(r'Estimated Energy / TeV')
    
    # fix legend handles. The handle for the reference is different form a line2d handle. this makes it consistent.
    from matplotlib.lines import Line2D
    handles = ax.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1]
    handles[2] = Line2D([0], [0], color=handles[2].lines[0].get_color())  
    legend = ax.legend(handles, labels, framealpha=0, borderaxespad=0.025)

    # add meta information to legend title
    legend.set_title('CTA Prod3B (Paranal HB9)')
    legend._legend_box.align = "left"
    legend.get_title().set_alpha(0.5)
    # legend.get_title().set_linespacing(1.1)


    plt.tight_layout(pad=0, rect=(0, 0, 1, 1))
    if output:
        n, _ = os.path.splitext(output)
        print(f"writing csv to {n + '.csv'}")
        df_sensitivity.to_csv(n + '.csv', index=False, na_rep='NaN')
        plt.savefig(output)

        with open(f'{n}_theta_cuts.txt', 'w') as f:
            f.write(cuts_to_latex(THETA_CUTS))

        with open(f'{n}_prediction_cuts.txt', 'w') as f:
            f.write(cuts_to_latex(PREDICTION_CUTS))

        with open(f'{n}_multiplicities.txt', 'w') as f:
            f.write(cuts_to_latex(MULTIPLICITIES))
    else:
        plt.show()


def cuts_to_latex(array):
    s = f'\{{ {array[0]}, {array[1]}, \\ldots, {array[-1]} \}} '
    return s


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()
