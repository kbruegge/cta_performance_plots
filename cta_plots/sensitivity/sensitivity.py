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
from cta_plots.spectrum import CrabSpectrum
from cta_plots.sensitivity import find_relative_sensitivity_poisson

crab = CrabSpectrum()


def calc_relative_sensitivity(gammas, background, bin_edges, alpha=0.2, n_jobs=4):
    results = []

    theta_cuts = np.arange(0.01, 0.17, 0.01)
    prediction_cuts = np.arange(0.0, 1, 0.05)
    multiplicities = np.arange(2, 10)

    # theta_cuts = np.arange(0.01, 0.40, 0.1)
    # prediction_cuts = np.arange(0.4, 1, 0.1)
    # multiplicities = [4]

    groups = pd.cut(gammas.gamma_energy_prediction_mean, bins=bin_edges)
    g = gammas.groupby(groups)

    groups = pd.cut(background.gamma_energy_prediction_mean, bins=bin_edges)
    b = background.groupby(groups)

    for (_, signal_in_range), (_, background_in_range) in tqdm(zip(g, b), total=len(bin_edges) - 1):
        best_sensitivity, best_prediction_cut, best_theta_cut, best_significance, best_mult = find_best_cuts(
            theta_cuts, prediction_cuts, multiplicities, signal_in_range, background_in_range, alpha=alpha, n_jobs=n_jobs
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
        n_off, n_off_counts, total_bkg_counts = calculate_n_off(
            background_gammalike, best_theta_cut, alpha=alpha
        )
        
        if not np.isnan(best_sensitivity):
            rs = find_relative_sensitivity_poisson(n_signal, n_off, n_signal_counts, n_off_counts, alpha=alpha)
            m, l, h = rs
        else:
            m, l, h = np.nan, np.nan, np.nan

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
@click.option('-o', '--output', type=click.Path(exists=False))
@click.option('-m', '--multiplicity', default=2)
@click.option('-t', '--t_obs', default=50)
@click.option('-c', '--color', default='xkcd:purple')
@click.option('--n_jobs', default=4)
@click.option('--reference/--no-reference', default=False)
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
    reference,
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

    if correct_bias:
        from scipy.stats import binned_statistic
        from cta_plots import create_interpolated_function

        e_reco = gammas.gamma_energy_prediction_mean
        e_true = gammas.mc_energy
        resolution = (e_reco - e_true) / e_true

        median, _, _ = binned_statistic(e_reco, resolution, statistic=np.nanmedian, bins=bin_edges)
        energy_bias = create_interpolated_function(bin_center, median, sigma=8)

        e_corrected = e_reco / (energy_bias(e_reco) + 1)
        gammas.gamma_energy_prediction_mean = e_corrected

        e_reco = background.gamma_energy_prediction_mean
        e_corrected = e_reco / (energy_bias(e_reco) + 1)
        background.gamma_energy_prediction_mean = e_corrected
        
    else:
        print(Fore.YELLOW + 'Not correcting for energy bias' + Fore.RESET)

    df_sensitivity = calc_relative_sensitivity(gammas, background, bin_edges, alpha=0.2, n_jobs=n_jobs)
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
    ax.set_ylim([4.5e-14, 3E-10])

    ylabel = '$\\text{E}^2 \\frac{\\text{dN}}{\\text{dE}} / \\text{erg}\;\\text{cm}^{-2}\\text{s}^{-1}$'
    ax.set_ylabel(ylabel)
    ax.set_xlabel(r'Estimated Energy / TeV')
    
    # fix legend handles. The handle for the reference is different form a line2d handle. this makes it consostent.
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
    else:
        plt.show()



if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()
