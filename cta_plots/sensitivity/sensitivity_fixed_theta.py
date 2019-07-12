import os
import astropy.units as u
import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from colorama import Fore
from fact.analysis import li_ma_significance
from tqdm import tqdm
from cta_plots.binning import make_default_cta_binning
from cta_plots import load_signal_events, load_background_events, load_angular_resolution_function 

# from cta_plots.sensitvity import find_relative_sensitivity_poisson, find_relative_sensitivity, check_validity
from cta_plots.sensitivity import find_relative_sensitivity_poisson, check_validity, find_relative_sensitivity, calculate_n_off, calculate_n_signal
# from cta_plots.sensitvity.plotting import plot_crab_flux, plot_reference, plot_requirement, plot_sensitivity
from cta_plots.sensitivity.plotting import plot_crab_flux, plot_reference, plot_requirement, plot_sensitivity
from cta_plots.spectrum import CrabSpectrum

crab = CrabSpectrum()


def find_best_prediction_cut(
    prediction_cuts, signal_events, background_events, angular_resolution, alpha=1, silent=False
):
    rs = []
    for pc in tqdm(prediction_cuts, disable=silent):
        m = signal_events.gamma_prediction_mean >= pc
        selected_signal = signal_events[m]
        m = background_events.gamma_prediction_mean >= pc
        selected_background = background_events[m]


        theta_cut = angular_resolution(selected_signal.gamma_energy_prediction_mean)
        n_signal, n_signal_count = calculate_n_signal(signal_events, theta_cut)
        theta_cut = angular_resolution(selected_background.gamma_energy_prediction_mean)
        n_off, n_off_count, total_bkg_counts = calculate_n_off(background_events, theta_cut, alpha=alpha)

        n_on = n_signal + alpha * n_off
        n_on_count = n_signal_count + alpha * n_off_count

        relative_sensitivity = find_relative_sensitivity(n_signal, n_off, alpha=alpha)

        significance = li_ma_significance(n_on, n_off, alpha=alpha)

        # valid = check_validity(n_signal_count, n_off_count, alpha=alpha)
        valid = check_validity(n_signal, n_off, total_bkg_counts=total_bkg_counts, alpha=alpha)
        if not valid:
            significance = 0
            relative_sensitivity = np.inf

        rs.append([relative_sensitivity, significance, pc, n_on_count, n_off_count])

    relative_sensitivities = np.array([r[0] for r in rs])
    significances = np.array([r[1] for r in rs])

    if (significances == 0).all():
        return np.nan, np.nan, np.nan

    max_index = np.nanargmin(relative_sensitivities)
    best_relative_sensitivity, best_significance, best_prediction_cut, on_counts, off_counts = rs[max_index]
    return best_prediction_cut, best_significance, best_relative_sensitivity


def calc_relative_sensitivity(gammas, background, bin_edges, angular_resolution, alpha=0.2):
    prediction_cuts = np.arange(0.1, 1, 0.01)

    groups = pd.cut(gammas.gamma_energy_prediction_mean, bins=bin_edges)
    g = gammas.groupby(groups)

    groups = pd.cut(background.gamma_energy_prediction_mean, bins=bin_edges)
    b = background.groupby(groups)

    results = []

    for (_, signal_in_range), (_, background_in_range) in tqdm(zip(g, b), total=len(bin_edges) - 1, disable=False):
        best_prediction_cut, best_significance, best_relative_sensitivity = find_best_prediction_cut(
            prediction_cuts, signal_in_range, background_in_range, angular_resolution, alpha=alpha, silent=False
        )

        gammas_gammalike = signal_in_range[
            signal_in_range.gamma_prediction_mean >= best_prediction_cut
        ]

        background_gammalike = background_in_range[
            background_in_range.gamma_prediction_mean >= best_prediction_cut
        ]

        theta_cut = angular_resolution(gammas_gammalike.gamma_energy_prediction_mean)
        n_signal, n_signal_count = calculate_n_signal(gammas_gammalike, theta_cut)
        theta_cut = angular_resolution(background_gammalike.gamma_energy_prediction_mean)
        n_off, n_off_count, total_bkg_counts = calculate_n_off(background_gammalike, theta_cut, alpha=alpha)


        if np.isnan(best_significance):
            relative_sensitivity = [np.nan, np.nan, np.nan]
        else:
            relative_sensitivity = find_relative_sensitivity_poisson(n_signal, n_off, n_signal_count, n_off_count, alpha=alpha)

        m, l, h = relative_sensitivity
        d = {
            'sensitivity': m,
            'sensitivity_low': l,
            'sensitivity_high': h,
            'prediction_cut': best_prediction_cut,
            'significance': best_significance,
            'signal_counts': n_signal_count,
            'background_counts': n_off_count,
            'weighted_signal_counts': n_signal,
            'weighted_background_counts': n_off,
        }
        results.append(d)

    results_df = pd.DataFrame(results)
    results_df['theta_cut'] = angular_resolution(np.sqrt(bin_edges[:-1] * bin_edges[1:]))
    results_df['e_min'] = bin_edges[:-1]
    results_df['e_max'] = bin_edges[1:]
    return results_df


@click.command()
@click.argument('gammas_path', type=click.Path(exists=True))
@click.argument('protons_path', type=click.Path(exists=True))
@click.argument('electrons_path', type=click.Path(exists=True))
@click.argument('angular_resolution_path', type=click.Path(exists=True))
@click.option('-o', '--output', type=click.Path(exists=False))
@click.option('-t', '--t_obs', default=50)
@click.option('-c', '--color', default='xkcd:red')
@click.option('--reference/--no-reference', default=False)
@click.option('--use_e_true/--no-use_e_true', default=False)
@click.option('--correct_bias/--no-correct_bias', default=True)
@click.option('--requirement/--no-requirement', default=False)
@click.option('--flux/--no-flux', default=True)
def main(
    gammas_path,
    protons_path,
    electrons_path,
    angular_resolution_path,
    output,
    t_obs,
    color,
    reference,
    use_e_true,
    correct_bias,
    requirement,
    flux,
):
    t_obs *= u.h

    gammas, source_alt, source_az = load_signal_events(gammas_path, assumed_obs_time=t_obs)
    background = load_background_events(
        protons_path, electrons_path, source_alt, source_az, assumed_obs_time=t_obs
    )

    e_min, e_max = 0.005 * u.TeV, 350 * u.TeV
    bin_edges, bin_center, _ = make_default_cta_binning(e_min=e_min, e_max=e_max)

    alpha = 0.2

    # multiplicity = pd.read_csv(angular_resolution_path)['multiplicity'][0]
    # if multiplicity > 2:
    #     gammas = gammas.query(f'num_triggered_telescopes >= {multiplicity}').copy()
    #     background = background.query(f'num_triggered_telescopes >= {multiplicity}').copy()
    #     label = f'This Analysis. Multiplicity > {multiplicity}'
    # else:
    #     label = 'This Analysis'

    angular_resolution = load_angular_resolution_function(angular_resolution_path)
    if use_e_true:
        print(Fore.YELLOW + 'Using True Energy! Results will look richtich schäbich' + Fore.RESET)
        gammas.gamma_energy_prediction_mean = gammas.mc_energy
        background.gamma_energy_prediction_mean = background.mc_energy


    if correct_bias:
        from scipy.stats import binned_statistic
        from cta_plots import create_interpolated_function

        e_reco = gammas.gamma_energy_prediction_mean
        e_true = gammas.mc_energy
        resolution = (e_reco - e_true) / e_true

        median, _, _ = binned_statistic(e_reco, resolution, statistic=np.nanmedian, bins=bin_edges)
        energy_bias = create_interpolated_function(bin_center, median, sigma=0.5)

        e_corrected = e_reco / (energy_bias(e_reco) + 1)
        gammas.gamma_energy_prediction_mean = e_corrected

        e_reco = background.gamma_energy_prediction_mean
        e_corrected = e_reco / (energy_bias(e_reco) + 1)
        background.gamma_energy_prediction_mean = e_corrected
        
    else:
        print(Fore.YELLOW + 'Not correcting for energy bias' + Fore.RESET)

    df_sensitivity = calc_relative_sensitivity(gammas, background, bin_edges, angular_resolution, alpha=alpha)
    print(df_sensitivity)

    ax = plot_sensitivity(df_sensitivity, bin_edges, bin_center, color=color, label='wow')

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
        # df_sensitivity['multiplicity'] = multiplicity
        df_sensitivity.to_csv(n + '.csv', index=False, na_rep='NaN')
        plt.savefig(output)
    else:
        plt.show()


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()
