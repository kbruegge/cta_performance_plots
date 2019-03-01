import os
import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import Angle
from scipy.stats import binned_statistic
from matplotlib.colors import PowerNorm
from astropy.coordinates.angle_utilities import angular_separation
from fact.io import read_data

from cta_plots.colors import default_cmap, main_color, main_color_complement
from cta_plots.coordinate_utils import calculate_distance_to_true_source_position
from cta_plots import make_default_cta_binning
from cta_plots import load_angular_resolution_requirement, apply_cuts, load_signal_events


@click.command()
@click.argument('input_dl3_file', type=click.Path(exists=True))
@click.option('-c', '--cuts_path', type=click.Path(exists=True))
@click.option('-m', '--multiplicity', default=-1)
@click.option('-o', '--output', type=click.Path(exists=False))
@click.option('--title', default=None)
@click.option('--reference/--no-reference', default=False)
@click.option('--complementary/--no-complementary', default=False)
@click.option('--plot_e_reco', is_flag=True, default=False)
def main(input_dl3_file, cuts_path, multiplicity, output, reference, complementary, title, plot_e_reco):

    if cuts_path and multiplicity > 0:
        print('Cannot perform two sets of cuts. Supply either multiplicity or path to optimized cuts file')
        return
    cols = [
        'mc_energy',
        'mc_alt',
        'mc_az',
        'alt',
        'az',
        'gamma_prediction_mean'
    ]
    if plot_e_reco:
        cols += ['gamma_energy_prediction_mean']
    if multiplicity:
        cols += ['num_triggered_telescopes']
    if cuts_path:
        cols += ['gamma_prediction_mean']

    df, _, _ = load_signal_events(input_dl3_file, calculate_weights=False, columns=cols) 

    if cuts_path:
        df = apply_cuts(df, cuts_path, theta_cuts=False)
    elif multiplicity > 2:
        df = df[df.num_triggered_telescopes >= multiplicity]

    if complementary:
        color = main_color_complement
    else:
        color = main_color

    distance = calculate_distance_to_true_source_position(df)

    n_bins = 20
    e_min, e_max = 0.005 * u.TeV, 200 * u.TeV
    bins, bin_center, _ = make_default_cta_binning(e_min=e_min, e_max=e_max)

    if plot_e_reco:
        x = df.gamma_energy_prediction_mean.values
    else:
        x = df.mc_energy.values

    y = distance

    b_68, bin_edges, _ = binned_statistic(x, y, statistic=lambda y: np.nanpercentile(y, 68), bins=bins)

    bin_centers = np.sqrt(bin_edges[1:] * bin_edges[:-1])
    bins_y = np.logspace(np.log10(0.005), np.log10(50.8), 100)

    log_emin, log_emax = np.log10(bins.min().value), np.log10(bins.max().value)
    log_ymin, log_ymax = np.log10(bins_y.min()), np.log10(bins_y.max())
    plt.hexbin(x, y, xscale='log', yscale='log', extent=(log_emin, log_emax, log_ymin, log_ymax), cmap=default_cmap, norm=PowerNorm(0.5))
    plt.colorbar()
    plt.plot(bin_centers, b_68, lw=2, color=color, label='68% Percentile')

    if reference:
        df = load_angular_resolution_requirement()
        plt.plot(df.energy, df.resolution, '--', color='#5b5b5b', label='Prod3B Reference')

    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('Distance to True Position / degree')
    if plot_e_reco:
        plt.xlabel(r'$E_{Reco} / TeV$')
    else:
        plt.xlabel(r'$E_{True} / TeV$')

    plt.legend()
    plt.tight_layout()
    
    if title:
        plt.title(title)
    
    if output:
        plt.savefig(output)
        df = pd.DataFrame({'resolution': b_68, 'energy': bin_center, 'multiplicity': multiplicity})
        n, _ = os.path.splitext(output)
        print(f"writing csv to {n + '.csv'}")
        df.to_csv(n + '.csv', index=False, na_rep='nan')
    else:
        plt.show()


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
