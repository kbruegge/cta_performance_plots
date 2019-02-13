import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import Angle
from scipy.stats import binned_statistic
from . import make_energy_bins
from . import load_angular_resolution_requirement
from matplotlib.colors import PowerNorm
from astropy.coordinates.angle_utilities import angular_separation
from .colors import default_cmap, main_color, main_color_complement
from .coordinate_utils import calculate_distance_to_true_source_position

@click.command()
@click.argument('input_dl3_file', type=click.Path(exists=True))
@click.option('-o', '--output', type=click.Path(exists=False))
@click.option('-t', '--threshold', default=0.0)
@click.option('-m', '--multiplicity', default=2)
@click.option('--title', default=None)
@click.option('--reference/--no-reference', default=False)
@click.option('--complementary/--no-complementary', default=False)
@click.option('--plot_e_reco', is_flag=True, default=False)
def main(input_dl3_file, output, threshold, reference, complementary, multiplicity, title, plot_e_reco):
    columns = ['mc_alt', 'mc_az', 'mc_energy', 'az', 'alt', 'gamma_energy_prediction_mean']

    if threshold > 0:
        columns.append('gamma_prediction_mean')
    if multiplicity > 2:
        columns.append('num_triggered_telescopes')

    df = pd.read_hdf(input_dl3_file, key='array_events', columns=columns).dropna()

    if threshold > 0:
        df = df.query(f'gamma_prediction_mean > {threshold}').copy()
    if multiplicity > 2:
        df = df.query(f'num_triggered_telescopes >= {multiplicity}').copy()


    if complementary:
        color = main_color_complement
    else:
        color = main_color

    distance = calculate_distance_to_true_source_position(df)

    bins, bin_center, bin_widths = make_energy_bins(e_min=0.01 * u.TeV, e_max=120 * u.TeV, bins=20)

    if plot_e_reco:
        x = df.gamma_energy_prediction_mean.values
    else:
        x = df.mc_energy.values

    y = distance

    b_68, bin_edges, binnumber = binned_statistic(x, y, statistic=lambda y: np.percentile(y, 68), bins=bins)

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
    else:
        plt.show()


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
