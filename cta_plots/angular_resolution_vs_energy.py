import click
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from astropy.coordinates import Angle
from scipy.stats import binned_statistic
import fact.io
from spectrum import make_energy_bins
from matplotlib.colors import PowerNorm
from astropy.coordinates.angle_utilities import angular_separation
from colors import default_cmap, main_color, main_color_complement



def calculate_distance_theta(df, source_alt=70 * u.deg, source_az=0 * u.deg):
    source_az = Angle(source_az).wrap_at(180 * u.deg)
    source_alt = Angle(source_alt)

    az = Angle(df.az_prediction.values, unit=u.rad).wrap_at(180 * u.deg)
    alt = Angle(df.alt_prediction.values, unit=u.rad)

    distance = angular_separation(source_az, source_alt, az, alt).to(u.deg)

    return distance


@click.command()
@click.argument('input_dl3_file', type=click.Path(exists=True))
@click.option('-o', '--output', type=click.Path(exists=False))
@click.option('-t', '--threshold', default=0.0)
@click.option('-m', '--multiplicity', default=2)
@click.option('--title', default=None)
@click.option('--reference/--no-reference', default=False)
@click.option('--complementary/--no-complementary', default=False)
def main(input_dl3_file, output, threshold, reference, complementary, multiplicity, title):
    columns = ['mc_alt', 'mc_az', 'mc_energy', 'az_prediction', 'alt_prediction']

    if threshold > 0:
        columns.append('gamma_prediction_mean')
    if multiplicity > 2:
        columns.append('num_triggered_telescopes')

    df = fact.io.read_data(input_dl3_file, key='array_events', columns=columns).dropna()

    if threshold > 0:
        df = df.query(f'gamma_prediction_mean > {threshold}').copy()
    if multiplicity > 2:
        df = df.query(f'num_triggered_telescopes >= {multiplicity}').copy()

    # df = df.query(f'(num_triggered_telescopes > 8)').copy()

    if complementary:
        color = main_color_complement
    else:
        color = main_color

    distance = calculate_distance_theta(df, source_alt=df.mc_alt.values * u.rad, source_az=df.mc_az.values * u.rad)

    bins, bin_center, bin_widths = make_energy_bins(e_min=0.003 * u.TeV, e_max=330 * u.TeV, bins=20)

    x = df.mc_energy.values
    y = distance

    b_50, bin_edges, binnumber = binned_statistic(x, y, statistic='median', bins=bins)

    # b_85, bin_edges, binnumber = binned_statistic(x, y, statistic=lambda y: np.percentile(y, 99), bins=bins)
    b_68, bin_edges, binnumber = binned_statistic(x, y, statistic=lambda y: np.percentile(y, 68), bins=bins)
    # b_15, bin_edges, binnumber = binned_statistic(x, y, statistic=lambda y: np.percentile(y, 32), bins=bins)

    bin_centers = np.sqrt(bin_edges[1:] * bin_edges[:-1])
    bins_y = np.logspace(np.log10(0.001), np.log10(100.8), 100)
    # h, _, _ = np.histogram2d(x, y, bins=[bins, bins_y])

    # plt.pcolormesh(bins, bins_y, h.T, cmap='Greys', norm=PowerNorm(0.25))
    log_emin, log_emax = np.log10(bins.min().value), np.log10(bins.max().value)
    log_ymin, log_ymax = np.log10(bins_y.min()), np.log10(bins_y.max())
    plt.figure(figsize=(8, 5))
    plt.hexbin(x, y, xscale='log', yscale='log', extent=(log_emin, log_emax, log_ymin, log_ymax), cmap=default_cmap, norm=PowerNorm(0.5))
    plt.colorbar()
    plt.plot(bin_centers, b_68, lw=2, color=color, label='68% Percentile')
    #plt.plot(bin_centers, b_50, '--', lw=1, color=color, label='Median')

    if reference:
        path = 'resources/CTA-Performance-prod3b-v1-South-20deg-50h-Angres.txt'
        df = pd.read_csv(path, delimiter='\t\t', skiprows=11, names=['energy', 'resolution'], engine='python')
        plt.plot(df.energy, df.resolution, '--', color='#5b5b5b', label='Prod3B Reference')

    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('Distance to True Position / degree')
    plt.xlabel(r'True Energy / TeV ')
    plt.ylim([0.001, 100.8])
    plt.legend()
    plt.tight_layout()
    if title:
        plt.title(title)
    if output:
        plt.savefig(output)
    else:
        plt.show()


if __name__ == "__main__":
    main()
