import click
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from astropy.coordinates import Angle
from scipy.stats import binned_statistic
import fact.io
from spectrum import make_energy_bins
import os
from astropy.coordinates.angle_utilities import angular_separation



def calculate_distance_theta(df, source_alt=70 * u.deg, source_az=0 * u.deg):
    source_az = Angle(source_az).wrap_at(180 * u.deg)
    source_alt = Angle(source_alt)

    az = Angle(df.az_prediction.values, unit=u.rad).wrap_at(180 * u.deg)
    alt = Angle(df.alt_prediction.values, unit=u.rad)

    distance = angular_separation(source_az, source_alt, az, alt).to(u.deg)

    return distance


@click.command()
@click.argument('input_files', nargs=-1, type=click.Path(exists=True))
@click.option('-o', '--output', type=click.Path(exists=False))
@click.option('-t', '--threshold', default=0.0)
@click.option('-m', '--multiplicity', default=2)
@click.option('--reference/--no-reference', default=False)
@click.option('--complementary/--no-complementary', default=False)
def main(input_files, output, threshold, multiplicity, reference, complementary):
    columns = ['mc_alt', 'mc_az', 'mc_energy', 'az_prediction', 'alt_prediction', 'num_triggered_telescopes']

    if threshold > 0:
        columns.append('gamma_prediction_mean')

    bins, bin_center, bin_widths = make_energy_bins(e_min=0.003 * u.TeV, e_max=330 * u.TeV, bins=15)

    for input_file in input_files:
        df = fact.io.read_data(input_file, key='array_events', columns=columns).dropna()
        print(len(df))

        if threshold > 0:
            df = df.query(f'gamma_prediction_mean > {threshold}').copy()
        if multiplicity > 2:
            df = df.query(f'num_triggered_telescopes >= {multiplicity}').copy()

        distance = calculate_distance_theta(df, source_alt=df.mc_alt.values * u.rad, source_az=df.mc_az.values * u.rad)

        b_68, bin_edges, binnumber = binned_statistic(df.mc_energy.values, distance, statistic=lambda y: np.percentile(y, 68), bins=bins)

        plt.step(bin_center, b_68, lw=2, label=os.path.basename(input_file), where='mid')

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
    if output:
        plt.savefig(output)
    else:
        plt.show()


if __name__ == "__main__":
    main()
