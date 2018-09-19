import click
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from scipy.stats import binned_statistic
import fact.io
from spectrum import make_energy_bins
from colors import color_cycle
from ctapipe.instrument import get_atmosphere_profile_functions


@click.command()
@click.argument('input_files', nargs=-1, type=click.Path(exists=True))
@click.option('-o', '--output', type=click.Path(exists=False))
@click.option('-s', '--site', default='paranal', type=click.Choice(['paranal', 'orm']))
@click.option('--complementary/--no-complementary', default=False)
@click.option('-l', '--label', default=None, multiple=True)
def main(input_files, output, site, complementary, label):

    bins, bin_center, bin_widths = make_energy_bins(e_min=0.003 * u.TeV, e_max=330 * u.TeV, bins=10)

    for input, color, l in zip(input_files, color_cycle, label):
        df = fact.io.read_data(input, key='array_events', columns=['mc_energy', 'h_max_prediction', 'mc_x_max']).dropna()
        thickness, altitude = get_atmosphere_profile_functions(site)

        mc_h_max = altitude(df.mc_x_max.values * u.Unit('g/cm^2')).value
        y = mc_h_max - df.h_max_prediction
        x = df.mc_energy.values

        b_50, bin_edges, binnumber = binned_statistic(x, y, statistic='median', bins=bins)


        bin_centers = np.sqrt(bin_edges[1:] * bin_edges[:-1])
        plt.step(bin_centers, b_50, lw=2, color=color, label=l, where='mid')

    plt.xscale('log')
    # plt.yscale('log')
    plt.ylabel('Distance to true H max  / meter')
    plt.xlabel(r'True Energy / TeV ')
    plt.legend()
    plt.tight_layout()
    if output:
        plt.savefig(output)
    else:
        plt.show()


if __name__ == "__main__":
    main()
