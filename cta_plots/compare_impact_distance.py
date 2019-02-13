import click
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from scipy.stats import binned_statistic
import fact.io
from spectrum import make_energy_bins
from colors import color_cycle


@click.command()
@click.argument('input_files', nargs=-1, type=click.Path(exists=True))
@click.option('-o', '--output', type=click.Path(exists=False))
@click.option('-t', '--threshold', default=0.0)
@click.option('-l', '--label', default=None, multiple=True)
@click.option('--complementary/--no-complementary', default=False)
def main(input_files, output, threshold, complementary, label):

    bins, bin_center, bin_widths = make_energy_bins(e_min=0.003 * u.TeV, e_max=330 * u.TeV, bins=10)
    columns = ['mc_core_x', 'mc_core_y', 'core_x_prediction', 'core_y_prediction', 'mc_energy']

    for input_file, color, l in zip(input_files, color_cycle, label):
        df = fact.io.read_data(input_file, key='array_events', columns=columns).dropna()

        distance = np.sqrt((df.mc_core_x - df.core_x_prediction)**2 + (df.mc_core_y - df.core_y_prediction)**2)

        x = df.mc_energy.values
        y = distance

        b_50, bin_edges, binnumber = binned_statistic(x, y, statistic='median', bins=bins)
        bin_centers = np.sqrt(bin_edges[1:] * bin_edges[:-1])

        plt.step(bin_centers, b_50, lw=2, color=color, label=l, where='mid')


    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('Distance to True Impact Position / meter')
    plt.xlabel(r'True Energy / TeV ')
    plt.legend()
    plt.tight_layout()
    if output:
        plt.savefig(output)
    else:
        plt.show()


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
