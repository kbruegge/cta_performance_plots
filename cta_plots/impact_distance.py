import click
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from scipy.stats import binned_statistic
import fact.io
from . import make_energy_bins
from matplotlib.colors import PowerNorm
from .colors import default_cmap, main_color


@click.command()
@click.argument('input_dl3_file', type=click.Path(exists=True))
@click.option('-o', '--output', type=click.Path(exists=False))
@click.option('-t', '--threshold', default=0.0)
@click.option('-m', '--multiplicity', default=2)
@click.option('-cm', '--colormap', default=default_cmap)
@click.option('-c', '--color', default=main_color)
def main(input_dl3_file, output, threshold, multiplicity, colormap, color):
    df = fact.io.read_data(input_dl3_file, key='array_events', ).dropna()

    if multiplicity > 2:
        df = df.query(f'num_triggered_telescopes >= {multiplicity}')

    distance = np.sqrt((df.mc_core_x - df.core_x_prediction)**2 + (df.mc_core_y - df.core_y_prediction)**2)

    bins, bin_center, bin_widths = make_energy_bins(e_min=0.003 * u.TeV, e_max=330 * u.TeV, bins=20)

    x = df.mc_energy.values
    y = distance

    b_50, bin_edges, binnumber = binned_statistic(x, y, statistic=np.nanmedian, bins=bins)

    bin_centers = np.sqrt(bin_edges[1:] * bin_edges[:-1])
    bins_y = np.logspace(np.log10(1), np.log10(1500.8), 100)

    log_emin, log_emax = np.log10(bins.min().value), np.log10(bins.max().value)
    log_ymin, log_ymax = np.log10(bins_y.min()), np.log10(bins_y.max())
    plt.hexbin(x, y, xscale='log', yscale='log', extent=(log_emin, log_emax, log_ymin, log_ymax), cmap=colormap, norm=PowerNorm(0.5))
    plt.colorbar()
    plt.plot(bin_centers, b_50, lw=2, color=color, label='Median')

    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('Distance to True Position / meter')
    plt.xlabel(r'True Energy / TeV ')
    plt.legend()
    plt.tight_layout()
    if output:
        plt.savefig(output)
    else:
        plt.show()


if __name__ == "__main__":
    main()
