import click
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from scipy.stats import binned_statistic
import fact.io
from . import make_energy_bins
from matplotlib.colors import PowerNorm
from .colors import default_cmap, main_color
from ctapipe.instrument import get_atmosphere_profile_functions



@click.command()
@click.argument('input_dl3_file', type=click.Path(exists=True))
@click.option('-o', '--output', type=click.Path(exists=False))
@click.option('-s', '--site', default='paranal', type=click.Choice(['paranal', 'orm']))
@click.option('-cm', '--colormap', default=default_cmap)
@click.option('-c', '--color', default=main_color)
def main(input_dl3_file, output, site, colormap, color):
    df = fact.io.read_data(input_dl3_file, key='array_events', columns=['mc_energy', 'h_max_prediction', 'mc_x_max']).dropna()
    df = df.loc[df.mc_x_max > 0]
    thickness, altitude = get_atmosphere_profile_functions(site)

    mc_h_max = altitude(df.mc_x_max.values * u.Unit('g/cm^2')).value
    y = mc_h_max - df.h_max_prediction

    bins, bin_center, bin_widths = make_energy_bins(e_min=0.003 * u.TeV, e_max=330 * u.TeV, bins=20)
    x = df.mc_energy.values

    b_50, bin_edges, binnumber = binned_statistic(x, y, statistic='median', bins=bins)

    bin_centers = np.sqrt(bin_edges[1:] * bin_edges[:-1])

    log_emin, log_emax = np.log10(bins.min().value), np.log10(bins.max().value)

    plt.hexbin(x, y, xscale='log', extent=(log_emin, log_emax, -5000, 5000), cmap=colormap, norm=PowerNorm(0.5))
    plt.colorbar()
    plt.plot(bin_centers, b_50, lw=2, color=color, label='Median')

    plt.xscale('log')
    plt.ylabel('Distance to true H max  / meter')
    plt.xlabel(r'$E_{True} / TeV$')

    plt.tight_layout()
    if output:
        plt.savefig(output)
    else:
        plt.show()


if __name__ == "__main__":
    main()
