import click
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
import fact.io
from . import make_energy_bins
from matplotlib.colors import LogNorm
from .colors import default_cmap



@click.command()
@click.argument('gamma_file', type=click.Path(exists=True))
@click.option('-o', '--output', type=click.Path(exists=False))
@click.option('-t', '--title', default=None)
@click.option('-cm', '--colormap', default=default_cmap)
@click.option('-n', '--norm', type=click.Choice(['log', 'linear']), default='log')
def main(gamma_file, output, title, colormap, norm):
    bins, bin_center, bin_widths = make_energy_bins(e_min=0.003 * u.TeV, e_max=330 * u.TeV, bins=40)

    gammas = fact.io.read_data(gamma_file, key='array_events').dropna()

    e_true = gammas.mc_energy
    e_reco = gammas.gamma_energy_prediction_mean

    h, _, _ = np.histogram2d(e_true, e_reco, bins=[bins, bins])

    if norm == 'log':
        norm = LogNorm()
    else:
        norm = None

    plt.pcolormesh(bins, bins, h.T, cmap=colormap, norm=norm)
    plt.colorbar()
    plt.plot([0.003, 330], [0.003, 330], lw=1, color='gray')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$E_{True} / TeV$')
    plt.xlabel(r'$E_{Reco} / TeV$')
    plt.tight_layout()

    if output:
        plt.savefig(output)
    else:
        plt.show()


if __name__ == "__main__":
    main()
