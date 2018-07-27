import click
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from scipy.stats import binned_statistic
import fact.io
from spectrum import make_energy_bins
from matplotlib.colors import PowerNorm
from colors import default_cmap, main_color


@click.command()
@click.argument('input_dl3_file', type=click.Path(exists=True))
@click.option('-o', '--output', type=click.Path(exists=False))
@click.option('-t', '--threshold', default=0.0)
@click.option('-c', '--color', default=main_color)
@click.option('--reference/--no-reference', default=False)
@click.option('--relative/--no-relative', default=True)
def main(input_dl3_file, output, threshold, color, reference, relative):
    bins, bin_center, bin_widths = make_energy_bins(e_min=0.01 * u.TeV, e_max=300 * u.TeV, bins=20)
    columns = ['array_event_id', 'mc_energy', 'gamma_prediction_mean', 'gamma_energy_prediction_mean']

    gammas = fact.io.read_data(input_dl3_file, key='array_events', columns=columns).dropna()
    if threshold > 0:
        gammas = gammas.query(f'gamma_prediction_mean > {threshold}').copy()

    gammas['energy_bin'] = pd.cut(gammas.mc_energy, bins)

    e_true = gammas.mc_energy
    e_reco = gammas.gamma_energy_prediction_mean

    if relative:
        resolution = (e_reco - e_true) / e_true
    else:
        resolution = (e_reco - e_true)

    b_50, bin_edges, binnumber = binned_statistic(e_true, resolution, statistic='median', bins=bins)
    iqr, bin_edges, binnumber = binned_statistic(e_true, resolution, statistic=lambda y: ((np.percentile(y, 84) - np.percentile(y, 16)) / 2), bins=bins)


    # bin_centers = np.sqrt(bin_edges[1:] * bin_edges[:-1])
    max_y = 2
    bins_y = np.linspace(-1, max_y, 40)
    # h, _, _ = np.histogram2d(e_true, resolution, bins=[bins, bins_y])

    # plt.pcolormesh(bins, bins_y, h.T, cmap='Greys', norm=PowerNorm(0.25))
    log_emin, log_emax = np.log10(bins.min().value), np.log10(bins.max().value)
    plt.hexbin(e_true, resolution, xscale='log', extent=(log_emin, log_emax, -1, max_y), cmap=default_cmap, norm=PowerNorm(0.5))
    plt.colorbar()
    if relative:
        label = '$IQR_{68}$ of $\\frac{E_R - E_T}{E_T}$'
    else:
        label = '$IQR_{68}$ of $E_R - E_T$'

    plt.hlines(iqr, bins[:-1], bins[1:], lw=2, color=color, label=label)

    if reference:
        path = 'resources/CTA-Performance-prod3b-v1-South-20deg-50h-Eres.txt'
        df = pd.read_csv(path, delimiter='\t\t', skiprows=11, names=['energy', 'resolution'], engine='python')
        plt.plot(df.energy, df.resolution, '--', color='#5b5b5b', label='Prod3B Reference')


    plt.xscale('log')
    if relative:
        plt.ylabel('$\\frac{E_R - E_T}{E_T}$')
    else:
        plt.yscale('log')
        plt.ylabel('$E_R - E_T$')
    plt.xlabel(r'$True Energy /  \mathrm{TeV}$')
    plt.ylim([bins_y.min(), bins_y.max()])
    plt.legend()
    plt.tight_layout()
    if output:
        plt.savefig(output)
    else:
        plt.show()


if __name__ == "__main__":
    main()
