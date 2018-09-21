import click
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from scipy.stats import binned_statistic
import fact.io
from . import make_energy_bins, load_energy_resolution_requirement
from matplotlib.colors import PowerNorm
from .colors import default_cmap, main_color


@click.command()
@click.argument('input_dl3_file', type=click.Path(exists=True))
@click.option('-o', '--output', type=click.Path(exists=False))
@click.option('-t', '--threshold', default=0.0)
@click.option('-m', '--multiplicity', default=2)
@click.option('-c', '--color', default=main_color)
@click.option('--reference/--no-reference', default=False)
@click.option('--relative/--no-relative', default=True)
@click.option('--plot_e_reco', is_flag=True, default=False)
def main(input_dl3_file, output, threshold, multiplicity, color, reference, relative, plot_e_reco):
    bins, bin_center, bin_widths = make_energy_bins(e_min=0.02 * u.TeV, e_max=150 * u.TeV, bins=20)
    columns = ['array_event_id', 'mc_energy', 'gamma_energy_prediction_mean', 'num_triggered_telescopes']

    if threshold > 0:
        columns.append('gamma_prediction_mean')

    gammas = fact.io.read_data(input_dl3_file, key='array_events', columns=columns).dropna()

    if multiplicity > 2:
        gammas = gammas.query(f'num_triggered_telescopes >= {multiplicity}').copy()

    if threshold > 0:
        gammas = gammas.query(f'gamma_prediction_mean > {threshold}').copy()

    e_true = gammas.mc_energy
    e_reco = gammas.gamma_energy_prediction_mean

    if plot_e_reco:
        e_x = e_reco
    else:
        e_x = e_true

    resolution = (e_reco - e_true) / e_true
    if relative:
        iqr, bin_edges, binnumber = binned_statistic(e_x, resolution, statistic=lambda y: ((np.percentile(y, 84) - np.percentile(y, 16)) / 2), bins=bins)
    else:
        iqr, bin_edges, binnumber = binned_statistic(e_x, resolution, statistic=lambda y: np.percentile(np.abs(y), 68), bins=bins)

    plt.ylabel('$\\frac{E_R}{E_T} - 1$')

    max_y = 1.
    bins_y = np.linspace(-0.5, max_y, 40)

    log_emin, log_emax = np.log10(bins.min().value), np.log10(bins.max().value)


    plt.hexbin(e_x, resolution, xscale='log', extent=(log_emin, log_emax, -1, max_y), cmap=default_cmap, norm=PowerNorm(0.5))
    plt.colorbar()

    if relative:
        label = '$0.5 \cdot IQR_{68}$ of $(E_R / E_T) - 1$'
    else:
        label = '$Q_{68}$ of $abs((E_R /  E_T) - 1)$'

    plt.hlines(iqr, bins[:-1], bins[1:], lw=2, color=color, label=label)

    if reference:
        df = load_energy_resolution_requirement()
        plt.plot(df.energy, df.resolution, '--', color='#5b5b5b', label='Prod3B Reference')

    plt.xscale('log')

    if plot_e_reco:
        plt.xlabel('$E_{Reco} / TeV$')
    else:
        plt.xlabel('$E_{True} / TeV$')
    plt.ylim([bins_y.min(), bins_y.max()])
    plt.legend()
    plt.tight_layout()
    if output:
        plt.savefig(output)
    else:
        plt.show()


if __name__ == "__main__":
    main()
