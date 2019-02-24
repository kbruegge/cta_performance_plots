import click
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from scipy.stats import binned_statistic
import fact.io
from cta_plots import make_energy_bins, load_energy_resolution_requirement
from matplotlib.colors import PowerNorm
from cta_plots.colors import default_cmap, main_color
import os

@click.command()
@click.argument('input_dl3_file', type=click.Path(exists=True))
@click.option('-o', '--output', type=click.Path(exists=False))
@click.option('-t', '--threshold', default=0.0)
@click.option('-m', '--multiplicity', default=2)
@click.option('-c', '--color', default=main_color)
def main(input_dl3_file, output, threshold, multiplicity, color, ):
    
    n_bins = 20
    e_min, e_max = 0.02 * u.TeV, 200 * u.TeV
    bins, bin_center, _ = make_energy_bins(e_min=e_min, e_max=e_max, bins=n_bins, centering='log')
    
    columns = ['array_event_id', 'mc_energy', 'gamma_energy_prediction_mean', 'num_triggered_telescopes']

    if threshold > 0:
        columns.append('gamma_prediction_mean')

    gammas = fact.io.read_data(input_dl3_file, key='array_events', columns=columns).dropna()

    if multiplicity > 2:
        gammas = gammas.query(f'num_triggered_telescopes >= {multiplicity}').copy()

    if threshold > 0:
        gammas = gammas.query(f'gamma_prediction_mean >= {threshold}').copy()

    e_true = gammas.mc_energy
    e_reco = gammas.gamma_energy_prediction_mean

    bias = (e_reco - e_true) / e_reco
    mean_bias, bin_edges, binnumber = binned_statistic(e_reco, bias, statistic='mean', bins=bins)

    plt.ylabel('$\\frac{{E_R} - {E_T}}{E_R}$')

    max_y = .7
    bins_y = np.linspace(-.7, max_y, 40)

    log_emin, log_emax = np.log10(bins.min().value), np.log10(bins.max().value)


    plt.hexbin(e_reco, bias, xscale='log', extent=(log_emin, log_emax, -1, max_y), cmap=default_cmap, norm=PowerNorm(0.5))
    plt.colorbar()

    plt.hlines(mean_bias, bins[:-1], bins[1:], lw=2, color=color)

    plt.xscale('log')

    plt.xlabel('$E_{Reco} / TeV$')
    plt.ylim([bins_y.min(), bins_y.max()])
    plt.tight_layout()
    if output:
        plt.savefig(output)
        df = pd.DataFrame({'bias': mean_bias, 'energy_prediction': bin_center, 'multiplicity': multiplicity})
        n, _ = os.path.splitext(output)
        print(f"writing csv to {n + '.csv'}")
        df.to_csv(n + '.csv', index=False)
    else:
        plt.show()


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
