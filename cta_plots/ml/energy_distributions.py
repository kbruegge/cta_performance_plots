import click
import numpy as np
import matplotlib.pyplot as plt
from fact.io import read_data
from cycler import cycler
from cta_plots import make_energy_bins, colors
import astropy.units as u

@click.command()
@click.argument(
    'predicted_gammas', type=click.Path(
        exists=True,
        dir_okay=False,
    ))
@click.argument(
    'predicted_protons', type=click.Path(
        exists=True,
        dir_okay=False,
    ))
@click.option(
    '-o', '--output_file', type=click.Path(
        exists=False,
        dir_okay=False,
    ))
@click.option('-w', '--what', default='mean', type=click.Choice(['per-telescope', 'mean', 'single', 'median', 'weighted-mean', 'min', 'max', 'brightest']))
@click.option('-n', '--n_bins', default=20)
def main(predicted_gammas, predicted_protons, output_file, what, n_bins):
    e_min, e_max = 0.02 * u.TeV, 200 * u.TeV
    bins, _, _ = make_energy_bins(e_min=e_min, e_max=e_max, bins=n_bins, centering='log')


    if what == 'mean':
        cols = ['mc_energy', 'gamma_energy_prediction_mean']
        gammas = read_data(predicted_gammas, key='array_events', columns=cols).dropna()
        protons = read_data(predicted_protons, key='array_events', columns=cols).dropna()

        fig, ax = plt.subplots(1)
        ax.hist(gammas.gamma_energy_prediction_mean.values, bins=bins, histtype='step', density=True, linewidth=2, color=colors.main_color, label='gammas')
        ax.hist(gammas.mc_energy.values, bins=bins, histtype='step', density=True, linewidth=2, color=colors.main_color, alpha=0.5)

        ax.hist(protons.gamma_energy_prediction_mean.values, bins=bins, histtype='step', density=True, linewidth=2, color=colors.main_color_complement, label='protons')
        ax.hist(protons.mc_energy.values, bins=bins, histtype='step', density=True, linewidth=2, color=colors.main_color_complement, alpha=0.5)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()

    # elif what == 'single':
    #     cols = ['mc_energy', 'gamma_energy_prediction']
    #     gammas = read_data(predicted_gammas, key='telescope_events', columns=cols).dropna()
    #     protons = read_data(predicted_protons, key='telescope_events', columns=cols).dropna()

    #     fig, ax = plt.subplots(1)
    #     ax.hist(gammas.gamma_prediction.values, bins=bins, histtype='step', density=True, linewidth=2)
    #     ax.hist(protons.gamma_prediction.values, bins=bins, histtype='step', density=True, linewidth=2)

    plt.xlabel('Energy / TeV')
    plt.ylabel('Normalized Counts')
    plt.tight_layout()
    if output_file:
        plt.savefig(output_file)
    else:
        plt.show()


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()
