import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.stats import binom_conf_interval
import astropy.units as u

from cta_plots import make_energy_bins, load_effective_area_requirement
from cta_plots.mc.spectrum import MCSpectrum, CrabSpectrum, CTAProtonSpectrum, CTAElectronSpectrum
from cta_plots.colors import color_cycle
from fact.io import read_data


@click.command()
@click.argument('gamma_file', type=click.Path(exists=True, dir_okay=False))
@click.argument('proton_file', type=click.Path(exists=True, dir_okay=False))
@click.argument('electron_file', type=click.Path(exists=True, dir_okay=False))
@click.option('-o', '--output', type=click.Path(exists=False))
@click.option('-b', '--n_bins', default=20, show_default=True)
@click.option('-t', '--threshold', default=0.0, show_default=True, help='prediction threshold to apply')
def main(gamma_file, proton_file, electron_file, output, n_bins, threshold):

    t_assumed_obs = 50 * u.h
    bins, bin_center, bin_widths = make_energy_bins(e_min=0.008 * u.TeV, e_max=200 * u.TeV, bins=n_bins)
    
    
    spectra = [CrabSpectrum(), CTAProtonSpectrum(), CTAElectronSpectrum()]
    labels = ['Gamma (Crab)', 'Proton', 'Electrons']
    iterator = zip([gamma_file, proton_file, electron_file], spectra, color_cycle, labels)
    for input_file, spectrum, color, label in iterator:
        
        events = read_data(input_file, key='array_events')
        runs = read_data(input_file, key='runs')
        mc_production = MCSpectrum.from_cta_runs(runs)

        if threshold > 0:
            events = events.loc[events.gamma_prediction_mean >= threshold]

        estimated_energies = events.gamma_energy_prediction_mean.values * u.TeV
        weights = mc_production.reweigh_to_other_spectrum(spectrum, estimated_energies, t_assumed_obs=t_assumed_obs)

        plt.hist(estimated_energies, bins=bins, weights=weights, histtype='step', lw=2, color=color, label=label)

    plt.legend()


    # plt.ylim([100, 1E8])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$E_{\mathrm{Reco}} /  \mathrm{TeV}$')
    plt.ylabel(f'Triggered Counts in {t_assumed_obs}')
    plt.tight_layout()
    if output:
        plt.savefig(output)
    else:
        plt.show()


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
