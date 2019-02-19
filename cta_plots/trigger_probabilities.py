import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.stats import binom_conf_interval
import astropy.units as u

from cta_plots import make_energy_bins, load_effective_area_requirement
from cta_plots.mc.spectrum import MCSpectrum
from cta_plots.colors import color_cycle
from itertools import zip_longest

from fact.io import read_data


@click.command()
@click.argument('input_files', type=click.Path(exists=True, dir_okay=False), nargs=-1)
@click.option('-l', '--labels', multiple=True)
@click.option('-o', '--output', type=click.Path(exists=False))
@click.option('-b', '--n_bins', default=20, show_default=True)
@click.option('-t', '--threshold', default=0.0, show_default=True, help='prediction threshold to apply')
@click.option('--reference/--no-reference', default=False)
def main(input_files, labels, output, n_bins, threshold, reference):

    bins, bin_center, bin_widths = make_energy_bins(e_min=0.008 * u.TeV, e_max=200 * u.TeV, bins=n_bins)

    for input_file, label, color in zip_longest(input_files, labels, color_cycle):
        
        if not input_file:
            break
        
        events = read_data(input_file, key='array_events')
        runs = read_data(input_file, key='runs')
        mc_production = MCSpectrum.from_cta_runs(runs)

        if threshold > 0:
            events = events.loc[events.gamma_prediction_mean >= threshold]

        energies = events.gamma_energy_prediction_mean.values

        hist_all = mc_production.expected_events_for_bins(energy_bins=bins)
        hist_selected, _ = np.histogram(energies, bins=bins)

        invalid = hist_selected > hist_all
        hist_selected[invalid] = hist_all[invalid]
        
        # use astropy to compute errors on that stuff
        lower_conf, upper_conf = binom_conf_interval(hist_selected, hist_all)

        # scale confidences to match and split
        lower_conf = lower_conf
        upper_conf = upper_conf

        trigger_probability = (hist_selected / hist_all)

        # matplotlib wants relative offsets for errors. the conf values are absolute.
        lower = trigger_probability - lower_conf
        upper = upper_conf - trigger_probability

        mask = trigger_probability > 0
        plt.errorbar(
            bin_center.value[mask],
            trigger_probability[mask],
            xerr=bin_widths.value[mask] / 2.0,
            yerr=[lower[mask], upper[mask]],
            linestyle='',
            color=color,
            label=label,
        )


    if reference:
        df = load_effective_area_requirement()
        plt.plot(df.energy, df.effective_area, '--', color='gray', label='Prod3b reference')

    plt.legend()


    # plt.ylim([100, 1E8])
    plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel(r'$E_{\mathrm{Reco}} /  \mathrm{TeV}$')
    plt.ylabel('Trigger Probabilty')
    plt.tight_layout()
    if output:
        plt.savefig(output)
    else:
        plt.show()


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
