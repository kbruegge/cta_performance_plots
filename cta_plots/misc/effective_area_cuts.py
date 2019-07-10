import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.stats import binom_conf_interval
import astropy.units as u

from cta_plots import make_energy_bins, load_effective_area_requirement, load_signal_events, apply_cuts
from cta_plots.mc.spectrum import MCSpectrum
from cta_plots.colors import main_color
from fact.io import read_data

cols = [
    'mc_energy',
    'gamma_prediction_mean',
    'gamma_energy_prediction_mean',
    'mc_alt',
    'mc_az',
    'alt',
    'az',
    'num_triggered_telescopes',
    'total_intensity',
]


@click.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('-o', '--output', type=click.Path(exists=False))
@click.option('-b', '--n_bins', default=20, show_default=True)
@click.option('-p', '--cuts_path', type=click.Path(exists=True))
@click.option('--reference/--no-reference', default=True)
def main(input_file, output, n_bins, cuts_path,  reference):

    bins, bin_center, bin_widths = make_energy_bins(e_min=0.008 * u.TeV, e_max=200 * u.TeV, bins=n_bins)

    gammas, _, _ = load_signal_events(input_file, columns=cols)

    if cuts_path:
        gammas = apply_cuts(gammas, cuts_path, theta_cuts=True, sigma=0)

    runs = read_data(input_file, key='runs')
    mc_production = MCSpectrum.from_cta_runs(runs)

    gammas_energy = gammas.gamma_energy_prediction_mean.values

    hist_all = mc_production.expected_events_for_bins(energy_bins=bins)
    hist_selected, _ = np.histogram(gammas_energy, bins=bins)

    invalid = hist_selected > hist_all
    hist_selected[invalid] = hist_all[invalid]
    # use astropy to compute errors on that stuff
    lower_conf, upper_conf = binom_conf_interval(hist_selected, hist_all)

    # scale confidences to match and split
    lower_conf = lower_conf * mc_production.generation_area
    upper_conf = upper_conf * mc_production.generation_area

    area = (hist_selected / hist_all) * mc_production.generation_area

    # matplotlib wants relative offsets for errors. the conf values are absolute.
    lower = area - lower_conf
    upper = upper_conf - area

    mask = area > 0
    plt.errorbar(
        bin_center.value[mask],
        area.value[mask],
        xerr=bin_widths.value[mask] / 2.0,
        yerr=[lower.value[mask], upper.value[mask]],
        linestyle='',
        color=main_color,
    )


    if reference:
        df = load_effective_area_requirement()
        plt.plot(df.energy, df.effective_area, '--', color='gray', label='Prod3b reference')

    plt.legend()


    plt.ylim([100, 1E8])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$E_{\mathrm{Reco}} /  \mathrm{TeV}$')
    plt.ylabel(r'$\mathrm{Mean Effective\; Area} / \mathrm{m}^2$')
    plt.tight_layout()
    if output:
        plt.savefig(output)
    else:
        plt.show()


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
