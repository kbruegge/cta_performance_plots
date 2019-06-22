import click
import matplotlib.pyplot as plt
import numpy as np

from astropy.stats import binom_conf_interval
import astropy.units as u

from cta_plots.binning import make_default_cta_binning
from cta_plots.sensitivity import load_effective_area_reference
from cta_plots.spectrum import MCSpectrum
from cta_plots.colors import color_cycle
from cta_plots import load_signal_events, apply_cuts, load_runs


@click.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('-l', '--label', multiple=True)
@click.option('-o', '--output', type=click.Path(exists=False))
@click.option('-m', '--multiplicity', default=2)
@click.option('-t', '--threshold', default=0.0, show_default=True, help='prediction threshold to apply', multiple=True)
@click.option('-p', '--cuts_path', type=click.Path(exists=True))
@click.option('--reference/--no-reference', default=True)
def main(input_file, label, output, multiplicity, threshold, cuts_path, reference):

    bins, bin_center, bin_widths = make_default_cta_binning(e_min=0.005 * u.TeV, bins_per_decade=15)

    gammas, _, _ = load_signal_events(input_file, calculate_weights=False, )

    if multiplicity > 2:
        gammas = gammas.query(f'num_triggered_telescopes >= {multiplicity}').copy()

    gammas = apply_cuts(gammas, cuts_path=cuts_path, sigma=1)

    runs = load_runs(input_file)
    mc_production = MCSpectrum.from_cta_runs(runs)

    if not threshold:
        threshold = [0]
    if not label:
        label = [None] * len(threshold)

    for t, c, l in zip(threshold, color_cycle, label):

        if t > 0:
            gammas = gammas.copy().loc[gammas.gamma_prediction_mean >= t]
        else:
            gammas = gammas.copy()
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
            color=c,
            label=l
        )


    if reference:
        df = load_effective_area_reference()
        plt.plot(df.energy, df.effective_area, '--', color='gray', label='Prod3b reference')

    plt.legend(framealpha=0)


    plt.ylim([100, 1E8])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Estimated Energy / TeV')
    plt.ylabel('Effective Area / $\\text{m}^2$')
    plt.tight_layout(pad=0, rect=(0.001, 0, 1, 0.99))

    if output:
        plt.savefig(output)
    else:
        plt.show()


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
