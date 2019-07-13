import click
import matplotlib.pyplot as plt
import numpy as np

from astropy.stats import binom_conf_interval
import astropy.units as u
import pandas as pd
# from scipy.interpolate import interp1d
from matplotlib import cm

from cta_plots.binning import make_default_cta_binning
from cta_plots.sensitivity import load_effective_area_reference
from cta_plots.spectrum import MCSpectrum
from cta_plots.colors import color_cycle
from cta_plots import load_signal_events, apply_cuts, load_runs, load_data_description, create_interpolated_function


def prediction_function(cuts_path, sigma=0):
    cuts = pd.read_csv(cuts_path)
    bin_center = np.sqrt(cuts.e_min * cuts.e_max)
    return create_interpolated_function(bin_center, cuts.prediction_cut, sigma=sigma)


@click.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('-o', '--output', type=click.Path(exists=False))
@click.option('-p', '--cuts_path', type=click.Path(exists=True))
@click.option('--reference/--no-reference', default=True)
@click.option('--cmap', default='magma')
def main(input_file, output, cuts_path, reference, cmap):

    bins, bin_center, bin_widths = make_default_cta_binning(e_min=0.005 * u.TeV, bins_per_decade=15)

    gammas, _, _ = load_signal_events(input_file, calculate_weights=False, )
    gammas.dropna(inplace=True)

    sigma = 1
    gammas = apply_cuts(gammas, cuts_path=cuts_path, theta_cuts=True, sigma=sigma)

    runs = load_runs(input_file)
    mc_production = MCSpectrum.from_cta_runs(runs)

    data_description = load_data_description(input_file, gammas, cuts_path=cuts_path)

    gammas_energy = gammas.mc_energy.values

    hist_all = mc_production.expected_events_for_bins(energy_bins=bins)
    hist_selected, _ = np.histogram(gammas_energy, bins=bins)

    invalid = hist_selected > hist_all
    hist_selected[invalid] = hist_all[invalid]
    # use astropy to compute errors on that stuff
    lower_conf, upper_conf = binom_conf_interval(hist_selected, hist_all, conf=0.95)

    # scale confidences to match and split
    lower_conf = lower_conf * mc_production.generation_area
    upper_conf = upper_conf * mc_production.generation_area

    area = (hist_selected / hist_all) * mc_production.generation_area

    # matplotlib wants relative offsets for errors. the conf values are absolute.
    lower = area - lower_conf
    upper = upper_conf - area
    
    mask = area > 0
    color = None
    if cuts_path:
        f_prediction = prediction_function(cuts_path, sigma=0)
        colormap = cm.get_cmap(cmap, 512)
        color = colormap(f_prediction(bin_center.value[mask]))

        sm = cm.ScalarMappable(cmap=colormap)
        plt.colorbar(sm, label='Prediction Threshold', pad=0.01)
        
    plt.errorbar(
        bin_center.value[mask],
        area.value[mask],
        xerr=bin_widths.value[mask] / 2.0,
        yerr=[lower.value[mask], upper.value[mask]],
        linestyle='',
        color=color if color is not None else next(color_cycle),
        # label='Effective Area'
    )


    if reference:
        df = load_effective_area_reference()
        plt.plot(df.energy, df.effective_area, '--', color='gray', label='Reference')

    legend = plt.legend(framealpha=0, loc='upper left', handletextpad=1)
    # renderer = plt.gcf().canvas.get_renderer()
    # shift = max([t.get_window_extent(renderer).width for t in legend.get_texts()])
    for t in legend.get_texts():
        # print(t, shift)
        t.set_multialignment('right')
    #     t.set_ha('left') # ha is alias for horizontalalignment
        # t.set_position((shift,0))

    legend.set_title(data_description)
    legend._legend_box.align = "left"
    legend.get_title().set_alpha(0.5)


    plt.ylim([800, 0.5E8])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('True Energy / TeV')
    plt.ylabel('Effective Area / $\\text{m}^2$')
    plt.tight_layout(pad=0, rect=(0.001, 0, 1.041, 0.99))

    if output:
        plt.savefig(output)
    else:
        plt.show()


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
