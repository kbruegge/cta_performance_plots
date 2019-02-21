import click
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import pandas as pd
from fact.analysis import li_ma_significance
from scipy.optimize import brute
from cta_plots.coordinate_utils import (
    calculate_distance_to_true_source_position,
    calculate_distance_to_point_source,
)
from cta_plots.mc.spectrum import MCSpectrum, CrabSpectrum, CosmicRaySpectrum, CTAElectronSpectrum
import matplotlib.offsetbox as offsetbox
from fact.io import read_data
from cta_plots.coordinate_utils import load_signal_events, load_background_events, ELECTRON_TYPE
from cta_plots.coordinate_utils import find_best_detection_significance


@click.command()
@click.argument('gammas_path', type=click.Path(exists=True))
@click.argument('protons_path', type=click.Path(exists=True))
@click.argument('electrons_path', type=click.Path(exists=True))
@click.option('-o', '--output', type=click.Path(exists=False))
@click.option('-j', '--n_jobs', default=-4)
def main(gammas_path, protons_path, electrons_path, output, n_jobs):

    t_obs = 5 * u.min

    gammas, source_alt, source_az = load_signal_events(gammas_path, assumed_obs_time=t_obs)
    background = load_background_events(protons_path, electrons_path, source_alt, source_az,  assumed_obs_time=t_obs,)

    theta_square_cuts = np.arange(0.01, 0.35, 0.05)
    prediction_cuts = np.arange(0.4, 1, 0.1)

    best_prediction_cut, best_theta_square_cut, best_significance = find_best_detection_significance(
        theta_square_cuts, prediction_cuts, gammas, background, alpha=1
    )

    gammas_gammalike = gammas.query(f'gamma_prediction_mean > {best_prediction_cut}').copy()
    background_gammalike = background.query(f'gamma_prediction_mean > {best_prediction_cut}').copy()
    on = gammas_gammalike
    off = background_gammalike

    bins = np.arange(0, 0.6, 0.01)
    h_off, _ = np.histogram(off['theta'] ** 2, bins=bins, weights=off.weight)
    h_on, _ = np.histogram(on['theta'] ** 2, bins=bins, weights=on.weight)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.step(bins[:-1], h_on + h_off.mean(), where='post', label='on events')
    ax.step(bins[:-1], h_off, where='post', label='off events protons')

    off_electrons = off.query(f'type == {ELECTRON_TYPE}')
    h_off_electrons, _ = np.histogram(
        off_electrons['theta'] ** 2, bins=bins, weights=off_electrons.weight
    )
    ax.step(bins[:-1], h_off_electrons, where='post', label='off events electrons')

    ax.set_ylim([0, max(h_on + h_off) * 1.18])
    ax.axhline(y=h_off.mean(), color='C1', lw=1, alpha=0.7)
    ax.axvline(x=best_theta_square_cut, color='gray', lw=1, alpha=0.7)

    textstr = '\n'.join(
        [
            f'Observation Time: {t_obs}',
            f'Prediction Threshold: {best_prediction_cut:.2f}',
            f'Theta Square Cut: {(best_theta_square_cut):.2f}',
            f'Significance: {best_significance:.2f}',
        ]
    )

    ob = offsetbox.AnchoredText(textstr, loc=1, prop={'fontsize': 9})
    ob.patch.set_facecolor('lightgray')
    ob.patch.set_alpha(0.1)
    ax.add_artist(ob)

    if output:
        plt.savefig(output)
    else:
        plt.show()


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()
