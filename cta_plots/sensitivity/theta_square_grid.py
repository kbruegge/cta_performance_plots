import click
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import pandas as pd
from cta_plots.coordinate_utils import (
    calculate_distance_to_true_source_position,
    calculate_distance_to_point_source,
)
from cta_plots.spectrum import MCSpectrum, CrabSpectrum, CosmicRaySpectrum, CTAElectronSpectrum
import matplotlib.offsetbox as offsetbox
from fact.io import read_data
from cta_plots import load_signal_events, load_background_events, ELECTRON_TYPE
# from cta_plots import load_signal_events, load_background_events, ELECTRON_TYPE
from cta_plots.sensitivity.optimize import find_best_cuts
from cta_plots.binning import make_energy_bins
from tqdm import tqdm


def add_theta_square_histogram(gammas_gammalike, background_gammalike, theta_square_cut, ax):
    on = gammas_gammalike
    off = background_gammalike

    bins = np.arange(0, 1, 0.015)

    h_off, _ = np.histogram(off['theta'] ** 2, bins=bins, weights=off.weight)
    h_on, _ = np.histogram(on['theta'] ** 2, bins=bins, weights=on.weight)

    # if True:
    #     h, _ = np.histogram(off['theta'] ** 2, bins=bins)
    #     is_valid = (h == 0).sum() < len(h)//2 # less than half of the bins have to be nonzero 
    #     print(f'Number of zero entries: {(h_off == 0).sum()}, length of hist : {len(h_off)},  ')
    #     print(f'Is valid: {is_valid} ')

    ax.step(bins[:-1], h_on + h_off.mean(), where='post', label='on events')
    ax.step(bins[:-1], h_off, where='post', label='off events protons')

    off_electrons = off.query(f'type == {ELECTRON_TYPE}')
    h_off_electrons, _ = np.histogram(off_electrons['theta'] ** 2, bins=bins, weights=off_electrons.weight)
    ax.step(bins[:-1],  h_off_electrons, where='post', label='off events electrons')
    
    ax.axhline(y=h_off.mean(), color='C1', lw=1, alpha=0.7)
    ax.set_ylim([0, max(h_on + h_off.mean()) * 1.18])


def add_text_to_axis(
    gammas_gammalike, background_gammalike, best_prediction_cut, best_theta_square_cut, best_significance, e_low, e_high, ax
):
    textstr = '\n'.join(
        [
            f'Significance: {best_significance:.2f}',
            f'Cuts: {best_theta_square_cut:.3f},  {best_prediction_cut:.3f}',
            f'Total Signal: {len(gammas_gammalike)}',
            f'Total Bkg: {len(background_gammalike)}',
            f'Energy Range {e_low:.3f}, {e_high:.3f}'
        ]
    )
    ob = offsetbox.AnchoredText(textstr, loc=1, prop={'fontsize': 7})
    ob.patch.set_facecolor('lightgray')
    ob.patch.set_alpha(0.1)
    ax.add_artist(ob)
    ax.axvline(x=best_theta_square_cut, color='gray', lw=1, alpha=0.7)


@click.command()
@click.argument('gammas_path', type=click.Path(exists=True))
@click.argument('protons_path', type=click.Path(exists=True))
@click.argument('electrons_path', type=click.Path(exists=True))
@click.option('-o', '--output', type=click.Path(exists=False))
def main(gammas_path, protons_path, electrons_path, output):

    t_obs = 30 * u.min

    gammas, source_alt, source_az = load_signal_events(gammas_path, assumed_obs_time=t_obs)
    background = load_background_events(
        protons_path, electrons_path, source_alt, source_az, assumed_obs_time=t_obs
    )
    n_bins = 20
    e_min, e_max = 0.02 * u.TeV, 200 * u.TeV
    bin_edges, _, _ = make_energy_bins(e_min=e_min, e_max=e_max, bins=n_bins, centering='log')

    theta_square_cuts = np.arange(0.01, 0.35, 0.02)
    prediction_cuts = np.arange(0.3, 1, 0.05)
    multiplicities = [2, 3, 4, 5, 6, 7]

    rows = int(np.sqrt(len(bin_edges)) + 1)
    cols = int(np.sqrt(len(bin_edges)))
    fig, axs = plt.subplots(rows, cols, figsize=(16, 16), constrained_layout=True, sharex=True)

    groups = pd.cut(gammas.gamma_energy_prediction_mean, bins=bin_edges)   
    g = gammas.groupby(groups) 

    groups = pd.cut(background.gamma_energy_prediction_mean, bins=bin_edges)   
    b = background.groupby(groups) 
    
    iterator = zip(g, b, axs.ravel())
    # alpha = 0.2

    for (n, signal_in_range), (_, background_in_range), ax in tqdm(iterator, total=n_bins):
        # print(f'Energy mean before passing data: {signal_in_range.gamma_energy_prediction_mean.mean()}')
        best_sensitivity, best_prediction_cut, best_theta_cut, best_significance, best_mult = find_best_cuts(
            theta_square_cuts, prediction_cuts, multiplicities, signal_in_range, background_in_range, alpha=1, criterion='significance', n_jobs=8
        )
        # print('--//----'*10)
        # print(f'Best prediction cut {best_prediction_cut}')
        # print(f'Best theta_square cut {best_theta_square_cut}')
        gammas_gammalike = signal_in_range[signal_in_range.gamma_prediction_mean >= best_prediction_cut]
        background_gammalike = background_in_range[background_in_range.gamma_prediction_mean >= best_prediction_cut]

        add_theta_square_histogram(gammas_gammalike, background_gammalike, best_theta_cut, ax)
        add_text_to_axis(
            gammas_gammalike,
            background_gammalike,
            best_prediction_cut,
            best_theta_cut,
            best_significance,
            n.left,
            n.right,
            ax,
        )

    if output:
        plt.savefig(output)
    else:
        plt.show()


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()
