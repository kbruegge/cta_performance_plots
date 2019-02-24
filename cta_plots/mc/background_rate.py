import os
import astropy.units as u
import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from colorama import Fore
from fact.analysis import li_ma_significance
from tqdm import tqdm

from cta_plots import (load_sensitivity_reference,
                       make_energy_bins,
                       load_background_events,
                       load_signal_events,
                       apply_cuts,
                       )

from cta_plots.sensitivity_utils import find_relative_sensitivity
from cta_plots.mc.spectrum import CrabSpectrum



def plot_refrence(ax=None):
    df = load_sensitivity_reference()
    bin_edges = sorted(list(set(df.e_min) | set(df.e_max))) * u.TeV
    bin_center = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    sensitivity = df.sensitivity.values * u.erg / (u.cm ** 2 * u.s)

    if not ax:
        ax = plt.gca()

    xerr = [np.abs(bin_edges[:-1] - bin_center).value, np.abs(bin_edges[1:] - bin_center).value]
    ax.errorbar(
        bin_center.value, sensitivity.value, xerr=xerr, linestyle='', color='#3e3e3e', label='Reference'
    )
    return ax


@click.command()
@click.argument('gammas_path', type=click.Path(exists=True))
@click.argument('protons_path', type=click.Path(exists=True))
@click.argument('electrons_path', type=click.Path(exists=True))
@click.option('-p', '--cuts_path', type=click.Path(exists=True))
@click.option('-o', '--output', type=click.Path(exists=False))
@click.option('-c', '--color', default='xkcd:red')
@click.option('--reference/--no-reference', default=False)
def main(
    gammas_path,
    protons_path,
    electrons_path,
    cuts_path,
    output,
    color,
    reference,
):
    t_obs = 1 * u.s
    gammas, source_alt, source_az = load_signal_events(gammas_path, assumed_obs_time=t_obs)
    background = load_background_events(
        protons_path, electrons_path, source_alt, source_az, assumed_obs_time=t_obs
    )

    n_bins = 20
    e_min, e_max = 0.02 * u.TeV, 200 * u.TeV
    bin_edges, _, _ = make_energy_bins(e_min=e_min, e_max=e_max, bins=n_bins, centering='log')
    print(len(background))
    if cuts_path:
        background = apply_cuts(background, cuts_path, theta_cuts=True)
    print(len(background))
    # from IPython import embed; embed()
    fig, ax = plt.subplots()
    ax.hist(background.gamma_energy_prediction_mean, bins=bin_edges, weights=background.weight, histtype='step', lw=2)
    ax.hist(background.gamma_energy_prediction_mean, bins=bin_edges, weights=np.ones_like(background.weight)/18358000000.0, histtype='step', lw=2)
    # ax.hist(background, bins=bin_edges, weight=background.weight)
    if reference:
        plot_refrence(ax)
    

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim([1e-2, 10 ** (2.5)])
    ax.set_xlabel(r'$E_{Reco} /  \mathrm{TeV}$')
    ax.legend()

    if output:
        fig.savefig(output)
    else:
        plt.show()


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()
