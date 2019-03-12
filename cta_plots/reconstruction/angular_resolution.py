import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binned_statistic
from matplotlib.colors import PowerNorm

import astropy.units as u

from cta_plots.colors import default_cmap, main_color, color_cycle
from cta_plots.coordinate_utils import calculate_distance_to_true_source_position
from cta_plots.binning import make_default_cta_binning
from . import load_angular_resolution_requirement
from .. import add_colorbar_to_figure


def plot_angular_resolution(reconstructed_events, reference, plot_e_reco, ax=None):

    df = reconstructed_events
    distance = calculate_distance_to_true_source_position(df)

    e_min, e_max = 0.005 * u.TeV, 200 * u.TeV
    bins, bin_center, _ = make_default_cta_binning(e_min=e_min, e_max=e_max)


    if plot_e_reco:
        x = df.gamma_energy_prediction_mean.values
    else:
        x = df.mc_energy.values

    y = distance

    b_68, bin_edges, _ = binned_statistic(x, y, statistic=lambda y: np.nanpercentile(y, 68), bins=bins)

    bin_centers = np.sqrt(bin_edges[1:] * bin_edges[:-1])
    bins_y = np.logspace(np.log10(0.005), np.log10(50.8), 100)

    log_emin, log_emax = np.log10(bins.min().value), np.log10(bins.max().value)
    log_ymin, log_ymax = np.log10(bins_y.min()), np.log10(bins_y.max())
    
    if not ax:
        fig, ax = plt.subplots(1, 1)
    
    im = ax.hexbin(x, y, xscale='log', yscale='log', extent=(log_emin, log_emax, log_ymin, log_ymax), cmap=default_cmap, norm=PowerNorm(0.5))
    
    add_colorbar_to_figure(im, fig, ax)
    ax.plot(bin_centers, b_68, lw=2, color=main_color, label='68% Percentile')

    if reference:
        df = load_angular_resolution_requirement()
        ax.plot(df.energy, df.resolution, '--', color='#5b5b5b', label='Prod3B Reference')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel('Distance to True Position / degree')
    if plot_e_reco:
        ax.set_xlabel(r'$E_{Reco} / TeV$')
    else:
        ax.set_xlabel(r'$E_{True} / TeV$')
    ax.legend()

    df = pd.DataFrame({
        'energy_prediction': bin_centers,
        'angular_resolution': b_68,
    })
    return ax, df



def plot_angular_resolution_per_multiplicity(reconstructed_events, reference, plot_e_reco, ax=None):
    df_all = reconstructed_events

    e_min, e_max = 0.02 * u.TeV, 200 * u.TeV
    bins, bin_center, bin_width = make_default_cta_binning(e_min=e_min, e_max=e_max)

    if not ax:
        fig, ax = plt.subplots(1, 1)

    mults = [2, 4, 8, 12, 25]

    for m, color in zip(mults, color_cycle):
        df = df_all.query(f'num_triggered_telescopes == {m}')
        if len(df) < 1000:
            continue
        if plot_e_reco:
            x = df.gamma_energy_prediction_mean.values
        else:
            x = df.mc_energy.values

        distance = calculate_distance_to_true_source_position(df)

        b_68, _, _ = binned_statistic(x, distance, statistic=lambda y: np.nanpercentile(y, 68), bins=bins)

        ax.errorbar(
            bin_center.value,
            b_68,
            xerr=bin_width.value / 2.0,
            linestyle='--',
            color=color,
            ecolor=color,
            ms=0,
            capsize=0,
            label=m,
        )


    if reference:
        df = load_angular_resolution_requirement()
        ax.plot(df.energy, df.resolution, '.', color='#5b5b5b', label='Prod3B Reference')

    ax.set_xscale('log')
    ax.set_ylabel('Distance to True Position / degree')
    if plot_e_reco:
        ax.set_xlabel(r'$E_{Reco} / TeV$')
    else:
        ax.set_xlabel(r'$E_{True} / TeV$')

    ax.legend()
    return ax

