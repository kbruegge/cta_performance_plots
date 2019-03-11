import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.colors import PowerNorm

import numpy as np
import astropy.units as u
from scipy.stats import binned_statistic

from ..colors import default_cmap, main_color
from . import add_colorbar_to_figure
from ..binning import make_default_cta_binning


def plot_impact(reconstructed_events, colormap=default_cmap, color=main_color, ax=None):
    df = reconstructed_events
    x = df.mc_core_x - df.core_x
    y = df.mc_core_y - df.core_y

    x_min = -50
    x_max = 50

    if not ax:
        fig, ax = plt.subplots(1, 1)

    im = ax.hexbin(x, y, extent=(x_min, x_max, x_min, x_max), cmap=colormap, norm=LogNorm())
    add_colorbar_to_figure(im, fig, ax)

    ax.set_ylim([x_min, x_max])
    ax.set_xlim([x_min, x_max])

    ax.set_xlabel('x offset to true impact / meter')
    ax.set_ylabel('y offset to true impact / meter')
    ax.tick_params(axis='x', which='major', pad=7)
    return ax


def plot_impact_distance(reconstructed_events, colormap=default_cmap, color=main_color, ax=None):
    df = reconstructed_events
    distance = np.sqrt((df.mc_core_x - df.core_x)**2 + (df.mc_core_y - df.core_y)**2)

    bins, bin_center, bin_widths = make_default_cta_binning(e_min=0.003 * u.TeV, e_max=330 * u.TeV)

    x = df.mc_energy.values
    y = distance

    b_50, bin_edges, binnumber = binned_statistic(x, y, statistic=np.nanmedian, bins=bins)

    bin_centers = np.sqrt(bin_edges[1:] * bin_edges[:-1])

    log_emin, log_emax = np.log10(bins.min().value), np.log10(bins.max().value)

    if not ax:
        fig, ax = plt.subplots(1, 1)

    im = ax.hexbin(x, y, xscale='log', extent=(log_emin, log_emax, 0, 120), cmap=colormap, norm=PowerNorm(0.5))
    add_colorbar_to_figure(im, fig, ax)

    ax.plot(bin_centers, b_50, lw=2, color=color, label='Median')

    ax.set_xscale('log')
    ax.set_ylabel('Distance to True Position / meter')
    ax.set_xlabel(r'$E_{True} / TeV$')
    ax.legend()
    return ax
