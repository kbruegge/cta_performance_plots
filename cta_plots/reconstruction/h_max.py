import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from scipy.stats import binned_statistic

from .. import add_colorbar_to_figure
from ..binning import make_default_cta_binning

from matplotlib.colors import PowerNorm
from ..colors import default_cmap, main_color
from ctapipe.instrument import get_atmosphere_profile_functions



def plot_h_max_distance(reconstructed_events, site='paranal', colormap=default_cmap, color=main_color, ax=None):
    df = reconstructed_events
    df = df.loc[df.mc_x_max > 0]
    thickness, altitude = get_atmosphere_profile_functions(site)

    mc_h_max = altitude(df.mc_x_max.values * u.Unit('g/cm^2')).value
    y = mc_h_max - df.h_max

    bins, bin_center, bin_widths = make_default_cta_binning(e_min=0.003 * u.TeV, e_max=330 * u.TeV)
    x = df.mc_energy.values

    b_50, bin_edges, binnumber = binned_statistic(x, y, statistic='median', bins=bins)

    bin_centers = np.sqrt(bin_edges[1:] * bin_edges[:-1])

    log_emin, log_emax = np.log10(bins.min().value), np.log10(bins.max().value)

    if not ax:
        fig, ax = plt.subplots(1, 1)
    

    im = ax.hexbin(x, y, xscale='log', extent=(log_emin, log_emax, 0.01, 3000), cmap=colormap, norm=PowerNorm(0.5))
    add_colorbar_to_figure(im, fig, ax)
    ax.plot(bin_centers, b_50, lw=2, color=color, label='Median')

    ax.set_xscale('log')
    ax.set_ylabel('Distance to true H max  / meter')
    ax.set_xlabel(r'$E_{True} / TeV$')
    ax.set_xlim([0.007, 300])
    return ax


def plot_h_max(reconstructed_events, site='paranal', colormap=default_cmap, color=main_color, ax=None):
    df = reconstructed_events
    df = df.loc[df.mc_x_max > 0]
    thickness, altitude = get_atmosphere_profile_functions(site)

    mc_h_max = altitude(df.mc_x_max.values * u.Unit('g/cm^2')).value

    bins, bin_center, bin_widths = make_default_cta_binning(e_min=0.01 * u.TeV, e_max=200 * u.TeV,)
    x = df.mc_energy.values

    b_50, bin_edges, binnumber = binned_statistic(x, df.h_max, statistic='median', bins=bins)
    b_84, _, _ = binned_statistic(x, df.h_max, statistic=lambda x: np.percentile(x, q=[84]), bins=bins)
    b_16, _, _ = binned_statistic(x, df.h_max, statistic=lambda x: np.percentile(x, q=[16]), bins=bins)

    log_emin, log_emax = np.log10(0.007), np.log10(300)

    if not ax:
        fig, ax = plt.subplots(1, 1)

    im = ax.hexbin(x, mc_h_max, xscale='log', extent=(log_emin, log_emax, 0, 17500), cmap=colormap, norm=PowerNorm(0.5))
    add_colorbar_to_figure(im, fig, ax, label='Counts')

    # ax.step(bins[:-1], b_50, lw=2, color=color, label='Median Prediction', where='post')
    ax.hlines(b_50[:-1], bins[:-2], bins[1:-1], lw=2, color=color, label='Median Prediction')
    # hardcore fix for stupid step plotting artifact
    # b_16[-1] = b_16[-2]
    # b_50[-1] = b_50[-2]
    # b_84[-1] = b_84[-2]
    ax.fill_between(bins[:-1], b_16, b_84, alpha=0.3, color=color, step='post')

    ax.set_xscale('log')
    ax.set_ylabel('Max Height / m')
    ax.set_xlabel('True Energy / TeV')
    ax.legend(framealpha=0.0)
    ax.set_xlim([0.007, 300])
    plt.tight_layout(pad=0, rect=(0, 0, 1.003, 1))
    return ax