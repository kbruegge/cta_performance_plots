import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

from . import load_sensitivity_reference, load_sensitivity_requirement
from cta_plots.spectrum import CrabLogParabola, CrabSpectrum


def plot_crab_flux(bin_edges, ax=None, curved=True, show_text=True):
    if curved:
        crab = CrabLogParabola()
    else:
        crab = CrabSpectrum()
    if not ax:
        ax = plt.gca()
    e = np.logspace(-3, 3, 300) * u.TeV
    flux = (crab.flux(e) * e ** 2).to_value(
        u.erg / (u.s * u.cm ** 2)
    )
    ax.plot(
        e, flux, ls='--', color='#a3a3a3', label='Crab Flux', lw=0.7, alpha=0.9
    )
    ax.plot(
        e, 0.1 * flux, ls='--', color='#a3a3a3', lw=0.7, alpha=0.75
    )
    ax.plot(
        e, 0.01 * flux, ls='--', color='#a3a3a3', lw=0.7, alpha=0.6
    )

    if show_text:
        y = 1.2 * flux[len(flux) // 2] 
        x = 1
        ax.text(x, y, '100 \\% Crab', color='gray', alpha=0.7, rotation=-20, size=5, ha='center', va='center')

        y = 0.12 * flux[len(flux) // 2] 
        x = 1
        ax.text(x, y, '10 \\% Crab', color='gray', alpha=0.6, rotation=-20, size=5, ha='center', va='center')

        y = 0.012 * flux[len(flux) // 2]
        x = 1
        ax.text(x, y, '1 \\% Crab', color='gray', alpha=0.5, rotation=-20, size=5, ha='center', va='center')
    return ax


def plot_requirement(ax=None):
    from scipy.interpolate import interp1d
    
    df = load_sensitivity_requirement()
    if not ax:
        ax = plt.gca()
    # f = interp1d(df.energy, df.sensitivity, kind='quadratic')
    # min_e, max_e = df.energy.min(), df.energy.max()
    # x = np.logspace(np.log10(min_e), np.log10(max_e), 300)
    # ax.plot(x, f(x), color='#888888', lw=1.2, label='CTA Requirement', alpha=0.8)
    ax.plot(df.energy, df.sensitivity, color='#888888', lw=1.2, label='CTA Requirement', alpha=0.8)
    # ax.plot(df.energy, df.sensitivity * 3, color='#bebebe', lw=0.5, label='Requirement Real Time')
    return ax


def plot_reference(ax=None):
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


def plot_sensitivity(rs, bin_edges, bin_center, color='blue', ax=None, **kwargs):
    crab = CrabLogParabola()
    
    sensitivity = rs.sensitivity.values * (crab.flux(bin_center) * bin_center ** 2).to_value(
        u.erg / (u.s * u.cm ** 2)
    )
    sensitivity_low = rs.sensitivity_low.values * (crab.flux(bin_center) * bin_center ** 2).to_value(
        u.erg / (u.s * u.cm ** 2)
    )
    sensitivity_high = rs.sensitivity_high.values * (crab.flux(bin_center) * bin_center ** 2).to_value(
        u.erg / (u.s * u.cm ** 2)
    )
    xerr = [np.abs(bin_edges[:-1] - bin_center).to_value('TeV'), np.abs(bin_edges[1:] - bin_center).to_value('TeV')]
    yerr = [np.abs(sensitivity - sensitivity_low), np.abs(sensitivity - sensitivity_high)]

    if not ax:
        ax = plt.gca()

    m = (rs['valid'] == True)

    werr = [xerr[0][m], xerr[1][m]] 
    serr = [yerr[0][m], yerr[1][m]] 
    ax.errorbar(
        bin_center[m].to_value('TeV'), sensitivity[m], xerr=werr, yerr=serr, linestyle='', ecolor=color, zorder=20, **kwargs
    )

    werr = [xerr[0][~m], xerr[1][~m]] 
    serr = [yerr[0][~m], yerr[1][~m]] 
    ax.errorbar(
        bin_center[~m].to_value('TeV'), sensitivity[~m], xerr=werr, yerr=serr, linestyle='', ecolor='gray', alpha=0.15, zorder=20, **kwargs
    )
    ax.errorbar(
        bin_center[~m].to_value('TeV'), sensitivity[~m], xerr=werr, yerr=serr, linestyle='', ecolor=color, alpha=0.3, zorder=20, **kwargs
    )

    return ax