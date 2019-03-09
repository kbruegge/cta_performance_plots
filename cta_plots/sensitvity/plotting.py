from io import BytesIO

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pkg_resources import resource_string

from cta_plots.mc.spectrum import CrabLogParabola, CrabSpectrum


def load_sensitivity_reference():
    path = '/ascii/CTA-Performance-prod3b-v1-South-20deg-50h-DiffSens.txt'
    r = resource_string('cta_plots.resources', path)
    df = pd.read_csv(
        BytesIO(r), delimiter='\t\t', skiprows=10, names=['e_min', 'e_max', 'sensitivity'], engine='python'
    )
    return df


def load_sensitivity_requirement():
    path = 'sensitivity_requirement_south_50.txt'
    r = resource_string('cta_plots.resources', path)
    df = pd.read_csv(
        BytesIO(r),
        delim_whitespace=True,
        names=['log_energy', 'sensitivity'],
        index_col=False,
        engine='python',
    )
    df['energy'] = 10 ** df.log_energy
    return df


def plot_crab_flux(bin_edges, ax=None, curved=True):
    if curved:
        crab = CrabLogParabola()
    else:
        crab = CrabSpectrum()
    if not ax:
        ax = plt.gca()
    ax.plot(
        bin_edges, crab.flux(bin_edges) * bin_edges ** 2, ls=':', lw=1, color='#a3a3a3', label='Crab Flux'
    )
    return ax


def plot_requirement(ax=None):
    df = load_sensitivity_requirement()
    if not ax:
        ax = plt.gca()
    ax.plot(df.energy, df.sensitivity, color='#888888', lw=1.2, label='Requirement Offline')
    ax.plot(df.energy, df.sensitivity * 3, color='#bebebe', lw=0.5, label='Requirement Real Time')
    return ax


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

def plot_sensitivity(rs, bin_edges, bin_center, color='blue', ax=None, **kwargs):
    crab = CrabSpectrum()
    sensitivity = rs.sensitivity.values * (crab.flux(bin_center) * bin_center ** 2).to(
        u.erg / (u.s * u.cm ** 2)
    )
    sensitivity_low = rs.sensitivity_low.values * (crab.flux(bin_center) * bin_center ** 2).to(
        u.erg / (u.s * u.cm ** 2)
    )
    sensitivity_high = rs.sensitivity_high.values * (crab.flux(bin_center) * bin_center ** 2).to(
        u.erg / (u.s * u.cm ** 2)
    )
    xerr = [np.abs(bin_edges[:-1] - bin_center).value, np.abs(bin_edges[1:] - bin_center).value]
    yerr = [np.abs(sensitivity - sensitivity_low).value, np.abs(sensitivity - sensitivity_high).value]

    if not ax:
        ax = plt.gca()
    ax.errorbar(
        bin_center.value, sensitivity.value, xerr=xerr, yerr=yerr, linestyle='', ecolor=color, **kwargs
    )
    return ax