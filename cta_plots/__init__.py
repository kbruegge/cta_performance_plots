import numpy as np
import astropy.units as u
from pkg_resources import resource_string
import pandas as pd
from io import BytesIO


def load_energy_resolution_requirement(site='paranal'):
    path = 'ascii/CTA-Performance-prod3b-v1-South-20deg-50h-Eres.txt'
    r = resource_string('cta_plots.resources', path)
    df = pd.read_csv(BytesIO(r), delimiter='\t\t', skiprows=11, names=['energy', 'resolution'], engine='python')
    return df


def load_angular_resolution_requirement(site='paranal'):
    path = 'ascii/CTA-Performance-prod3b-v1-South-20deg-50h-Angres.txt'
    r = resource_string('cta_plots.resources', path)
    df = pd.read_csv(BytesIO(r), delimiter='\t\t', skiprows=11, names=['energy', 'resolution'], engine='python')
    return df


def load_effective_area_requirement(site='paranal'):
    path = 'ascii/CTA-Performance-prod3b-v1-South-20deg-50h-EffAreaNoDirectionCut.txt'
    r = resource_string('cta_plots.resources', path)
    df = pd.read_csv(BytesIO(r), delimiter='\t\t', skiprows=11, names=['energy', 'effective_area'], engine='python')
    return df


@u.quantity_input(energies=u.TeV, e_min=u.TeV, e_max=u.TeV)
def make_energy_bins(
        energies=None,
        e_min=None,
        e_max=None,
        bins=10,
        centering='linear',
):
    if energies is not None and len(energies) >= 2:
        e_min = min(energies)
        e_max = max(energies)

    unit = e_min.unit

    low = np.log10(e_min.value)
    high = np.log10(e_max.value)
    bin_edges = np.logspace(low, high, endpoint=True, num=bins + 1) * unit

    if centering == 'log':
        bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    else:
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    bin_widths = np.diff(bin_edges)

    return bin_edges, bin_centers, bin_widths
