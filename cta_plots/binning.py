import astropy.units as u
import numpy as np


def make_energy_bins(energies=None, e_min=None, e_max=None, bins=10, centering='linear'):
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


def make_default_cta_binning(e_min=0.02 * u.TeV, e_max=200 * u.TeV, centering='log', overflow=False, bins_per_decade=5):

    bin_edges = np.logspace(np.log10(0.002), np.log10(2000), 6 * bins_per_decade + 1)
    idx = np.searchsorted(bin_edges, [e_min.to_value(u.TeV), e_max.to_value(u.TeV)])
    max_idx = min(idx[1] + 1, len(bin_edges) - 1)
    bin_edges = bin_edges[idx[0]:max_idx]
    if overflow:
        bin_edges = np.append(bin_edges, 10000)
        bin_edges = np.append(0, bin_edges)

    if centering == 'log':
        bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    else:
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_widths = np.diff(bin_edges)

    return bin_edges * u.TeV, bin_centers * u.TeV, bin_widths * u.TeV


def make_energy_bins_per_decade(e_min, e_max, n_bins_per_decade=10, overflow=False, centering='linear'):
    bin_edges = np.logspace(-3, 3, (6 * n_bins_per_decade) + 1)
    
    idx = np.searchsorted(bin_edges, [e_min.to_value(u.TeV), e_max.to_value(u.TeV)])
    bin_edges = bin_edges[idx[0]:idx[1]]
    if overflow:
        bin_edges = np.append(bin_edges, 10000)
        bin_edges = np.append(0, bin_edges)

    bin_edges *= u.TeV

    if centering == 'log':
        bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    else:
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_widths = np.diff(bin_edges)

    return bin_edges, bin_centers, bin_widths

