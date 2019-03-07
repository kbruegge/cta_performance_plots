import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as mticker
from matplotlib.colors import LogNorm
import numpy as np
import astropy.units as u
from gammapy.irf import EnergyDispersion2D
from scipy.ndimage import gaussian_filter
from . import _bin_center, _log_tick_formatter


def energy_dispersion_3d_plot(irf_file_path, ax=None, hdu="ENERGY DISPERSION"):
    if not ax:
        fig = plt.figure(figsize=(10, 7),)
        ax = fig.add_subplot(111, projection='3d')


    edisp = EnergyDispersion2D.read(irf_file_path, hdu=hdu)
    energy_reco = np.logspace(-2, 2, 20) * u.TeV
    offsets = np.linspace(0, 6, 7) * u.deg

    Z = []
    for offset in offsets:
        erf = edisp.to_energy_dispersion(offset)
        zs  = erf.get_resolution(energy_reco).value
        Z.append(zs)

    X, Y = np.meshgrid(energy_reco, offsets)
    Z = np.vstack(Z)
    mask = ~np.isfinite(Z)
    Z[mask] = np.nanmean(Z)
    Z = gaussian_filter(Z, sigma=0.8)

    X, Y, Z = np.log10(X.to_value('TeV')).ravel(), Y.ravel(), Z.ravel()
    surf = ax.plot_trisurf(X, Y, Z, cmap='viridis', vmin=0, vmax=np.nanpercentile(Z, 99), antialiased=True)

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_log_tick_formatter))

    return ax

