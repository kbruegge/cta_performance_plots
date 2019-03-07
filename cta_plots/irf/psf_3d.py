import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as mticker
from matplotlib.colors import LogNorm
import numpy as np
import astropy.units as u
from gammapy.irf import PSF3D
from scipy.ndimage import gaussian_filter
from . import _bin_center, _log_tick_formatter, _log_scale_formatter


def psf_3d_plot(irf_file_path, ax=None, hdu="PSF"):
    if not ax:
        fig = plt.figure(figsize=(10, 7),)
        ax = fig.add_subplot(111, projection='3d')

    psf = PSF3D.read(irf_file_path, hdu=hdu)
    energy_reco = np.logspace(-2, 2, 20) * u.TeV
    offsets = np.linspace(0, 6, 7) * u.deg

    d = np.linspace(0.1, 10, 11)[1::2]
    ticks = np.log10(d)
    ticks = np.append(ticks, 1 + ticks)

    X, Y = np.meshgrid(energy_reco, offsets)
    Z = []
    for x in energy_reco:
        zs = psf.containment_radius(x, offsets, fraction=0.68)
        Z.append(zs)

    Z = np.vstack(Z).T
    surf = ax.plot_surface(np.log10(X.to_value('TeV')), Y, np.log10(Z), cmap='viridis',  linewidth=1, antialiased=True)

    ax.xaxis.set_major_locator(mticker.FixedLocator([-2, -1, 0, 1, 2 ]))
    ax.zaxis.set_major_locator(mticker.FixedLocator(ticks))

    ax.zaxis.set_major_formatter(mticker.FuncFormatter(_log_scale_formatter))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_log_tick_formatter))
    ax.set_zlim([-1 ,1])
    ax.set_ylabel('Offset in FoV / deg')
    ax.set_zlabel('Angular Resolution / deg')
    ax.set_xlabel('Reconstructed Energy / TeV')
    return ax

