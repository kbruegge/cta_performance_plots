import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as mticker
from matplotlib.colors import LogNorm
import numpy as np
import astropy.units as u
from gammapy.irf import Background3D
from scipy.ndimage import gaussian_filter
from . import _bin_center, _log_data, _log_tick_formatter


def background_3d_plot(irf_file_path, ax=None, hdu="BACKGROUND"):
    if not ax:
        fig = plt.figure(figsize=(10, 7),)
        ax = fig.add_subplot(111, projection='3d')
    
    bkg3d = Background3D.read(irf_file_path, hdu=hdu)

    energy_reco = np.logspace(-2, 2, 20) * u.TeV
    offset_centers = _bin_center(bkg3d.data.axes[1].bins)

    data = bkg3d.data.data.sum(axis=0).value
    data = _log_data(data)
    
    vmin, vmax = -4, 1.5
    data = np.clip(data, vmin, vmax)
    
    X, Y = np.meshgrid(offset_centers, offset_centers)

    X, Y, Z, = X.ravel(), Y.ravel(), gaussian_filter(data, sigma=0.8).ravel()
    surf = ax.plot_trisurf(X, Y, Z, cmap='viridis', antialiased=True)

    ax.zaxis.set_major_locator(mticker.FixedLocator([-4, -3, -2, -1, 0, 1,]))
    ax.zaxis.set_major_formatter(mticker.FuncFormatter(_log_tick_formatter))
    ax.set_zlim([-4 ,1])

    return ax

