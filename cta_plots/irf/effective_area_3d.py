import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as mticker
from matplotlib.colors import LogNorm
import numpy as np
import astropy.units as u
from gammapy.irf import EffectiveAreaTable2D
from scipy.ndimage import gaussian_filter
from . import _bin_center, _log_tick_formatter

def effective_area_3d_plot(irf_file_path, ax=None, hdu="EFFECTIVE AREA"):
    if not ax:
        fig = plt.figure(figsize=(10, 7),)
        ax = fig.add_subplot(111, projection='3d')

    energy_reco = np.logspace(-2, 2, 20) * u.TeV
    
    offsets = np.linspace(0, 6, 13) * u.deg

    aeff = EffectiveAreaTable2D.read(irf_file_path, hdu=hdu)

    X, Y = np.meshgrid(_bin_center(energy_reco), offsets)

    Z = []
    for offset in offsets: 
        t = aeff.to_effective_area_table(offset=offset, energy=energy_reco)
        Z.append(t.data.data)

    Z = np.vstack(Z).value * u.m**2

    X, Y, Z = np.log10(X.to_value('TeV')).ravel(), Y.ravel(), Z.to('km2').ravel()
    surf = ax.plot_trisurf(X, Y, Z, cmap='viridis', vmin=0, vmax=np.nanpercentile(Z, 99), antialiased=True)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_log_tick_formatter))
    ax.view_init(15, -140)

    ax.set_ylabel('Offset in FoV / deg')
    ax.set_zlabel('Effective Area / $km^2$')
    ax.set_xlabel('Reconstructed Energy / TeV')
    return ax

