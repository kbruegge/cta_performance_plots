import click
import matplotlib.pyplot as plt
from cta_plots.mc.spectrum import CTAElectronSpectrum, CosmicRaySpectrumPDG, CTAProtonSpectrum,  CrabSpectrum
import astropy.units as u
from cta_plots.colors import color_cycle
from cta_plots import make_energy_bins
import numpy as np

@click.command()
@click.option('-o', '--output', type=click.Path(exists=False))
@click.option('-n', '--n_bins', type=int, default=40)
def main(output, n_bins):
    
    bins, _, _ = make_energy_bins(e_min=0.008 * u.TeV, e_max=200 * u.TeV, bins=n_bins)

    spectrum = CTAElectronSpectrum()
    counts = spectrum.expected_events_for_bins(area=1*u.km**2, t_obs=1*u.h, energy_bins=bins, solid_angle=5*u.deg)
    plt.step(bins[0:-1], counts, where='post', color=next(color_cycle), label='Electrons')

    
    spectrum = CosmicRaySpectrumPDG()
    counts = spectrum.expected_events_for_bins(area=1*u.km**2, t_obs=1*u.h, energy_bins=bins, solid_angle=5*u.deg)
    plt.step(bins[0:-1], counts, where='post', color=next(color_cycle), label='Cosmic Rays PDG')
    
    spectrum = CTAProtonSpectrum()
    counts = spectrum.expected_events_for_bins(area=1*u.km**2, t_obs=1*u.h, energy_bins=bins, solid_angle=5*u.deg)
    plt.step(bins[0:-1], counts, where='post', color=next(color_cycle), label='CTA Proton Spectrum')
    
    spectrum = CrabSpectrum()
    counts = spectrum.expected_events_for_bins(area=1*u.km**2, t_obs=1*u.h, energy_bins=bins, solid_angle=5*u.deg)
    plt.step(bins[0:-1], counts, where='post', color=next(color_cycle), label='Crab Nebula')

    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel(f'Energy / {bins.unit}')
    plt.ylabel(f'Counts')
    plt.legend()

    if output:
        plt.savefig(output)
    else:
        plt.show()


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()