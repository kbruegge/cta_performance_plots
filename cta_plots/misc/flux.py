import click
import matplotlib.pyplot as plt
from cta_plots.mc.spectrum import CTAElectronSpectrum, CosmicRaySpectrumPDG, CTAProtonSpectrum,  CrabSpectrum
import astropy.units as u
from cta_plots.colors import color_cycle
import numpy as np

@click.command()
@click.option('-o', '--output', type=click.Path(exists=False))
@click.option('-p', '--power', default=0, help='exponent by which to transform the fluxes')
def main(output, power):
    
    energies = np.logspace(-2, 2, 200) * u.TeV

    spectrum = CTAElectronSpectrum()
    flux = spectrum.flux(energies)
    y = energies**power * flux
    plt.plot(energies, y, color=next(color_cycle), label='Electrons')
    
    spectrum = CosmicRaySpectrumPDG()
    flux = spectrum.flux(energies)
    y = energies**power * flux
    plt.plot(energies, y, color=next(color_cycle), label='Cosmic Ray PDG')
    
    spectrum = CTAProtonSpectrum()
    flux = spectrum.flux(energies)
    y = energies**power * flux
    plt.plot(energies, y, color=next(color_cycle), label='CTA Proton Spectrum')

    spectrum = CrabSpectrum()
    flux = spectrum.flux(energies)
    y = energies**power * flux
    plt.plot(energies, y, color=next(color_cycle), label='Crab Spectrum')
    

    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel(f'Energy / {energies.unit}')
    plt.ylabel(f'Flux / {y.unit}')
    plt.legend()
    # from IPython import embed; embed()
    if output:
        plt.savefig(output)
    else:
        plt.show()


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()