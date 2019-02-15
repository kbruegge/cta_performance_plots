import click
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import pandas as pd
from fact.analysis import li_ma_significance
from scipy.optimize import brute
from .coordinate_utils import calculate_distance_to_true_source_position, calculate_distance_to_point_source
from .spectrum import MCSpectrum, CrabSpectrum, CosmicRaySpectrum
from tqdm import tqdm
from itertools import product
from colorama import Fore
from joblib import Parallel, delayed
import matplotlib.offsetbox as offsetbox



@click.command()
@click.argument('gammas_dl3', type=click.Path(exists=True))
@click.argument('protons_dl3', type=click.Path(exists=True))
@click.option('-o', '--output', type=click.Path(exists=False))
def main(gammas_dl3, protons_dl3, output):

    t_obs = 5 * u.min

    gammas = pd.read_hdf(gammas_dl3, key='array_events')

    gamma_runs = pd.read_hdf(gammas_dl3, key='runs')
    mc_production_gamma = MCSpectrum.from_cta_runs(gamma_runs)

    protons = pd.read_hdf(protons_dl3, key='array_events')
    if (gamma_runs.mc_diffuse == 1).any():
        print(Fore.RED + 'Need point-like gammas to do theta square plot')
        print(Fore.RESET)
        return -1

    source_az = gammas.mc_az.iloc[0] * u.deg
    source_alt = gammas.mc_alt.iloc[0] * u.deg

    gammas['theta'] = calculate_distance_to_point_source(gammas, source_alt=source_alt, source_az=source_az).to(u.deg).value
    protons['theta'] = calculate_distance_to_point_source(protons, source_alt=source_alt, source_az=source_az).to(u.deg).value

    proton_runs = pd.read_hdf(protons_dl3, key='runs')
    mc_production_proton = MCSpectrum.from_cta_runs(proton_runs)

    crab = CrabSpectrum()
    cosmic = CosmicRaySpectrum()

    gammas['weight'] = mc_production_gamma.reweigh_to_other_spectrum(crab, gammas.mc_energy.values * u.TeV, t_assumed_obs=t_obs)
    protons['weight'] = mc_production_proton.reweigh_to_other_spectrum(cosmic, protons.mc_energy.values * u.TeV, t_assumed_obs=t_obs)

    theta_square_cuts = np.arange(0.01, 0.35, 0.01)
    prediction_cuts = np.arange(0.0, 1, 0.05)  
    iterator = tqdm(product(theta_square_cuts, prediction_cuts), total=len(theta_square_cuts) * len(prediction_cuts))

    rs = Parallel(n_jobs=-4, pre_dispatch='3*n_jobs')(delayed(calculate_significance)(gammas, protons, t, p) for t, p in iterator)
    max_index = np.argmax([r[0] for r in rs])
    best_significance, best_theta_square_cut, best_prediction_cut = rs[max_index]
    
    gammas_gammalike = gammas.query(f'gamma_prediction_mean > {best_prediction_cut}').copy()
    protons_gammalike = protons.query(f'gamma_prediction_mean > {best_prediction_cut}').copy()

    
    on = gammas_gammalike
    off = protons_gammalike

    bins = np.arange(0, 0.6, 0.01)
    h_off, _ = np.histogram(off['theta']**2, bins=bins, weights=off.weight)
    h_on, _ = np.histogram(on['theta']**2, bins=bins, weights=on.weight)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.step(bins[:-1], h_on + h_off.mean(), where='post', label='on events')
    ax.step(bins[:-1], h_off, where='post', label='off events')
    ax.set_ylim([0, max(h_on + h_off) * 1.18])
    ax.axhline(y=h_off.mean(), color='C1', lw=1, alpha=0.7)
    ax.axvline(x=best_theta_square_cut, color='gray', lw=1, alpha=0.7)

    textstr = '\n'.join([
        f'Observation Time: {t_obs}',
        f'Prediction Threshold: {best_prediction_cut:.2f}',
        f'Theta Square Cut: {(best_theta_square_cut):.2f}',
        f'Significance: {best_significance:.2f}',
    ])

    # these are matplotlib.patch.Patch properties
    # place a text box in upper left in axes coords
    ob = offsetbox.AnchoredText(textstr, loc=1, prop={'fontsize': 9})
    ob.patch.set_facecolor('lightgray')
    ob.patch.set_alpha(0.1)
    ax.add_artist(ob)

    # plt.text(best_theta_cut**2 + 0.05, max(h_on + h_off + 100), textstr, fontsize=10, verticalalignment='top', bbox=props)

    # plt.legend()

    # s = calculate_significance(gammas, protons, best_theta_cut, best_prediction_cut)
    # plt.title(f'Significane is {s}')

    # sens = relative_sensitivity(n_on, n_off, alpha=1)
    # rate = crab._integral(e_min, e_max)
    # print(f'Integral sensitivity {rate.to("1/(cm^2 s)") * sens}, relative flux {sens}')

    if output:
        plt.savefig(output)
    else:
        plt.show()

def calculate_significance(gammas, protons, theta_square_cut, prediction_cut):
    gammas_gammalike = gammas.query(f'gamma_prediction_mean >= {prediction_cut}').copy()
    protons_gammalike = protons.query(f'gamma_prediction_mean >= {prediction_cut}').copy()
    off_bins = np.arange(0, 0.5, np.sqrt(theta_square_cut))
    h, _ = np.histogram(protons_gammalike['theta']**2, bins=off_bins, weights=protons_gammalike.weight)
    n_off = h.mean()

    n_on = gammas_gammalike.query(f'theta <= {np.sqrt(theta_square_cut)}').weight.sum() + n_off

    return li_ma_significance(n_on, n_off, alpha=1), theta_square_cut, prediction_cut


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()
