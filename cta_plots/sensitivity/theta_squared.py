import click
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

import matplotlib.offsetbox as offsetbox
from cta_plots import load_signal_events, load_background_events, ELECTRON_TYPE
from cta_plots.sensitivity.optimize import find_best_cuts


@click.command()
@click.argument('gammas_path', type=click.Path(exists=True))
@click.argument('protons_path', type=click.Path(exists=True))
@click.argument('electrons_path', type=click.Path(exists=True))
@click.option('-o', '--output', type=click.Path(exists=False))
@click.option('-j', '--n_jobs', default=4)
def main(gammas_path, protons_path, electrons_path, output, n_jobs):

    t_obs = 1 * u.min

    gammas, source_alt, source_az = load_signal_events(gammas_path, assumed_obs_time=t_obs)
    background = load_background_events(protons_path, electrons_path, source_alt, source_az, assumed_obs_time=t_obs,)

    theta_uts = np.arange(0.01, 0.35, 0.01)
    prediction_cuts = np.arange(0.0, 1, 0.05)
    multiplicities = [2, 3, 4, 5, 6, 7, 8]

    # theta_uts = np.arange(0.01, 0.35, 0.2)
    # prediction_cuts = np.arange(0.0, 1, 0.25)
    # multiplicities = [2]

    best_sensitivity, best_prediction_cut, best_theta_cut, best_significance, best_mult = find_best_cuts(
        theta_uts, prediction_cuts, multiplicities, gammas, background, alpha=1, criterion='significance', n_jobs=n_jobs
    )

    gammas_gammalike = gammas.query(f'gamma_prediction_mean > {best_prediction_cut}').copy()
    background_gammalike = background.query(f'gamma_prediction_mean > {best_prediction_cut}').copy()
    on = gammas_gammalike
    off = background_gammalike

    bins = np.arange(0, 0.5, 0.01)
    h_off, _ = np.histogram(off['theta'] ** 2, bins=bins, weights=off.weight)
    h_on, _ = np.histogram(on['theta'] ** 2, bins=bins, weights=on.weight)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.step(bins[:-1], h_on + h_off.mean(), where='post', label='on events')
    ax.step(bins[:-1], h_off, where='post', label='off events protons', color='black', alpha=0.7)
    ax.fill_between(bins[:-1], h_off, step='post', alpha=0.4, color='gray')


    off_electrons = off.query(f'type == {ELECTRON_TYPE}')
    h_off_electrons, _ = np.histogram(
        off_electrons['theta'] ** 2, bins=bins, weights=off_electrons.weight
    )
    ax.step(bins[:-1], h_off_electrons, where='post', label='off events electrons', color='gray')

    ax.set_ylim([0, max(h_on + h_off) * 1.18])
    # ax.axhline(y=h_off.mean(), color='C1', lw=1, alpha=0.7)
    ax.axvline(x=best_theta_cut**2, color='gray', lw=1, alpha=0.7)

    header = '\\begin{tabular}{l@{\hskip 0.2in}l}'
    titlestr = f'\\multicolumn{2}{{l}}{{\\textbf{{Source Detected with {best_significance:.1f}$\\sigma$ }}}} \\\\'
    textstr = '\\\\'.join(
        [
            f'Observation Time & \\SI{{{t_obs.to_value(u.s)}}}{{\\second}}',
            f'Prediction Threshold & {best_prediction_cut:.2f}',
            f'Radius & \\SI{{{(best_theta_cut):.2f}}}{{\\degree}}',
            f'Multiplicity & {int(best_mult)}',
        ]
    )
    footer = '\\end{tabular}'
    
    text = header + titlestr + textstr + footer
    ob = offsetbox.AnchoredText(text, loc=1, prop={'fontsize': 9, 'alpha': 0.5})
    # ob.patch.set_facecolor('lightgray')
    ob.patch.set_alpha(0.0)
    ax.add_artist(ob)

    ax.set_xlabel('$\\theta^2$')
    ax.set_ylabel('Counts')

    ax.set_xlim([0, 0.42])
        
    plt.tight_layout(pad=0)

    if output:
        plt.savefig(output)
    else:
        plt.show()


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()
