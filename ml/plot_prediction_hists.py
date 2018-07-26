import click
import numpy as np
import matplotlib.pyplot as plt
import fact.io
from cycler import cycler


@click.command()
@click.argument(
    'predicted_gammas', type=click.Path(
        exists=True,
        dir_okay=False,
    ))
@click.argument(
    'predicted_protons', type=click.Path(
        exists=True,
        dir_okay=False,
    ))
@click.option(
    '-o', '--output_file', type=click.Path(
        exists=False,
        dir_okay=False,
    ))
@click.option('-w', '--what', default='mean', type=click.Choice(['per-telescope', 'mean', 'single', 'median', 'weighted-mean', 'min', 'max', 'brightest']))
def main(predicted_gammas, predicted_protons, output_file, what):
    bins = np.linspace(0, 1, 100)

    if what == 'mean':
        cols = ['gamma_prediction_mean']
        gammas = fact.io.read_data(predicted_gammas, key='array_events', columns=cols).dropna()
        protons = fact.io.read_data(predicted_protons, key='array_events', columns=cols).dropna()

        fig, ax = plt.subplots(1)
        ax.hist(gammas.gamma_prediction_mean.values, bins=bins, histtype='step', density=True, linewidth=2)
        ax.hist(protons.gamma_prediction_mean.values, bins=bins, histtype='step', density=True, linewidth=2, color='gray')


    if what == 'weighted-mean':
        cols = ['gamma_prediction', 'array_event_id', 'run_id', 'intensity']
        gammas = fact.io.read_data(predicted_gammas, key='telescope_events', columns=cols).dropna()
        protons = fact.io.read_data(predicted_protons, key='telescope_events', columns=cols).dropna()

        gammas['weight'] = np.log10(gammas.intensity)
        gammas['weighted_prediction'] = gammas.gamma_prediction * gammas.weight
        group = gammas.groupby(['array_event_id', 'run_id'])
        gw = group['weighted_prediction'].sum() / group.weight.sum()

        protons['weight'] = np.log10(protons.intensity)
        protons['weighted_prediction'] = protons.gamma_prediction * protons.weight
        group = protons.groupby(['array_event_id', 'run_id'])
        pw = group['weighted_prediction'].sum() / group.weight.sum()


        fig, ax = plt.subplots(1)
        ax.hist(gw, bins=bins, histtype='step', density=True, linewidth=2)
        ax.hist(pw, bins=bins, histtype='step', density=True, linewidth=2, color='gray')

    elif what == 'min':
        cols = ['gamma_prediction', 'array_event_id', 'run_id']
        gammas = fact.io.read_data(predicted_gammas, key='telescope_events', columns=cols).dropna()
        protons = fact.io.read_data(predicted_protons, key='telescope_events', columns=cols).dropna()

        g = gammas.groupby(['array_event_id', 'run_id'])['gamma_prediction'].min()
        p = protons.groupby(['array_event_id', 'run_id'])['gamma_prediction'].min()

        fig, ax = plt.subplots(1)
        ax.hist(g, bins=bins, histtype='step', density=True, linewidth=2)
        ax.hist(p, bins=bins, histtype='step', density=True, linewidth=2)

    elif what == 'max':
        cols = ['gamma_prediction', 'array_event_id', 'run_id']
        gammas = fact.io.read_data(predicted_gammas, key='telescope_events', columns=cols).dropna()
        protons = fact.io.read_data(predicted_protons, key='telescope_events', columns=cols).dropna()

        g = gammas.groupby(['array_event_id', 'run_id'], sort=False)['gamma_prediction'].max()
        p = protons.groupby(['array_event_id', 'run_id'], sort=False)['gamma_prediction'].max()

        fig, ax = plt.subplots(1)
        ax.hist(g, bins=bins, histtype='step', density=True, linewidth=2)
        ax.hist(p, bins=bins, histtype='step', density=True, linewidth=2)


    elif what == 'brightest':
        cols = ['gamma_prediction', 'array_event_id', 'run_id', 'intensity']
        gammas = fact.io.read_data(predicted_gammas, key='telescope_events', columns=cols).dropna()
        protons = fact.io.read_data(predicted_protons, key='telescope_events', columns=cols).dropna()

        idx = gammas.groupby(['array_event_id', 'run_id'], sort=False)['intensity'].idxmax()
        g = gammas.loc[idx].gamma_prediction.values

        idx = protons.groupby(['array_event_id', 'run_id'], sort=False)['intensity'].idxmax()
        p = protons.loc[idx].gamma_prediction.values

        fig, ax = plt.subplots(1)
        ax.hist(g, bins=bins, histtype='step', density=True, linewidth=2)
        ax.hist(p, bins=bins, histtype='step', density=True, linewidth=2)


    elif what == 'median':
        cols = ['gamma_prediction', 'array_event_id', 'run_id']
        gammas = fact.io.read_data(predicted_gammas, key='telescope_events', columns=cols).dropna()
        protons = fact.io.read_data(predicted_protons, key='telescope_events', columns=cols).dropna()

        g = gammas.groupby(['array_event_id', 'run_id'])['gamma_prediction'].median()
        p = protons.groupby(['array_event_id', 'run_id'])['gamma_prediction'].median()

        fig, ax = plt.subplots(1)
        ax.hist(g, bins=bins, histtype='step', density=True, linewidth=2)
        ax.hist(p, bins=bins, histtype='step', density=True, linewidth=2)

    elif what == 'single':
        cols = ['gamma_prediction']
        gammas = fact.io.read_data(predicted_gammas, key='telescope_events', columns=cols).dropna()
        protons = fact.io.read_data(predicted_protons, key='telescope_events', columns=cols).dropna()

        fig, ax = plt.subplots(1)
        ax.hist(gammas.gamma_prediction.values, bins=bins, histtype='step', density=True, linewidth=2)
        ax.hist(protons.gamma_prediction.values, bins=bins, histtype='step', density=True, linewidth=2)

    elif what == 'per-telescope':
        cols = ['telescope_type_name', 'gamma_prediction']
        gammas = fact.io.read_data(predicted_gammas, key='telescope_events', columns=cols).dropna()
        protons = fact.io.read_data(predicted_protons, key='telescope_events', columns=cols).dropna()

        fig, ax = plt.subplots(1)
        for name, group in gammas.groupby('telescope_type_name', sort=False):
            ax.hist(group.gamma_prediction.values, bins=bins, label=f'gamma prediction {name}', histtype='step', density=True, linewidth=2)

        color_cycle = cycler(color=['gray', 'darkgray', 'black'])
        ax.set_prop_cycle(color_cycle)
        for name, group in protons.groupby('telescope_type_name', sort=False):
            ax.hist(group.gamma_prediction.values, bins=bins, label=f'proton prediction {name}', histtype='step', density=True)

        plt.legend(loc='upper left')

    if output_file:
        plt.savefig(output_file)
    else:
        plt.show()


if __name__ == '__main__':
    main()
