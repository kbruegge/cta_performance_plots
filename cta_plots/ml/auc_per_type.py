import click
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.metrics import roc_curve, roc_auc_score
import fact.io
from ..colors import telescope_color


def add_rectangles(ax, offset=0.1):

    kwargs = {
        'linewidth': 1,
        'edgecolor': 'white',
        'facecolor': 'white',
        'alpha': 0.6,
    }

    # left
    rect = patches.Rectangle((0 - offset, 0), 0 + offset, 1 + offset, **kwargs)
    ax.add_patch(rect)

    # right
    rect = patches.Rectangle((1, 0 - offset), 0 + offset, 1 + offset, **kwargs)
    ax.add_patch(rect)

    # top
    rect = patches.Rectangle((0, 1), 1 + offset, 0 + offset, **kwargs)
    ax.add_patch(rect)

    # bottom
    rect = patches.Rectangle((0 - offset, 0), 1 + offset, 0 - offset, **kwargs)
    ax.add_patch(rect)



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
    '-w', '--what', type=click.Choice(['single', 'mean']), default='single')
@click.option(
    '-o', '--output', type=click.Path(
        exists=False,
        dir_okay=False,
    ))
def main(predicted_gammas, predicted_protons, what, output):
    cols = ['gamma_prediction', 'array_event_id', 'run_id', 'telescope_type_name']

    gammas = fact.io.read_data(predicted_gammas, key='telescope_events', columns=cols).dropna()
    protons = fact.io.read_data(predicted_protons, key='telescope_events', columns=cols).dropna()

    for tel_type in ['LST', 'MST', 'SST']:
        tel_gammas = gammas.query(f'telescope_type_name == "{tel_type}"')
        tel_protons = protons.query(f'telescope_type_name == "{tel_type}"')
        if what == 'mean':
            prediction_gammas = tel_gammas.groupby(['array_event_id', 'run_id'])['gamma_prediction'].mean()
            prediction_protons = tel_protons.groupby(['array_event_id', 'run_id'])['gamma_prediction'].mean()
        else:
            prediction_gammas = tel_gammas.gamma_prediction
            prediction_protons = tel_protons.gamma_prediction

        gamma_labels = np.ones_like(prediction_gammas)
        proton_labels = np.zeros_like(prediction_protons)

        y_score = np.hstack([prediction_gammas, prediction_protons])
        y_true = np.hstack([gamma_labels, proton_labels])

        fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
        auc = roc_auc_score(y_true, y_score)
        plt.plot(fpr, tpr, lw=2, label=f'AUC for {tel_type}:  {auc:{1}.{3}}', color=telescope_color[tel_type])

    add_rectangles(plt.gca())
    plt.legend()
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.tight_layout()
    if output:
        plt.savefig(output)
    else:
        plt.show()


if __name__ == '__main__':
    main()
