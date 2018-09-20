import click
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.metrics import roc_curve, roc_auc_score
import fact.io
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset


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
    '-o', '--output', type=click.Path(
        exists=False,
        dir_okay=False,
    ))
def main(predicted_gammas, predicted_protons, output):
    gammas = fact.io.read_data(predicted_gammas, key='array_events', columns=['gamma_prediction_mean']).dropna()
    mean_prediction_gammas = gammas.gamma_prediction_mean
    gamma_labels = np.ones_like(mean_prediction_gammas)

    protons = fact.io.read_data(predicted_protons, key='array_events', columns=['gamma_prediction_mean']).dropna()
    mean_prediction_protons = protons.gamma_prediction_mean
    proton_labels = np.zeros_like(mean_prediction_protons)

    y_score = np.hstack([mean_prediction_gammas, mean_prediction_protons])
    y_true = np.hstack([gamma_labels, proton_labels])

    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    auc = roc_auc_score(y_true, y_score)

    plt.plot(fpr, tpr, lw=2)

    add_rectangles(plt.gca())

    plt.text(
        0.95,
        0.1,
        'Area Under Curve: ${:.4f}$'.format(auc),
        verticalalignment='bottom',
        horizontalalignment='right',
        color='#404040',
        fontsize=11
    )

    if False:
        ax = plt.gca()
        axins = zoomed_inset_axes(ax, 2, loc='upper center')  # zoom = 6
        axins.plot(fpr, tpr, lw=2)

        # sub region of the original image
        x1, x2, y1, y2 = 0.8, 1, 0, 0.5
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        axins.set_xticks([])
        axins.set_yticks([])

        # draw a bbox of the region of the inset axes in the parent axes and
        # connecting lines between the bbox and the inset axes area
        mark_inset(ax, axins, loc1=3, loc2=4, ec='0.7')

        plt.setp(axins.spines.values(), color='darkgray')

    plt.tight_layout()
    if output:
        plt.savefig(output)
    else:
        plt.show()


if __name__ == '__main__':
    main()
