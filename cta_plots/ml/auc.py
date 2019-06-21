import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection

from sklearn.metrics import roc_curve, roc_auc_score
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from ..colors import telescope_color
from ..binning import make_default_cta_binning
import astropy.units as u
import pandas as pd


name_to_id = {"LST": 1, "MST": 2, "SST": 3}
id_to_name = {1: "LST", 2: "MST", 3: "SST"}


def add_rectangles(ax, offset=0.15):

    kwargs = {"linewidth": 1, "edgecolor": "none", "facecolor": "white", "alpha": 0.6, 'zorder':-10}

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


def plot_balanced_acc(prediction_gammas, prediction_protons, ax=None, cmap='magma'):
    gamma_labels = np.ones_like(prediction_gammas)
    proton_labels = np.zeros_like(prediction_protons)

    y_score = np.hstack([prediction_gammas, prediction_protons])
    y_true = np.hstack([gamma_labels, proton_labels])

    fpr, tpr, threshold = roc_curve(y_true, y_score, pos_label=1)

    # m  = (threshold >=0 ) & (threshold <= 1)
    fpr, tpr, threshold = fpr[1:], tpr[1:], threshold[1:]
    if not ax:
        fig, ax = plt.subplots(1, 1)
    add_rectangles(ax)

    tnr = 1 - fpr
    bacc = 0.5 * (tpr + tnr)
    ax.scatter(threshold, bacc, c=threshold, rasterized=True, s=3, cmap=cmap)

    
    ax.set_ylabel('Balanced Accuracy')
    ax.set_xlabel('Prediction Threshold')

    ax.set_aspect('equal')
    return ax


def plot_quick_auc(prediction_gammas, prediction_protons, label='', ax=None, cmap='magma'):
    # see https://matplotlib.org/3.1.0/gallery/lines_bars_and_markers/multicolored_line.html
    # is ugly and why yoo slow. just rasterize scatter plot
    if not ax:
        fig, ax = plt.subplots(1, 1)
    
    add_rectangles(ax)
    gamma_labels = np.ones_like(prediction_gammas)
    proton_labels = np.zeros_like(prediction_protons)

    y_score = np.hstack([prediction_gammas, prediction_protons])
    y_true = np.hstack([gamma_labels, proton_labels])

    fpr, tpr, threshold = roc_curve(y_true, y_score, pos_label=1)
    auc = roc_auc_score(y_true, y_score)

    fpr, tpr, threshold = fpr[1:], tpr[1:], threshold[1:]

    ax.scatter(fpr, tpr, c=threshold, rasterized=True, s=3, cmap=cmap)
    ax.text(0.2, 0.2, f'area under curve:  {auc:.2f}', alpha=0.5)

    ax.set_ylabel('TPR')
    ax.set_xlabel('FPR')
    ax.set_aspect('equal')
    return ax


def plot_auc(gammas, protons, what='mean', inset=False, label='', ax=None):
    if what == "mean":
        prediction_gammas = gammas.groupby(["array_event_id", "run_id"])["gamma_prediction"].mean()
        prediction_protons = protons.groupby(["array_event_id", "run_id"])["gamma_prediction"].mean()
    elif what == "median":
        prediction_gammas = gammas.groupby(["array_event_id", "run_id"])["gamma_prediction"].median()
        prediction_protons = protons.groupby(["array_event_id", "run_id"])["gamma_prediction"].median()
    elif what == 'single':
        prediction_gammas = gammas.gamma_prediction
        prediction_protons = protons.gamma_prediction
    elif what == "min":
        prediction_gammas = gammas.groupby(["array_event_id", "run_id"])["gamma_prediction"].min()
        prediction_protons = protons.groupby(["array_event_id", "run_id"])["gamma_prediction"].min()
    elif what == "max":
        prediction_gammas = gammas.groupby(["array_event_id", "run_id"], )["gamma_prediction"].max()
        prediction_protons = protons.groupby(["array_event_id", "run_id"],)["gamma_prediction"].max()
    elif what == "brightest":
        idx = gammas.groupby(["array_event_id", "run_id"])["intensity"].idxmax()
        prediction_gammas = gammas.loc[idx].gamma_prediction.values

        idx = protons.groupby(["array_event_id", "run_id"])["intensity"].idxmax()
        prediction_protons = protons.loc[idx].gamma_prediction.values

    gamma_labels = np.ones_like(prediction_gammas)
    proton_labels = np.zeros_like(prediction_protons)

    y_score = np.hstack([prediction_gammas, prediction_protons])
    y_true = np.hstack([gamma_labels, proton_labels])

    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    auc = roc_auc_score(y_true, y_score)

    if not ax:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = plt.gcf()
    if label:
        label_text = f'Aggregation: "{label}" AuC: {auc:.3f}'
    else:
        label_text = f'Area under Curve: {auc:.3f}'
    ax.plot(fpr, tpr, lw=2, label=label_text)

    legend = ax.legend(loc='lower right', framealpha=0.5)

    # see this SO post for right alignment of text.
    # https://stackoverflow.com/a/16858263/2154625
    
    renderer = fig.canvas.get_renderer()
    shift = max([t.get_window_extent(renderer).width for t in legend.get_texts()])
    for t in legend.get_texts():
        t.set_ha('right')  # ha is alias for horizontalalignment
        t.set_position((shift, 0))

    add_rectangles(ax)

 
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")

    if inset:
        axins = zoomed_inset_axes(ax, 2, loc="upper center")  # zoom = 6
        axins.plot(fpr, tpr, lw=2)

        # sub region of the original image
        x1, x2, y1, y2 = 0.05, 0.2, 0.7, 1
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        axins.set_xticks([])
        axins.set_yticks([])

        # draw a bbox of the region of the inset axes in the parent axes and
        # connecting lines between the bbox and the inset axes area
        mark_inset(ax, axins, loc1=3, loc2=4, ec="0.7")

        # axins.spines.color = 'darkgray'

    plt.tight_layout(pad=0)
    return ax



def plot_auc_per_type(gammas, protons, what, box, ax=None):

    if not ax:
        fig, ax = plt.subplots(1, 1)

    for tel_type in ["LST", "MST", "SST"]:
        tel_gammas = gammas.query(f'telescope_type_id == "{name_to_id[tel_type]}"')
        tel_protons = protons.query(f'telescope_type_id == "{name_to_id[tel_type]}"')
        if what == "mean":
            prediction_gammas = tel_gammas.groupby(["array_event_id", "run_id"])[
                "gamma_prediction"
            ].mean()
            prediction_protons = tel_protons.groupby(["array_event_id", "run_id"])[
                "gamma_prediction"
            ].mean()
        else:
            prediction_gammas = tel_gammas.gamma_prediction
            prediction_protons = tel_protons.gamma_prediction

        gamma_labels = np.ones_like(prediction_gammas)
        proton_labels = np.zeros_like(prediction_protons)

        y_score = np.hstack([prediction_gammas, prediction_protons])
        y_true = np.hstack([gamma_labels, proton_labels])

        fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
        auc = roc_auc_score(y_true, y_score)
        ax.plot(
            fpr,
            tpr,
            lw=2,
            label=f"AUC for {tel_type}:  {auc:{1}.{3}}",
            color=telescope_color[tel_type],
        )
        ax.legend()
    if box:
        add_rectangles(ax)


def plot_auc_vs_energy(gammas, protons, e_reco=False, sample=False, ax=None):


    bins, bin_center, bin_widths = make_default_cta_binning(
        e_min=0.008 * u.TeV, e_max=300 * u.TeV
    )

    if e_reco:
        key = "gamma_energy_prediction"
    else:
        key = "mc_energy"

    gammas["energy_bin"] = pd.cut(gammas[key], bins)
    protons["energy_bin"] = pd.cut(protons[key], bins)

    if not ax:
        fig, ax = plt.subplots(1, 1)

    for tel_type in ["SST", "MST", "LST"]:
        aucs = []
        for b in gammas.energy_bin.cat.categories:

            tel_gammas = gammas[
                (gammas.energy_bin == b)
                &
                (gammas.telescope_type_id == name_to_id[tel_type])
            ]
            if sample:
                tel_protons = protons[protons.telescope_type_id == name_to_id[tel_type]]
            else:
                tel_protons = protons[
                    (protons.energy_bin == b)
                    & 
                    (protons.telescope_type_id == name_to_id[tel_type])
                ]

            if len(tel_gammas) < 350 or len(tel_protons) < 350:
                aucs.append(np.nan)
            else:
                mean_prediction_gammas = tel_gammas.groupby(
                    ["array_event_id", "run_id"]
                )["gamma_prediction"].mean()
                gamma_labels = np.ones_like(mean_prediction_gammas)

                mean_prediction_protons = tel_protons.groupby(
                    ["array_event_id", "run_id"]
                )["gamma_prediction"].mean()
                proton_labels = np.zeros_like(mean_prediction_protons)

                y_score = np.hstack([mean_prediction_gammas, mean_prediction_protons])
                y_true = np.hstack([gamma_labels, proton_labels])

                aucs.append(roc_auc_score(y_true, y_score))

        ax.errorbar(
            bin_center.value,
            aucs,
            xerr=bin_widths.value / 2.0,
            linestyle="--",
            label=tel_type,
            ecolor="gray",
            ms=0,
            capsize=0,
            color=telescope_color[tel_type],
        )

    ax.set_xscale("log")

    if e_reco:
        label = r"$E_{Reco} / TeV$"
    else:
        label = r"$E_{True} / TeV$"
    ax.set_xlabel(label)
    ax.set_ylabel("Area Under RoC Curve")
    ax.legend()
    return ax

