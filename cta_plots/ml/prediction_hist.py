import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from ..colors import telescope_color


id_to_name = {1: "LST", 2: "MST", 3: "SST"}
name_to_id = {"LST": 1, "MST": 2, "SST": 3}


def plot_quick_histogram(gamma_prediction, proton_prediction, ax=None):
    bins = np.linspace(0, 1, 100)
    if not ax:
        fig, ax = plt.subplots(1)

    signal_color = 'C0'
    h, _, _ = ax.hist(
        gamma_prediction,
        bins=bins,
        histtype="stepfilled",
        density=True,
        linewidth=2,
        color=signal_color,
        facecolor=signal_color,
        alpha=0.4,
        label=None,
    )
    ax.hlines(h, bins[:-1], bins[1:], color=signal_color, label='Gamma')
    h, _, _ = ax.hist(
        proton_prediction,
        bins=bins,
        histtype="stepfilled",
        density=True,
        linewidth=2,
        color='gray',
        facecolor='gray',
        alpha=0.4,
        label=None,
    )
    ax.hlines(h, bins[:-1], bins[1:], color='gray', label='Proton')

    ax.set_xlabel("Prediction Threshold")
    ax.set_ylabel("Normalized Counts")
    return ax


def plot_prediction_histogram(gammas, protons, what='mean', ax=None):
    if what == "mean":
        gamma_prediction = gammas.groupby(["array_event_id", "run_id"])[
            "gamma_prediction"
        ].mean()
        proton_prediction = protons.groupby(["array_event_id", "run_id"])[
            "gamma_prediction"
        ].mean()

        ax.hist(
            gamma_prediction,
            bins=bins,
            histtype="step",
            density=True,
            linewidth=2,
            label='Gamma Prediction'
        )
        ax.hist(
            proton_prediction,
            bins=bins,
            histtype="step",
            density=True,
            linewidth=2,
            color="gray",
            label='Proton Prediction'
        )

    if what == "weighted-mean":
        
        gammas["weight"] = np.log10(gammas.intensity)
        gammas["weighted_prediction"] = gammas.gamma_prediction * gammas.weight
        group = gammas.groupby(["array_event_id", "run_id"])
        gw = group["weighted_prediction"].sum() / group.weight.sum()

        protons["weight"] = np.log10(protons.intensity)
        protons["weighted_prediction"] = protons.gamma_prediction * protons.weight
        group = protons.groupby(["array_event_id", "run_id"])
        pw = group["weighted_prediction"].sum() / group.weight.sum()

        ax.hist(gw, bins=bins, histtype="step", density=True, linewidth=2)
        ax.hist(pw, bins=bins, histtype="step", density=True, linewidth=2, color="gray")

    elif what == "min":

        g = gammas.groupby(["array_event_id", "run_id"])["gamma_prediction"].min()
        p = protons.groupby(["array_event_id", "run_id"])["gamma_prediction"].min()

        ax.hist(g, bins=bins, histtype="step", density=True, linewidth=2)
        ax.hist(p, bins=bins, histtype="step", density=True, linewidth=2)

    elif what == "max":

        g = gammas.groupby(["array_event_id", "run_id"], sort=False)[
            "gamma_prediction"
        ].max()
        p = protons.groupby(["array_event_id", "run_id"], sort=False)[
            "gamma_prediction"
        ].max()

        ax.hist(g, bins=bins, histtype="step", density=True, linewidth=2)
        ax.hist(p, bins=bins, histtype="step", density=True, linewidth=2)

    elif what == "brightest":

        idx = gammas.groupby(["array_event_id", "run_id"], sort=False)[
            "intensity"
        ].idxmax()
        g = gammas.loc[idx].gamma_prediction.values

        idx = protons.groupby(["array_event_id", "run_id"], sort=False)[
            "intensity"
        ].idxmax()
        p = protons.loc[idx].gamma_prediction.values

        ax.hist(g, bins=bins, histtype="step", density=True, linewidth=2)
        ax.hist(p, bins=bins, histtype="step", density=True, linewidth=2)

    elif what == "median":

        g = gammas.groupby(["array_event_id", "run_id"])["gamma_prediction"].median()
        p = protons.groupby(["array_event_id", "run_id"])["gamma_prediction"].median()

        ax.hist(g, bins=bins, histtype="step", density=True, linewidth=2)
        ax.hist(p, bins=bins, histtype="step", density=True, linewidth=2)

    elif what == "single":

        ax.hist(
            gammas.gamma_prediction.values,
            bins=bins,
            histtype="step",
            density=True,
            linewidth=2,
        )
        ax.hist(
            protons.gamma_prediction.values,
            bins=bins,
            histtype="step",
            density=True,
            linewidth=2,
        )

    elif what == "per-telescope":

        for type_id, group in gammas.groupby("telescope_type_id", sort=False):
            name = id_to_name[type_id]
            color = telescope_color[name]
            ax.hist(
                group.gamma_prediction.values,
                bins=bins,
                label=f"gamma prediction {name}",
                histtype="step",
                density=True,
                linewidth=2,
                color=color,
            )

        color_cycle = cycler(color=["gray", "darkgray", "black"])
        ax.set_prop_cycle(color_cycle)
        for type_id, group in protons.groupby("telescope_type_id", sort=False):
            name = id_to_name[type_id]
            ax.hist(
                group.gamma_prediction.values,
                bins=bins,
                label=f"proton prediction {name}",
                histtype="step",
                density=True,
            )

        ax.legend(loc="upper left")

    ax.set_xlabel("Prediction Threshold")
    ax.set_ylabel("Normalized Counts")
    return ax