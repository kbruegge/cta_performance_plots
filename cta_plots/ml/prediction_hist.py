import numpy as np
import matplotlib.pyplot as plt
import fact.io
from cycler import cycler
from ..colors import telescope_color


id_to_name = {1: "LST", 2: "MST", 3: "SST"}
name_to_id = {"LST": 1, "MST": 2, "SST": 3}


def plot_prediction_histogram(predicted_gammas, predicted_protons, what, ax=None):
    bins = np.linspace(0, 1, 100)
    if not ax:
        fig, ax = plt.subplots(1)
    if what == "mean":
        cols = ["gamma_prediction_mean"]
        gammas = fact.io.read_data(
            predicted_gammas, key="array_events", columns=cols
        ).dropna()
        protons = fact.io.read_data(
            predicted_protons, key="array_events", columns=cols
        ).dropna()

        ax.hist(
            gammas.gamma_prediction_mean.values,
            bins=bins,
            histtype="step",
            density=True,
            linewidth=2,
        )
        ax.hist(
            protons.gamma_prediction_mean.values,
            bins=bins,
            histtype="step",
            density=True,
            linewidth=2,
            color="gray",
        )

    if what == "weighted-mean":
        cols = ["gamma_prediction", "array_event_id", "run_id", "intensity"]
        gammas = fact.io.read_data(
            predicted_gammas, key="telescope_events", columns=cols
        ).dropna()
        protons = fact.io.read_data(
            predicted_protons, key="telescope_events", columns=cols
        ).dropna()

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
        cols = ["gamma_prediction", "array_event_id", "run_id"]
        gammas = fact.io.read_data(
            predicted_gammas, key="telescope_events", columns=cols
        ).dropna()
        protons = fact.io.read_data(
            predicted_protons, key="telescope_events", columns=cols
        ).dropna()

        g = gammas.groupby(["array_event_id", "run_id"])["gamma_prediction"].min()
        p = protons.groupby(["array_event_id", "run_id"])["gamma_prediction"].min()

        fig, ax = plt.subplots(1)
        ax.hist(g, bins=bins, histtype="step", density=True, linewidth=2)
        ax.hist(p, bins=bins, histtype="step", density=True, linewidth=2)

    elif what == "max":
        cols = ["gamma_prediction", "array_event_id", "run_id"]
        gammas = fact.io.read_data(
            predicted_gammas, key="telescope_events", columns=cols
        ).dropna()
        protons = fact.io.read_data(
            predicted_protons, key="telescope_events", columns=cols
        ).dropna()

        g = gammas.groupby(["array_event_id", "run_id"], sort=False)[
            "gamma_prediction"
        ].max()
        p = protons.groupby(["array_event_id", "run_id"], sort=False)[
            "gamma_prediction"
        ].max()

        fig, ax = plt.subplots(1)
        ax.hist(g, bins=bins, histtype="step", density=True, linewidth=2)
        ax.hist(p, bins=bins, histtype="step", density=True, linewidth=2)

    elif what == "brightest":
        cols = ["gamma_prediction", "array_event_id", "run_id", "intensity"]
        gammas = fact.io.read_data(
            predicted_gammas, key="telescope_events", columns=cols
        ).dropna()
        protons = fact.io.read_data(
            predicted_protons, key="telescope_events", columns=cols
        ).dropna()

        idx = gammas.groupby(["array_event_id", "run_id"], sort=False)[
            "intensity"
        ].idxmax()
        g = gammas.loc[idx].gamma_prediction.values

        idx = protons.groupby(["array_event_id", "run_id"], sort=False)[
            "intensity"
        ].idxmax()
        p = protons.loc[idx].gamma_prediction.values

        fig, ax = plt.subplots(1)
        ax.hist(g, bins=bins, histtype="step", density=True, linewidth=2)
        ax.hist(p, bins=bins, histtype="step", density=True, linewidth=2)

    elif what == "median":
        cols = ["gamma_prediction", "array_event_id", "run_id"]
        gammas = fact.io.read_data(
            predicted_gammas, key="telescope_events", columns=cols
        ).dropna()
        protons = fact.io.read_data(
            predicted_protons, key="telescope_events", columns=cols
        ).dropna()

        g = gammas.groupby(["array_event_id", "run_id"])["gamma_prediction"].median()
        p = protons.groupby(["array_event_id", "run_id"])["gamma_prediction"].median()

        ax.hist(g, bins=bins, histtype="step", density=True, linewidth=2)
        ax.hist(p, bins=bins, histtype="step", density=True, linewidth=2)

    elif what == "single":
        cols = ["gamma_prediction"]
        gammas = fact.io.read_data(
            predicted_gammas, key="telescope_events", columns=cols
        ).dropna()
        protons = fact.io.read_data(
            predicted_protons, key="telescope_events", columns=cols
        ).dropna()

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
        cols = ["telescope_type_id", "gamma_prediction"]
        gammas = fact.io.read_data(
            predicted_gammas, key="telescope_events", columns=cols
        ).dropna()
        protons = fact.io.read_data(
            predicted_protons, key="telescope_events", columns=cols
        ).dropna()

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

    ax.set_xlabel("Classifier Score")
    ax.set_ylabel("Normalized Counts")
