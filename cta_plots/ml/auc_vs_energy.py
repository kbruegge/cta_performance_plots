import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from ..colors import telescope_color
from ..binning import make_default_cta_binning
import astropy.units as u

import fact.io

columns = [
    "array_event_id",
    "gamma_prediction",
    "gamma_energy_prediction",
    "telescope_type_id",
    "run_id",
]
columns_array = ["run_id", "array_event_id", "mc_energy", "total_intensity"]
id_to_name = {1: "LST", 2: "MST", 3: "SST"}
name_to_id = {"LST": 1, "MST": 2, "SST": 3}


def plot_auc_vs_energy(
    predicted_gammas, predicted_protons, e_reco=False, sample=False, ax=None
):

    telecope_events = fact.io.read_data(
        predicted_gammas, key="telescope_events", columns=columns, last=1000000
    ).dropna()
    array_events = fact.io.read_data(
        predicted_gammas, key="array_events", columns=columns_array
    )
    gammas = pd.merge(telecope_events, array_events, on=["run_id", "array_event_id"])

    telecope_events = fact.io.read_data(
        predicted_protons, key="telescope_events", columns=columns, last=1000000
    ).dropna()
    array_events = fact.io.read_data(
        predicted_protons, key="array_events", columns=columns_array
    )
    protons = pd.merge(telecope_events, array_events, on=["run_id", "array_event_id"])

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
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    for tel_type in ["SST", "MST", "LST"]:
        aucs = []
        for b in tqdm(gammas.energy_bin.cat.categories):

            tel_gammas = gammas[
                (gammas.energy_bin == b)
                & (gammas.telescope_type_id == name_to_id[tel_type])
            ]
            if sample:
                tel_protons = protons[protons.telescope_type_id == name_to_id[tel_type]]
            else:
                tel_protons = protons[
                    (protons.energy_bin == b)
                    & (protons.telescope_type_id == name_to_id[tel_type])
                ]

            if len(tel_gammas) < 100 or len(tel_protons) < 100:
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
    # add_rectangles(plt.gca())
