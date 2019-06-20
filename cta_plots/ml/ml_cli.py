import os

import click
import matplotlib.pyplot as plt
import numpy as np
from cta_plots.ml.auc import plot_auc, plot_auc_per_type, plot_auc_vs_energy, plot_balanced_acc, plot_quick_auc
from cta_plots.ml.importances import plot_importances
from cta_plots.ml.prediction_hist import plot_prediction_histogram, plot_quick_histogram
from cta_plots.ml.energy import plot_resolution, plot_bias
import fact.io
from .. import load_signal_events, apply_cuts


def _apply_flags(ctx, ax, data=None):
    if ctx.obj["YLIM"]:
        ax.set_ylim(ctx.obj["YLIM"])

    output = ctx.obj["OUTPUT"]
    if output:
        plt.savefig(output)
        if data is not None:
            n, _ = os.path.splitext(output)
            data.to_csv(n + '.csv', index=False, na_rep='NaN', )
    else:
        plt.show()




def _load_data(path, dropna=True):
    cols = [
        'mc_alt',
        'mc_az',
        'alt',
        'az',
        'gamma_prediction_mean',
        'gamma_energy_prediction_mean',
        'num_triggered_telescopes',
        'num_triggered_lst',
        'num_triggered_sst',
        'num_triggered_mst',
        'mc_energy',
    ]

    df, _, _ = load_signal_events(path, calculate_weights=False, columns=cols) 
    if dropna:
        df.dropna(inplace=True)
    return df


def _load_predictions(gammas_path, protons_path):
    cols = ["gamma_prediction_mean", "array_event_id", "run_id"]

    gammas = fact.io.read_data(
        gammas_path, key="array_events", columns=cols
    ).dropna()
    protons = fact.io.read_data(
        protons_path, key="array_events", columns=cols
    ).dropna()

    return gammas.gamma_prediction_mean, protons.gamma_prediction_mean


def _load_telescope_data(gammas_path, protons_path):
    cols = ["gamma_prediction", "gamma_energy_prediction", "array_event_id", "run_id", "telescope_type_id", "intensity"]

    gammas = fact.io.read_data(
        gammas_path, key="telescope_events", columns=cols
    ).dropna()
    protons = fact.io.read_data(
        protons_path, key="telescope_events", columns=cols
    ).dropna()

    return gammas, protons



@click.group(invoke_without_command=True)
@click.option("--debug/--no-debug", default=False)
@click.option("--ylim", default=None, nargs=2, type=np.float)
@click.option('-o', '--output', type=click.Path(exists=False))
@click.pass_context
def cli(ctx, debug, ylim, output):
    # ensure that ctx.obj exists and is a dict (in case `cli()` is called
    # by means other than the `if` block below
    # see https://click.palletsprojects.com/en/7.x/commands/#nested-handling-and-contexts
    ctx.ensure_object(dict)
    ctx.obj["DEBUG"] = debug
    ctx.obj["OUTPUT"] = output
    ctx.obj["YLIM"] = ylim
    if debug and ctx.invoked_subcommand is None:
        click.echo("I was invoked without subcommand")
    elif debug:
        click.echo(f"I am about to invoke {ctx.invoked_subcommand} with {ctx.obj}")


@cli.command()
@click.argument("reconstructed_events", type=click.Path())
@click.option('--reference/--no-reference', default=False)
@click.option('--relative/--no-relative', default=True)
@click.option('--plot_e_reco', is_flag=True, default=False)
@click.option('-c', '--cuts_path', type=click.Path(exists=True))
@click.pass_context
def energy_resolution(ctx, reconstructed_events, reference, relative, plot_e_reco, cuts_path):
    reconstructed_events = _load_data(reconstructed_events, dropna=True)
    if cuts_path:
        reconstructed_events = apply_cuts(reconstructed_events, cuts_path)
    e_true = reconstructed_events.mc_energy
    e_reco = reconstructed_events.gamma_energy_prediction_mean
    ax, df = plot_resolution(e_true, e_reco, reference=reference, relative=relative, plot_e_reco=plot_e_reco)
    _apply_flags(ctx, ax, data=df)


@cli.command()
@click.argument("reconstructed_events", type=click.Path())
@click.option('-c', '--cuts_path', type=click.Path(exists=True))
@click.pass_context
def energy_bias(ctx, reconstructed_events, cuts_path):
    reconstructed_events = _load_data(reconstructed_events, dropna=True)
    if cuts_path:
        reconstructed_events = apply_cuts(reconstructed_events, cuts_path)
    e_true = reconstructed_events.mc_energy
    e_reco = reconstructed_events.gamma_energy_prediction_mean
    ax, df = plot_bias(e_true, e_reco)
    _apply_flags(ctx, ax, data=df)



@cli.command()
@click.argument("gammas", type=click.Path())
@click.argument("protons", type=click.Path())
@click.option(
    "-w",
    "--what",
    default=["mean"],
    type=click.Choice(
        [
            "per-telescope",
            "mean",
            "single",
            "median",
            "weighted-mean",
            "min",
            "max",
            "brightest",
        ]
    ),
    multiple=True
)
@click.pass_context
def auc(ctx, gammas, protons, what):
    gammas, protons = _load_telescope_data(gammas, protons)
    if len(what) == 1:
        ax = plot_auc(gammas, protons, what=what[0], inset=False)
    else:
        fig, ax = plt.subplots(1, 1)
        for w in what:
            ax = plot_auc(gammas, protons, what=w, ax=ax, label=w)

        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])

    _apply_flags(ctx, ax)



@cli.command()
@click.argument("gammas", type=click.Path())
@click.argument("protons", type=click.Path())
@click.option("-w", "--what", type=click.Choice(["single", "mean"]), default="single")
@click.option("--box/--no-box", default=True)
@click.pass_context
def auc_per_type(ctx, gammas, protons, what, box,):
    gammas, protons = _load_telescope_data(gammas, protons)
    ax = plot_auc_per_type(gammas, protons, what, box)
    _apply_flags(ctx, ax)



@cli.command()
@click.argument("gammas", type=click.Path())
@click.argument("protons", type=click.Path())
@click.option(
    "--sample/--no-sample",
    default=True,
    help="Whether to sample bkg events from all energies",
)
@click.option(
    "--e_reco/--no-e_reco", default=True, help="Whether to plot vs reconstructed energy"
)
@click.pass_context
def auc_vs_energy(ctx, gammas, protons, sample, e_reco,):
    gammas, protons = _load_telescope_data(gammas, protons)
    ax = plot_auc_vs_energy(gammas, protons, e_reco, sample)
    _apply_flags(ctx, ax)


@cli.command()
@click.argument("model", type=click.Path())
@click.option("-c", "--color", default="crimson")
@click.pass_context
def importances(ctx, model, color):
    fig = plt.gcf()
    size = list(fig.get_size_inches())
    # print(size)
    size[1] += 1.5
    fig, ax = plt.subplots(1, 1, figsize=(size))
    size = list(fig.get_size_inches())
    # print(size)
    ax = plot_importances(model, color=color, ax=ax)
    plt.tight_layout(pad=0, rect=(-0.0135, 0, 1.006, 1))
    _apply_flags(ctx, ax)


@cli.command()
@click.argument("gammas", type=click.Path())
@click.argument("protons", type=click.Path())
@click.option(
    "-w",
    "--what",
    default="mean",
    type=click.Choice(
        [
            "per-telescope",
            "mean",
            "single",
            "median",
            "weighted-mean",
            "min",
            "max",
            "brightest",
        ]
    ),
)
@click.pass_context
def histogram(ctx, gammas, protons, what,):
    gammas, protons = _load_telescope_data(gammas, protons)
    ax = plot_prediction_histogram(gammas, protons, what)
    ax.set_xlim([0, 1])
    ax.legend()
    plt.tight_layout(pad=0)
    _apply_flags(ctx, ax)


@cli.command()
@click.argument("gammas", type=click.Path())
@click.argument("protons", type=click.Path())
@click.option("--box/--no-box", default=True)
@click.pass_context
def roc_acc(ctx, gammas, protons, box,):
    fig, [ax1, ax2] = plt.subplots(1, 2, dpi=400)

    cmap = 'plasma'
    gamma_prediction, protons_prediction = _load_predictions(gammas, protons)
    plot_quick_auc(gamma_prediction, protons_prediction, ax=ax1, cmap=cmap)

    plot_balanced_acc(gamma_prediction, protons_prediction, ax=ax2, cmap=cmap)

    ax1.set_ylim([-0.075, 1.075])
    ax1.set_xlim([-0.075, 1.075])    
    
    ax2.set_ylim([-0.075, 1.075])
    ax2.set_xlim([-0.075, 1.075])

    plt.tight_layout(pad=0, rect=(-0.002, 0, 1.00, 1))
    plt.subplots_adjust(wspace=0.23)
    _apply_flags(ctx, ax1)

@cli.command()
@click.argument("gammas", type=click.Path())
@click.argument("protons", type=click.Path())
@click.option("--box/--no-box", default=True)
@click.pass_context
def hist(ctx, gammas, protons, box,):
    gamma_prediction, protons_prediction = _load_predictions(gammas, protons)
    ax = plot_quick_histogram(gamma_prediction, protons_prediction,)
    ax.legend()

    plt.tight_layout(pad=0)
    _apply_flags(ctx, ax)


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    cli(obj={})
