import os

import click
import matplotlib.pyplot as plt
import numpy as np
from cta_plots.ml.auc import plot_auc, plot_auc_per_type, plot_auc_vs_energy, plot_balanced_acc, plot_quick_auc
from cta_plots.ml.prediction_hist import plot_quick_histogram
import fact.io


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
    cols = ["gamma_prediction_mean", "gamma_energy_prediction_mean", "array_event_id", "run_id", "total_intensity"]

    gammas = fact.io.read_data(
        gammas_path, key="array_events", columns=cols
    ).dropna()
    protons = fact.io.read_data(
        protons_path, key="array_events", columns=cols
    ).dropna()

    return gammas, protons



@click.group(invoke_without_command=True)
@click.argument("gammas", type=click.Path())
@click.argument("protons", type=click.Path())
@click.option("--debug/--no-debug", default=False)
@click.option("--ylim", default=None, nargs=2, type=np.float)
@click.option('-o', '--output', type=click.Path(exists=False))
@click.option('-c', '--cuts_path', type=click.Path(exists=True))
@click.pass_context
def cli(ctx, gammas, protons, debug, ylim, output, cuts_path):
    # ensure that ctx.obj exists and is a dict (in case `cli()` is called
    # by means other than the `if` block below
    # see https://click.palletsprojects.com/en/7.x/commands/#nested-handling-and-contexts
    ctx.ensure_object(dict)
    ctx.obj["DEBUG"] = debug
    ctx.obj["OUTPUT"] = output
    ctx.obj["YLIM"] = ylim

    gammas, protons = _load_telescope_data(gammas, protons)
    ctx.obj["GAMMAS"] = gammas
    ctx.obj["PROTONS"] = protons

    if debug and ctx.invoked_subcommand is None:
        click.echo("I was invoked without subcommand")
    elif debug:
        click.echo(f"I am about to invoke {ctx.invoked_subcommand} with {ctx.obj}")






@cli.command()
@click.option(
    "--sample/--no-sample",
    default=True,
    help="Whether to sample bkg events from all energies",
)
@click.option(
    "--e_reco/--no-e_reco", default=True, help="Whether to plot vs reconstructed energy"
)
@click.pass_context
def auc_vs_energy(ctx, sample, e_reco,):
    gammas = ctx.obj['GAMMAS']
    protons = ctx.obj['PROTONS']
    ax = plot_auc_vs_energy(gammas, protons, e_reco, sample)
    _apply_flags(ctx, ax)


@cli.command()
@click.option("--box/--no-box", default=True)
@click.pass_context
def roc_acc(ctx, box):
    gammas = ctx.obj['GAMMAS']
    protons = ctx.obj['PROTONS']
    gamma_prediction, protons_prediction = gammas.gamma_prediction_mean, protons.gamma_prediction_mean

    fig, [ax1, ax2] = plt.subplots(1, 2, dpi=800)

    cmap = 'plasma'
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
@click.option("--box/--no-box", default=True)
@click.pass_context
def hist(ctx, box,):
    gammas = ctx.obj['GAMMAS']
    protons = ctx.obj['PROTONS']
    gamma_prediction, protons_prediction = gammas.gamma_prediction_mean, protons.gamma_prediction_mean

    ax = plot_quick_histogram(gamma_prediction, protons_prediction,)
    ax.legend()

    plt.tight_layout(pad=0)
    _apply_flags(ctx, ax)


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    cli(obj={})
