import click
import matplotlib.pyplot as plt
import numpy as np
from cta_plots.ml.auc import plot_auc
from cta_plots.ml.auc_per_type import plot_auc_per_type
from cta_plots.ml.auc_vs_energy import plot_auc_vs_energy
from cta_plots.ml.importances import plot_importances
from cta_plots.ml.prediction_hist import plot_prediction_histogram


def _apply_flags(ctx, ax, output):
    if ctx.obj["YLIM"]:
        ax.set_ylim(ctx.obj["YLIM"])

    if output:
        plt.savefig(output)
    else:
        plt.show()


@click.group(invoke_without_command=True)
@click.option("--debug/--no-debug", default=False)
@click.option("--ylim", default=None, nargs=2, type=np.float)
@click.pass_context
def cli(ctx, debug, ylim):
    # ensure that ctx.obj exists and is a dict (in case `cli()` is called
    # by means other than the `if` block below
    # see https://click.palletsprojects.com/en/7.x/commands/#nested-handling-and-contexts
    ctx.ensure_object(dict)
    ctx.obj["DEBUG"] = debug
    ctx.obj["YLIM"] = ylim
    if debug and ctx.invoked_subcommand is None:
        click.echo("I was invoked without subcommand")
    elif debug:
        click.echo(f"I am about to invoke {ctx.invoked_subcommand} with {ctx.obj}")


@cli.command()
@click.argument("gammas", type=click.Path())
@click.argument("protons", type=click.Path())
@click.option("-o", "--output", type=click.Path(exists=False))
@click.pass_context
def auc(ctx, gammas, protons, output):
    ax = plot_auc(gammas, protons, inset=False)
    _apply_flags(ctx, ax, output)


@cli.command()
@click.argument("model", type=click.Path())
@click.option("-o", "--output", type=click.Path(exists=False))
@click.option("-c", "--color", default="crimson")
@click.pass_context
def importances(ctx, model, color, output):
    ax = plot_importances(model, color=color)
    _apply_flags(ctx, ax, output)


@cli.command()
@click.argument("gammas", type=click.Path())
@click.argument("protons", type=click.Path())
@click.option("-w", "--what", type=click.Choice(["single", "mean"]), default="single")
@click.option("-o", "--output", type=click.Path(exists=False, dir_okay=False))
@click.option("--box/--no-box", default=True)
@click.pass_context
def auc_per_type(ctx, gammas, protons, what, box, output):
    ax = plot_auc_per_type(gammas, protons, what, box)
    _apply_flags(ctx, ax, output)


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
@click.option("-o", "--output", type=click.Path(exists=False, dir_okay=False))
@click.pass_context
def auc_vs_energy(ctx, gammas, protons, sample, e_reco, output):
    ax = plot_auc_vs_energy(gammas, protons, e_reco, sample)
    _apply_flags(ctx, ax, output)


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
@click.option("-o", "--output", type=click.Path(exists=False, dir_okay=False))
@click.pass_context
def prediction_histogram(ctx, gammas, protons, what, output):
    ax = plot_prediction_histogram(gammas, protons, what)
    _apply_flags(ctx, ax, output)


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    cli(obj={})
