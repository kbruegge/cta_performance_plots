import os

import click
import numpy as np
import matplotlib.pyplot as plt
from cta_plots import apply_cuts
from cta_plots.reconstruction.angular_resolution import plot_angular_resolution, plot_angular_resolution_per_multiplicity
from cta_plots.reconstruction.h_max import plot_h_max, plot_h_max_distance
from cta_plots.reconstruction.impact import plot_impact, plot_impact_distance
from cta_plots import load_signal_events
from cta_plots.colors import main_color, default_cmap


def _apply_flags(ctx, ax, data=None):
    if ctx.obj["YLIM"]:
        ax.set_ylim(ctx.obj["YLIM"])

    if ctx.obj["YLOG"] is True:
        ax.set_yscale('log')

    legend = ctx.obj["LEGEND"]
    if legend is False:
        ax.get_legend().remove()

    output = ctx.obj["OUTPUT"]
    if output:
        plt.tight_layout(pad=0, rect=(0, 0, 1.002, 1))
        plt.savefig(output)
        if data is not None:
            n, _ = os.path.splitext(output)
            data.to_csv(n + '.csv', index=False, na_rep='NaN', )
    else:
        plt.show()


def _load_data(path, dropna=True):
    cols = [
        'mc_energy',
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
        'h_max',
        'mc_x_max',
        'core_x',
        'core_y',
        'mc_core_x',
        'mc_core_y',
    ]

    df, _, _ = load_signal_events(path, calculate_weights=False, columns=cols) 
    if dropna:
        df.dropna(inplace=True)
    return df


@click.group(invoke_without_command=True)
@click.option("--debug/--no-debug", default=False)
@click.option("--dropna/--no-dropna", default=True)
@click.option("--legend/--no-legend", default=True)
@click.option("--ylog/--no-ylog", default=True)
@click.option("--ylim", default=None, nargs=2, type=np.float)
@click.option('-o', '--output', type=click.Path(exists=False))
@click.argument('path', type=click.Path(exists=True))
@click.pass_context
def cli(ctx, path, debug, dropna, legend, ylog, ylim, output):
    # ensure that ctx.obj exists and is a dict (in case `cli()` is called
    # by means other than the `if` block below
    # see https://click.palletsprojects.com/en/7.x/commands/#nested-handling-and-contexts
    ctx.ensure_object(dict)
    ctx.obj["DEBUG"] = debug
    ctx.obj["OUTPUT"] = output
    ctx.obj["LEGEND"] = legend
    ctx.obj["YLIM"] = ylim
    ctx.obj["YLOG"] = ylog
    ctx.obj["DATA"] = _load_data(path, dropna)
    if debug and ctx.invoked_subcommand is None:
        print("I was invoked without subcommand")
    elif debug:
        print(f"I am about to invoke {ctx.invoked_subcommand} with {ctx.obj}")


@cli.command()
@click.option('--reference/--no-reference', default=False)
@click.option('--plot_e_reco', is_flag=True, default=False)
@click.option('-c', '--cuts_path', type=click.Path(exists=True))
@click.pass_context
def angular_resolution(ctx, reference, plot_e_reco, cuts_path):
    reconstructed_events = ctx.obj["DATA"]
    ylog = ctx.obj["YLOG"]
    ylim = ctx.obj["YLIM"]
    if cuts_path:
        reconstructed_events = apply_cuts(reconstructed_events, cuts_path, theta_cuts=False)
    ax, df = plot_angular_resolution(reconstructed_events, reference, plot_e_reco, ylog=ylog, ylim=ylim)
    _apply_flags(ctx, ax, data=df)


@cli.command()
@click.option('--reference/--no-reference', default=False)
@click.option('--plot_e_reco', is_flag=True, default=False)
@click.option('-c', '--cuts_path', type=click.Path(exists=True))
@click.pass_context
def angular_resolution_per_tel(ctx, reference, plot_e_reco, cuts_path):
    reconstructed_events = ctx.obj["DATA"]
    if cuts_path:
        reconstructed_events = apply_cuts(reconstructed_events, cuts_path, theta_cuts=False, )
    ax = plot_angular_resolution_per_multiplicity(reconstructed_events, reference, plot_e_reco)
    _apply_flags(ctx, ax)


@cli.command()
@click.option('-c', '--cuts_path', type=click.Path(exists=True))
@click.option('--color', default=main_color)
@click.option('--cmap', default=default_cmap)
@click.pass_context
def h_max(ctx, cuts_path, color, cmap):
    reconstructed_events = ctx.obj["DATA"]
    if cuts_path:
        reconstructed_events = apply_cuts(reconstructed_events, cuts_path, theta_cuts=False)
    ax = plot_h_max(reconstructed_events, color=color, colormap=cmap)
    _apply_flags(ctx, ax)


@cli.command()
@click.option('--plot_e_reco', is_flag=True, default=True)
@click.option('--color', default=main_color)
@click.option('--cmap', default=default_cmap)
@click.pass_context
def h_max_distance(ctx, cuts_path, color, cmap):
    reconstructed_events = ctx.obj["DATA"]
    if cuts_path:
        reconstructed_events = apply_cuts(reconstructed_events, cuts_path, theta_cuts=False)
    ax = plot_h_max_distance(reconstructed_events, color=color, colormap=cmap)
    _apply_flags(ctx, ax)


@cli.command()
@click.option('-c', '--cuts_path', type=click.Path(exists=True))
@click.option('--color', default=main_color)
@click.option('--cmap', default=default_cmap)
@click.pass_context
def impact(ctx, cuts_path, color, cmap):
    reconstructed_events = ctx.obj["DATA"]
    if cuts_path:
        reconstructed_events = apply_cuts(reconstructed_events, cuts_path, theta_cuts=False)
    ax = plot_impact(reconstructed_events, color=color, colormap=cmap)
    _apply_flags(ctx, ax)


@cli.command()
@click.option('-c', '--cuts_path', type=click.Path(exists=True))
@click.option('--color', default=main_color)
@click.option('--cmap', default=default_cmap)
@click.pass_context
def impact_distance(ctx, cuts_path, color, cmap):
    reconstructed_events = ctx.obj["DATA"]
    if cuts_path:
        reconstructed_events = apply_cuts(reconstructed_events, cuts_path, theta_cuts=False)
    ax = plot_impact_distance(reconstructed_events, color=color, colormap=cmap)
    _apply_flags(ctx, ax)


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    cli(obj={})