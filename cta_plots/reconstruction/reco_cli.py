import os

import click
import numpy as np
import matplotlib.pyplot as plt
import h5py
from cta_plots import apply_cuts
from cta_plots.reconstruction.angular_resolution import plot_angular_resolution, plot_angular_resolution_per_multiplicity
from cta_plots.reconstruction.h_max import plot_h_max, plot_h_max_distance
from cta_plots.reconstruction.impact import plot_impact, plot_impact_distance
from cta_plots.reconstruction.energy import plot_resolution
from cta_plots import load_signal_events, load_data_description
from cta_plots.colors import main_color, default_cmap


def _apply_flags(ctx, axs, data=None):
    try:
        iter(axs)
    except TypeError:
        axs = [axs]

    if ctx.obj["YLIM"]:
        for ax in axs:
            ax.set_ylim(ctx.obj["YLIM"])

    if ctx.obj["YLOG"] is True:
        for ax in axs:
            ax.set_yscale('log')

    if ctx.obj["DESC"]:
        for ax in axs:
            l = ax.get_legend()
            # t = ''
            s = ctx.obj["DESC"]
            # if l.get_title():
            #     t = l.get_title().get_text()
            #     s += f'\n {t}'
            l.set_title(s)
            l._legend_box.align = "left"
            l.get_title().set_alpha(0.5)

            # from IPython import embed; embed()
            # ax.text(0.02, 0.87, s=ctx.obj["DESC"], alpha=0.5, transform=ax.transAxes,)

    legend = ctx.obj["LEGEND"]
    if legend is False:
        for ax in axs:
            ax.get_legend().remove()

    output = ctx.obj["OUTPUT"]
    if output:
        plt.savefig(output)
        if data is not None:
            n, _ = os.path.splitext(output)
            data.to_csv(n + '.csv', index=False, na_rep='NaN', )
    else:
        plt.show()



def _column_exists(path, column, key):
    with h5py.File(path, "r") as f:
        group = f.get(key)
        return column in group.keys()


def _load_data(path, cuts_path=None, dropna=True):
    cols = [
        'mc_energy',
        'mc_alt',
        'mc_az',
        'alt',
        'az',
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

    for col in ['gamma_energy_prediction_mean', 'gamma_prediction_mean']:
        if _column_exists(path, col, 'array_events'):
            cols.append(col)

    df, _, _ = load_signal_events(path, calculate_weights=False, columns=cols) 
    if dropna:
        df.dropna(inplace=True)
    if cuts_path:
        df = apply_cuts(df, cuts_path, theta_cuts=False, sigma=0)
    return df


@click.group(invoke_without_command=True, chain=True)
@click.option("--debug/--no-debug", default=False)
@click.option("--dropna/--no-dropna", default=True)
@click.option("--legend/--no-legend", default=True)
@click.option("--ylog/--no-ylog", default=True)
@click.option("--tag/--no-tag", default=True, help='flag to add informaiton text to plot')
@click.option("--ylim", default=None, nargs=2, type=np.float)
@click.option('-o', '--output', type=click.Path(exists=False))
@click.option('-c', '--cuts_path', type=click.Path(exists=True))
@click.argument('path', type=click.Path(exists=True))
@click.pass_context
def cli(ctx, path, debug, dropna, legend, ylog, ylim, tag, cuts_path, output):
    # ensure that ctx.obj exists and is a dict (in case `cli()` is called
    # by means other than the `if` block below
    # see https://click.palletsprojects.com/en/7.x/commands/#nested-handling-and-contexts
    ctx.ensure_object(dict)
    ctx.obj["DEBUG"] = debug
    ctx.obj["OUTPUT"] = output
    ctx.obj["LEGEND"] = legend
    ctx.obj["YLIM"] = ylim
    ctx.obj["YLOG"] = ylog
    data = _load_data(path, dropna=dropna, cuts_path=cuts_path)
    ctx.obj["DATA"] = data
    if tag:
        ctx.obj["DESC"] = load_data_description(path, data, cuts_path=cuts_path)

    if debug and ctx.invoked_subcommand is None:
        print("I was invoked without subcommand")
    elif debug:
        print(f"I am about to invoke {ctx.invoked_subcommand} with {ctx.obj}")


@cli.command()
@click.option('--reference/--no-reference', default=False)
@click.option('--plot_e_reco', is_flag=True, default=False)
@click.pass_context
def angular_resolution(ctx, reference, plot_e_reco):
    reconstructed_events = ctx.obj["DATA"]
    ylog = ctx.obj["YLOG"]
    ylim = ctx.obj["YLIM"]
    ax, df = plot_angular_resolution(reconstructed_events, reference, plot_e_reco, ylog=ylog, ylim=ylim)
    _apply_flags(ctx, ax, data=df)


@cli.command()
@click.option('--reference/--no-reference', default=False)
@click.option('--plot_e_reco', is_flag=True, default=False)
@click.pass_context
def angular_resolution_multiplicity(ctx, reference, plot_e_reco):
    reconstructed_events = ctx.obj["DATA"]
    ax = plot_angular_resolution_per_multiplicity(reconstructed_events, reference, plot_e_reco)
    _apply_flags(ctx, ax)


@cli.command()
@click.option('--color', default=main_color)
@click.option('--cmap', default=default_cmap)
@click.pass_context
def h_max(ctx, color, cmap):
    reconstructed_events = ctx.obj["DATA"]
    ax = plot_h_max(reconstructed_events, color=color, colormap=cmap)
    _apply_flags(ctx, ax)


@cli.command()
@click.option('--plot_e_reco', is_flag=True, default=True)
@click.option('--color', default=main_color)
@click.option('--cmap', default=default_cmap)
@click.pass_context
def h_max_distance(ctx, color, cmap):
    reconstructed_events = ctx.obj["DATA"]
    ax = plot_h_max_distance(reconstructed_events, color=color, colormap=cmap)
    _apply_flags(ctx, ax)


@cli.command()
@click.option('--color', default=main_color)
@click.option('--cmap', default=default_cmap)
@click.pass_context
def impact(ctx, color, cmap):
    reconstructed_events = ctx.obj["DATA"]
    ax = plot_impact(reconstructed_events, color=color, colormap=cmap)
    _apply_flags(ctx, ax)


@cli.command()
@click.option('--color', default=main_color)
@click.option('--cmap', default=default_cmap)
@click.pass_context
def impact_distance(ctx, color, cmap):
    reconstructed_events = ctx.obj["DATA"]
    ax = plot_impact_distance(reconstructed_events, color=color, colormap=cmap)
    _apply_flags(ctx, ax)



@cli.command()
@click.option('--reference/--no-reference', default=False)
@click.option('--method', default='relative', type=click.Choice(['cta', 'relative', 'absolute']))
@click.option('--plot_e_reco', is_flag=True, default=False)
@click.option('--plot_bias', is_flag=True, default=False)
@click.pass_context
def energy_resolution(ctx, reference, method, plot_e_reco, plot_bias):

    reconstructed_events = ctx.obj['DATA']

    e_true = reconstructed_events.mc_energy
    e_reco = reconstructed_events.gamma_energy_prediction_mean
    ax, df = plot_resolution(e_true, e_reco, reference=reference, method=method, plot_e_reco=plot_e_reco, plot_bias=plot_bias)
    ctx.obj["YLOG"] = False
    _apply_flags(ctx, ax, data=df)


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    cli(obj={})