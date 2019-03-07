import click
from cta_plots.irf.effective_area_3d import effective_area_3d_plot
from cta_plots.irf.background_3d import background_3d_plot
from cta_plots.irf.energy_dispersion_3d import energy_dispersion_3d_plot
from cta_plots.irf.psf_3d import psf_3d_plot
import matplotlib.pyplot as plt

@click.group(invoke_without_command=True)
@click.option('--debug/--no-debug', default=False)
@click.pass_context
def cli(ctx, debug):
    print(ctx.invoked_subcommand)
    ctx.ensure_object(dict)
    ctx.obj['DEBUG'] = debug
    if debug and ctx.invoked_subcommand is None:
        click.echo('I was invoked without subcommand')
    elif debug:
        click.echo(f'I am about to invoke {ctx.invoked_subcommand}')


@cli.command()
@click.argument('irf_file_path', type=click.Path())
@click.option('-o', '--output', type=click.Path(exists=False))
@click.pass_context
def effective_area(ctx, irf_file_path, output):
    fig = plt.figure(figsize=(10, 7),)
    ax = fig.add_subplot(111, projection='3d')
    effective_area_3d_plot(irf_file_path, ax=ax)

    if output:
        plt.savefig(output)
    else:
        plt.show()


@cli.command()
@click.argument('irf_file_path', type=click.Path())
@click.option('-o', '--output', type=click.Path(exists=False))
@click.pass_context
def background(ctx, irf_file_path, output):
    fig = plt.figure(figsize=(10, 7),)
    ax = fig.add_subplot(111, projection='3d')
    background_3d_plot(irf_file_path, ax=ax)

    if output:
        plt.savefig(output)
    else:
        plt.show()


@cli.command()
@click.argument('irf_file_path', type=click.Path())
@click.option('-o', '--output', type=click.Path(exists=False))
@click.pass_context
def energy_dispersion(ctx, irf_file_path, output):
    fig = plt.figure(figsize=(10, 7),)
    ax = fig.add_subplot(111, projection='3d')
    energy_dispersion_3d_plot(irf_file_path, ax=ax)

    if output:
        plt.savefig(output)
    else:
        plt.show()


@cli.command()
@click.argument('irf_file_path', type=click.Path())
@click.option('-o', '--output', type=click.Path(exists=False))
@click.pass_context
def psf(ctx, irf_file_path, output):
    fig = plt.figure(figsize=(10, 7),)
    ax = fig.add_subplot(111, projection='3d')
    psf_3d_plot(irf_file_path, ax=ax)

    if output:
        plt.savefig(output)
    else:
        plt.show()

# @click.group()
# @click.option('--debug/--no-debug', default=False)
# def cli(debug):
#     click.echo('Debug mode is %s' % ('on' if debug else 'off'))

# @cli.command()  # @cli, not @click!
# def sync():
#     click.echo('Syncing')

if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    cli(obj={})