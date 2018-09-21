import click
import matplotlib.pyplot as plt

import fact.io
from .colors import default_cmap, main_color
from matplotlib.colors import LogNorm


@click.command()
@click.argument('input_dl3_file', type=click.Path(exists=True))
@click.option('-o', '--output', type=click.Path(exists=False))
@click.option('-m', '--multiplicity', default=2)
@click.option('-cm', '--colormap', default=default_cmap)
@click.option('-c', '--color', default=main_color)
def main(input_dl3_file, output, multiplicity, colormap, color):
    df = fact.io.read_data(input_dl3_file, key='array_events', ).dropna()

    if multiplicity > 2:
        df = df.query(f'num_triggered_telescopes >= {multiplicity}')

    x = df.mc_core_x - df.core_x_prediction
    y = df.mc_core_y - df.core_y_prediction

    x_min = -50
    x_max = 50
    plt.hexbin(x, y, extent=(x_min, x_max, x_min, x_max), cmap=colormap, norm=LogNorm())
    #
    plt.ylim([x_min, x_max])
    plt.xlim([x_min, x_max])
    plt.colorbar()

    plt.xlabel('x offset to true impact / meter')
    plt.ylabel('y offset to true impact / meter')
    ax = plt.gca()
    ax.tick_params(axis='x', which='major', pad=7)
    plt.tight_layout()
    if output:
        plt.savefig(output)
    else:
        plt.show()


if __name__ == "__main__":
    main()
