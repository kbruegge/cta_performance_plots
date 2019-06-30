from sklearn.calibration import CalibratedClassifierCV
import pandas as pd
from sklearn.externals import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import click


def plot_importances(model_path, color, ax=None):
    model = joblib.load(model_path)
    feature_names = model.feature_names

    if isinstance(model, CalibratedClassifierCV):
        model = model.base_estimator

    feature_importances = [est.feature_importances_ for est in model.estimators_]

    df = pd.DataFrame(data=feature_importances, columns=feature_names)
    df = df.melt(var_name="feature", value_name="importance")

    if not ax:
        fig, ax = plt.subplots(1, 1)

    sns.boxplot(y="feature", x="importance", data=df, color="gray", fliersize=0, ax=ax)
    sns.stripplot(
        y="feature", x="importance", data=df, jitter=True, color=color, alpha=0.3, ax=ax
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.set_ylabel("")
    ax.set_xlabel("Feature Importance")
    
    def shorten_name(f):
        if len(f) > 20:
            return f'{f[:17]}...'
        return f

    feature_names = [shorten_name(f) for f in feature_names]
    # from IPython import embed; embed()
    feature_names = [f'{{ \\texttt{{{s}}}}}' for s in feature_names]
    ax.set_yticklabels([f'{{ \\texttt{{{s}}}}}' for s in feature_names])
    ax.tick_params(axis="y", width=0)
    ax.tick_params(axis='y', which='major', labelsize=6)
    ax.tick_params(axis='y', which='minor', labelsize=6)

    for l in ax.get_yticklabels():
        l.set_ha('left')
    ax.get_yaxis().set_tick_params(pad=70)

    return ax



@click.command()
@click.argument("model", type=click.Path())
@click.option("-c", "--color", default="crimson")
@click.option("--xlim", default=None, nargs=2, type=float)
@click.option("-o", "--output")
def main(model, color, xlim, output):
    fig = plt.gcf()
    size = list(fig.get_size_inches())
    # print(size)
    size[1] += 1.5
    fig, ax = plt.subplots(1, 1, figsize=(size))
    size = list(fig.get_size_inches())
    # print(size)
    ax = plot_importances(model, color=color, ax=ax)
    plt.tight_layout(pad=0, rect=(-0.0135, 0, 1.006, 1))
    if xlim:
        plt.xlim(xlim)
    if output:
        plt.savefig(output)
    else:
        plt.show()
