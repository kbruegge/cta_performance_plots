from sklearn.calibration import CalibratedClassifierCV
import pandas as pd
import matplotlib.pyplot as plt
import click
import yaml
from sklearn.externals import joblib
import seaborn as sns
from ..colors import main_color


def get_feature_names(config, type):
    with open(config) as f:
        d = yaml.load(f)
        r = list(d[type]['features'])
        s = list(d[type]['feature_generation']['features'].keys())
        return r + s


@click.command()
@click.argument(
    'model_path', type=click.Path(
        exists=True,
        dir_okay=False,
    ))
@click.option(
    '-o', '--output', type=click.Path(
        exists=False,
        dir_okay=False,
    ))
@click.option('-c', '--color', default=main_color)
def main(model_path, output, color):
    model = joblib.load(model_path)
    feature_names = model.feature_names

    if isinstance(model, CalibratedClassifierCV):
        model = model.base_estimator

    feature_importances = [est.feature_importances_ for est in model.estimators_]

    df = pd.DataFrame(data=feature_importances, columns=feature_names)
    df = df.melt(var_name='feature', value_name='importance')

    sns.boxplot(y='feature', x='importance', data=df, color='gray', fliersize=0)
    sns.stripplot(y='feature', x='importance', data=df, jitter=True, color=color, alpha=0.3)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.set_ylabel('')
    ax.set_xlabel('Feature Importance')
    ax.tick_params(axis='y', width=0)

    plt.tight_layout()

    if output:
        plt.savefig(output)
    else:
        plt.show()


if __name__ == '__main__':
    main()
