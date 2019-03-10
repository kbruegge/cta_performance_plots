from sklearn.calibration import CalibratedClassifierCV
import pandas as pd
from sklearn.externals import joblib
import seaborn as sns
import matplotlib.pyplot as plt


def plot_importances(model_path, color, ax=None):
    model = joblib.load(model_path)
    feature_names = model.feature_names

    if isinstance(model, CalibratedClassifierCV):
        model = model.base_estimator

    feature_importances = [est.feature_importances_ for est in model.estimators_]

    df = pd.DataFrame(data=feature_importances, columns=feature_names)
    df = df.melt(var_name='feature', value_name='importance')

    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(10, 7),)

    sns.boxplot(y='feature', x='importance', data=df, color='gray', fliersize=0, ax=ax)
    sns.stripplot(y='feature', x='importance', data=df, jitter=True, color=color, alpha=0.3, ax=ax)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.set_ylabel('')
    ax.set_xlabel('Feature Importance')
    ax.tick_params(axis='y', width=0)

    return ax
