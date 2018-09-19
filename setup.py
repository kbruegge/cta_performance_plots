from setuptools import setup, find_packages

setup(
    name='cta_plots',
    version='0.0.1',
    description='a collection of plotting scrtipts for CTA',
    url='https://github.com/fact-project/fact_plots',
    author='Kai Brügge',
    author_email='kai.bruegge@tu-dortmund.de',
    license='BEER',
    packages=find_packages(),
    install_requires=[
        'click',
        'h5py',
        'matplotlib>=2.1',
        'numexpr',
        'numpy',
        'pandas',
        'pyfact>=0.20.1',
        'pytest',
        'python-dateutil',
        'pytz',
        'scipy',
        'tables',
        'tqdm',
    ],
    zip_safe=False,
    entry_points={
        'console_scripts': [
            'cta_plot_h_max_distance = cta_plots.h_max_distance:main',
            'cta_plot_energy_resolution = cta_plots.energy_resolution:main',
            'cta_plot_angular_resolution = cta_plots.angular_resolution_vs_energy:main',
            'cta_plot_impact_distance = cta_plots.impact_distance:main',
            'cta_plot_auc_per_type = cta_plots.ml.auc_per_type:main',
            'cta_plot_auc = cta_plots.ml.auc:main',
            'cta_plot_prediction_hist = cta_plots.ml.prediction_hist:main',
        ],
    }
)
