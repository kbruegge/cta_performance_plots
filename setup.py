from setuptools import setup, find_packages

setup(
    name='cta_plots',
    version='0.0.2',
    description='A collection of plotting scrtipts for CTA',
    url='https://github.com/mackaiver/cta_performance_plots',
    author='Kai Brügge',
    author_email='kai.bruegge@tu-dortmund.de',
    license='BEER',
    package_data={
        'resources/ascii': ['*.txt'],
        'resources/': ['*.txt'],
    },
    packages=find_packages(),
    include_package_data=True,
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
        'seaborn',
        'colorama',
    ],
    zip_safe=False,
    entry_points={
        'console_scripts': [
            'cta_plot_effective_area = cta_plots.mc.effective_area:main',
            'cta_plot_sensitivity = cta_plots.mc.sensitivity:main',
            'cta_plot_theta_square = cta_plots.mc.theta_squared:main',
            'cta_plot_triggered_counts = cta_plots.mc.triggered_counts:main',
            'cta_plot_ml = cta_plots.ml.ml_cli:cli',
            'cta_plot_reco = cta_plots.reconstruction.reco_cli:cli',
            'cta_plot_irf = cta_plots.irf.irf_cli:cli',
        ],
    }
)
