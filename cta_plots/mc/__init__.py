import pandas as pd
from pkg_resources import resource_string
from io import BytesIO


def load_background_reference():
    path = 'ascii/CTA-Performance-prod3b-v1-South-20deg-50h-BackgroundSqdeg.txt'
    r = resource_string('cta_plots.resources', path)
    df = pd.read_csv(
        BytesIO(r), delimiter='\t\t', skiprows=11, names=['e_min', 'e_max', 'rate'], engine='python'
    )
    return df


def load_energy_resolution_requirement(site='paranal'):
    path = 'ascii/CTA-Performance-prod3b-v1-South-20deg-50h-Eres.txt'
    r = resource_string('cta_plots.resources', path)
    df = pd.read_csv(
        BytesIO(r), delimiter='\t\t', skiprows=11, names=['energy', 'resolution'], engine='python'
    )
    return df


def load_angular_resolution_requirement(site='paranal'):
    path = 'ascii/CTA-Performance-prod3b-v1-South-20deg-50h-Angres.txt'
    r = resource_string('cta_plots.resources', path)
    df = pd.read_csv(
        BytesIO(r), delimiter='\t\t', skiprows=11, names=['energy', 'resolution'], engine='python'
    )
    return df


def load_effective_area_requirement(site='paranal'):
    path = 'ascii/CTA-Performance-prod3b-v1-South-20deg-50h-EffAreaNoDirectionCut.txt'
    r = resource_string('cta_plots.resources', path)
    df = pd.read_csv(
        BytesIO(r), delimiter='\t\t', skiprows=11, names=['energy', 'effective_area'], engine='python'
    )
    return df

