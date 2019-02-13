from astropy.coordinates import Angle
from astropy.coordinates.angle_utilities import angular_separation
import astropy.units as u

def calculate_distance_to_true_source_position(df):
    source_az = Angle(df.mc_az.values * u.deg).wrap_at(180 * u.deg)
    source_alt = Angle(df.mc_alt.values * u.deg)

    az = Angle(df.az.values*u.deg).wrap_at(180 * u.deg)
    alt = Angle(df.alt.values*u.deg)

    distance = angular_separation(source_az, source_alt, az, alt).to(u.deg)
    return distance