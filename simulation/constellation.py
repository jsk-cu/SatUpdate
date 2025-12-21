#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Constellation Generation Module

Provides factory functions for creating satellite constellations:
- Random orbits
- Walker-Delta constellations
- Walker-Star constellations
"""

import math
import random
from typing import List, Optional, Tuple

from .orbit import EllipticalOrbit, EARTH_RADIUS_KM, EARTH_MASS_KG
from .satellite import Satellite


def create_circular_orbit(
    altitude: float,
    inclination: float,
    raan: float = 0.0,
    argument_of_periapsis: float = 0.0,
    earth_radius: float = EARTH_RADIUS_KM,
    earth_mass: float = EARTH_MASS_KG
) -> EllipticalOrbit:
    """
    Create a circular orbit at a given altitude.
    
    Parameters
    ----------
    altitude : float
        Orbital altitude above Earth's surface in km
    inclination : float
        Orbital inclination in radians
    raan : float
        Right ascension of ascending node in radians
    argument_of_periapsis : float
        Argument of periapsis in radians (usually 0 for circular orbits)
    earth_radius : float
        Earth radius in km
    earth_mass : float
        Earth mass in kg
    
    Returns
    -------
    EllipticalOrbit
        A circular orbit with the specified parameters
    """
    radius = earth_radius + altitude
    return EllipticalOrbit(
        apoapsis=radius,
        periapsis=radius,
        inclination=inclination,
        longitude_of_ascending_node=raan,
        argument_of_periapsis=argument_of_periapsis,
        earth_radius=earth_radius,
        earth_mass=earth_mass
    )


def create_random_orbit(
    min_periapsis_altitude: float = 300,
    max_periapsis_altitude: float = 2000,
    max_apoapsis_altitude: float = 40000,
    earth_radius: float = EARTH_RADIUS_KM,
    earth_mass: float = EARTH_MASS_KG
) -> EllipticalOrbit:
    """
    Create a random elliptical orbit with reasonable parameters.
    
    Parameters
    ----------
    min_periapsis_altitude : float
        Minimum periapsis altitude in km (default 300)
    max_periapsis_altitude : float
        Maximum periapsis altitude in km (default 2000)
    max_apoapsis_altitude : float
        Maximum apoapsis altitude in km (default 40000)
    earth_radius : float
        Earth radius in km
    earth_mass : float
        Earth mass in kg
    
    Returns
    -------
    EllipticalOrbit
        A randomly generated orbit
    """
    periapsis_altitude = random.uniform(min_periapsis_altitude, max_periapsis_altitude)
    apoapsis_altitude = periapsis_altitude + random.uniform(0, max_apoapsis_altitude - periapsis_altitude)
    
    inclination = random.uniform(0, math.pi)
    raan = random.uniform(0, 2 * math.pi)
    arg_periapsis = random.uniform(0, 2 * math.pi)
    
    return EllipticalOrbit(
        apoapsis=earth_radius + apoapsis_altitude,
        periapsis=earth_radius + periapsis_altitude,
        inclination=inclination,
        longitude_of_ascending_node=raan,
        argument_of_periapsis=arg_periapsis,
        earth_radius=earth_radius,
        earth_mass=earth_mass
    )


def create_random_constellation(
    num_satellites: int,
    min_periapsis_altitude: float = 300,
    max_periapsis_altitude: float = 2000,
    max_apoapsis_altitude: float = 40000,
    earth_radius: float = EARTH_RADIUS_KM,
    earth_mass: float = EARTH_MASS_KG,
    seed: Optional[int] = None
) -> Tuple[List[EllipticalOrbit], List[Satellite]]:
    """
    Create a constellation with random orbits.
    
    Parameters
    ----------
    num_satellites : int
        Number of satellites to create
    min_periapsis_altitude : float
        Minimum periapsis altitude in km
    max_periapsis_altitude : float
        Maximum periapsis altitude in km
    max_apoapsis_altitude : float
        Maximum apoapsis altitude in km
    earth_radius : float
        Earth radius in km
    earth_mass : float
        Earth mass in kg
    seed : Optional[int]
        Random seed for reproducibility
    
    Returns
    -------
    Tuple[List[EllipticalOrbit], List[Satellite]]
        Lists of orbits and satellites
    """
    if seed is not None:
        random.seed(seed)
    
    orbits = []
    satellites = []
    
    for i in range(num_satellites):
        orbit = create_random_orbit(
            min_periapsis_altitude=min_periapsis_altitude,
            max_periapsis_altitude=max_periapsis_altitude,
            max_apoapsis_altitude=max_apoapsis_altitude,
            earth_radius=earth_radius,
            earth_mass=earth_mass
        )
        orbits.append(orbit)
        
        initial_position = random.uniform(0, 1)
        satellite = Satellite(
            orbit=orbit,
            initial_position=initial_position,
            satellite_id=f"RND-{i+1:03d}"
        )
        satellites.append(satellite)
    
    return orbits, satellites


def create_walker_delta_constellation(
    num_planes: int,
    sats_per_plane: int,
    altitude: float,
    inclination: float,
    phasing_parameter: int = 1,
    earth_radius: float = EARTH_RADIUS_KM,
    earth_mass: float = EARTH_MASS_KG
) -> Tuple[List[EllipticalOrbit], List[Satellite]]:
    """
    Create a Walker-Delta constellation.
    
    In a Walker-Delta pattern (also called Walker-Star when inclination is near-polar),
    orbital planes are evenly distributed around the equator, and satellites in each
    plane are evenly spaced. The phasing parameter determines the relative phase
    offset between adjacent planes.
    
    Walker notation: i:T/P/F where:
    - i = inclination
    - T = total number of satellites
    - P = number of planes
    - F = phasing parameter (0 to P-1)
    
    Parameters
    ----------
    num_planes : int
        Number of orbital planes (P)
    sats_per_plane : int
        Number of satellites per plane
    altitude : float
        Orbital altitude in km (circular orbits)
    inclination : float
        Orbital inclination in radians
    phasing_parameter : int
        Phasing parameter F (0 to num_planes-1), determines phase offset between planes.
        F=0 means all planes have same phasing, F=1 is typical Walker Delta
    earth_radius : float
        Earth radius in km
    earth_mass : float
        Earth mass in kg
    
    Returns
    -------
    Tuple[List[EllipticalOrbit], List[Satellite]]
        Lists of orbits and satellites
    """
    total_satellites = num_planes * sats_per_plane
    
    # Normalize phasing parameter
    F = phasing_parameter % num_planes
    
    orbits = []
    satellites = []
    
    for plane in range(num_planes):
        # RAAN for this plane - evenly distributed around 360 degrees
        raan = 2 * math.pi * plane / num_planes
        
        # Create orbit for this plane
        orbit = create_circular_orbit(
            altitude=altitude,
            inclination=inclination,
            raan=raan,
            earth_radius=earth_radius,
            earth_mass=earth_mass
        )
        orbits.append(orbit)
        
        for sat in range(sats_per_plane):
            # Phase within plane - evenly spaced
            base_phase = sat / sats_per_plane
            
            # Walker Delta phasing: offset between planes
            # Phase offset = F * plane / (P * S) where S is sats per plane
            phase_offset = F * plane / total_satellites
            
            initial_position = (base_phase + phase_offset) % 1.0
            
            satellite = Satellite(
                orbit=orbit,
                initial_position=initial_position,
                satellite_id=f"WD-P{plane+1}S{sat+1}"
            )
            satellites.append(satellite)
    
    return orbits, satellites


def create_walker_star_constellation(
    num_planes: int,
    sats_per_plane: int,
    altitude: float,
    inclination: float = math.pi / 2,  # Default to polar orbit
    phasing_parameter: int = 1,
    earth_radius: float = EARTH_RADIUS_KM,
    earth_mass: float = EARTH_MASS_KG
) -> Tuple[List[EllipticalOrbit], List[Satellite]]:
    """
    Create a Walker-Star constellation.
    
    A Walker-Star constellation is similar to Walker-Delta but uses polar or
    near-polar inclinations, with orbital planes distributed over 180 degrees
    of RAAN rather than 360 degrees. This creates a "star" pattern when viewed
    from above the pole.
    
    Parameters
    ----------
    num_planes : int
        Number of orbital planes
    sats_per_plane : int
        Number of satellites per plane
    altitude : float
        Orbital altitude in km (circular orbits)
    inclination : float
        Orbital inclination in radians (default π/2 = 90° polar)
    phasing_parameter : int
        Phasing parameter F, determines phase offset between planes
    earth_radius : float
        Earth radius in km
    earth_mass : float
        Earth mass in kg
    
    Returns
    -------
    Tuple[List[EllipticalOrbit], List[Satellite]]
        Lists of orbits and satellites
    """
    total_satellites = num_planes * sats_per_plane
    
    # Normalize phasing parameter
    F = phasing_parameter % num_planes if num_planes > 0 else 0
    
    orbits = []
    satellites = []
    
    for plane in range(num_planes):
        # RAAN for this plane - distributed over 180 degrees (π radians)
        # This is the key difference from Walker-Delta
        raan = math.pi * plane / num_planes
        
        # Create orbit for this plane
        orbit = create_circular_orbit(
            altitude=altitude,
            inclination=inclination,
            raan=raan,
            earth_radius=earth_radius,
            earth_mass=earth_mass
        )
        orbits.append(orbit)
        
        for sat in range(sats_per_plane):
            # Phase within plane - evenly spaced
            base_phase = sat / sats_per_plane
            
            # Walker phasing offset between planes
            phase_offset = F * plane / total_satellites
            
            initial_position = (base_phase + phase_offset) % 1.0
            
            satellite = Satellite(
                orbit=orbit,
                initial_position=initial_position,
                satellite_id=f"WS-P{plane+1}S{sat+1}"
            )
            satellites.append(satellite)
    
    return orbits, satellites


def create_starlink_like_constellation(
    num_planes: int = 72,
    sats_per_plane: int = 22,
    altitude: float = 550,
    inclination_deg: float = 53.0,
    earth_radius: float = EARTH_RADIUS_KM,
    earth_mass: float = EARTH_MASS_KG
) -> Tuple[List[EllipticalOrbit], List[Satellite]]:
    """
    Create a Starlink-like LEO constellation.
    
    Based on SpaceX Starlink shell 1 parameters.
    
    Parameters
    ----------
    num_planes : int
        Number of orbital planes (default 72)
    sats_per_plane : int
        Satellites per plane (default 22)
    altitude : float
        Orbital altitude in km (default 550)
    inclination_deg : float
        Inclination in degrees (default 53.0)
    earth_radius : float
        Earth radius in km
    earth_mass : float
        Earth mass in kg
    
    Returns
    -------
    Tuple[List[EllipticalOrbit], List[Satellite]]
        Lists of orbits and satellites
    """
    return create_walker_delta_constellation(
        num_planes=num_planes,
        sats_per_plane=sats_per_plane,
        altitude=altitude,
        inclination=math.radians(inclination_deg),
        phasing_parameter=1,
        earth_radius=earth_radius,
        earth_mass=earth_mass
    )


def create_gps_like_constellation(
    earth_radius: float = EARTH_RADIUS_KM,
    earth_mass: float = EARTH_MASS_KG
) -> Tuple[List[EllipticalOrbit], List[Satellite]]:
    """
    Create a GPS-like MEO constellation.
    
    Based on GPS constellation: 6 planes, 4+ satellites per plane,
    20,200 km altitude, 55° inclination.
    
    Parameters
    ----------
    earth_radius : float
        Earth radius in km
    earth_mass : float
        Earth mass in kg
    
    Returns
    -------
    Tuple[List[EllipticalOrbit], List[Satellite]]
        Lists of orbits and satellites
    """
    return create_walker_delta_constellation(
        num_planes=6,
        sats_per_plane=4,
        altitude=20200,
        inclination=math.radians(55),
        phasing_parameter=1,
        earth_radius=earth_radius,
        earth_mass=earth_mass
    )


class ConstellationFactory:
    """
    Factory class providing static methods for constellation creation.
    Provides an object-oriented interface to constellation generators.
    """
    
    @staticmethod
    def random(
        num_satellites: int,
        min_periapsis_altitude: float = 300,
        max_periapsis_altitude: float = 2000,
        max_apoapsis_altitude: float = 40000,
        seed: Optional[int] = None
    ) -> Tuple[List[EllipticalOrbit], List[Satellite]]:
        """Create a random constellation."""
        return create_random_constellation(
            num_satellites=num_satellites,
            min_periapsis_altitude=min_periapsis_altitude,
            max_periapsis_altitude=max_periapsis_altitude,
            max_apoapsis_altitude=max_apoapsis_altitude,
            seed=seed
        )
    
    @staticmethod
    def walker_delta(
        num_planes: int,
        sats_per_plane: int,
        altitude: float,
        inclination: float,
        phasing_parameter: int = 1
    ) -> Tuple[List[EllipticalOrbit], List[Satellite]]:
        """Create a Walker-Delta constellation."""
        return create_walker_delta_constellation(
            num_planes=num_planes,
            sats_per_plane=sats_per_plane,
            altitude=altitude,
            inclination=inclination,
            phasing_parameter=phasing_parameter
        )
    
    @staticmethod
    def walker_star(
        num_planes: int,
        sats_per_plane: int,
        altitude: float,
        inclination: float = math.pi / 2,
        phasing_parameter: int = 1
    ) -> Tuple[List[EllipticalOrbit], List[Satellite]]:
        """Create a Walker-Star constellation."""
        return create_walker_star_constellation(
            num_planes=num_planes,
            sats_per_plane=sats_per_plane,
            altitude=altitude,
            inclination=inclination,
            phasing_parameter=phasing_parameter
        )
    
    @staticmethod
    def starlink_like(
        num_planes: int = 72,
        sats_per_plane: int = 22
    ) -> Tuple[List[EllipticalOrbit], List[Satellite]]:
        """Create a Starlink-like constellation."""
        return create_starlink_like_constellation(
            num_planes=num_planes,
            sats_per_plane=sats_per_plane
        )
    
    @staticmethod
    def gps_like() -> Tuple[List[EllipticalOrbit], List[Satellite]]:
        """Create a GPS-like constellation."""
        return create_gps_like_constellation()


if __name__ == "__main__":
    print("Constellation Generation Demo")
    print("=" * 60)
    
    # Random constellation
    print("\n1. Random Constellation (5 satellites):")
    orbits, satellites = create_random_constellation(5, seed=42)
    for sat in satellites:
        print(f"   {sat}")
    
    # Walker-Delta
    print("\n2. Walker-Delta Constellation (3 planes × 4 sats):")
    orbits, satellites = create_walker_delta_constellation(
        num_planes=3,
        sats_per_plane=4,
        altitude=550,
        inclination=math.radians(53)
    )
    for sat in satellites:
        print(f"   {sat}")
    
    # Walker-Star
    print("\n3. Walker-Star Constellation (4 planes × 3 sats, polar):")
    orbits, satellites = create_walker_star_constellation(
        num_planes=4,
        sats_per_plane=3,
        altitude=800,
        inclination=math.radians(86)
    )
    for sat in satellites:
        print(f"   {sat}")