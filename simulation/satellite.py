#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Satellite Class for Constellation Simulator

Each satellite has an orbit and a position within that orbit (0 to 1).
Position advances based on timesteps as a fraction of orbital period.
"""

import math
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

from .orbit import EllipticalOrbit, EARTH_RADIUS_KM, EARTH_MASS_KG


@dataclass
class GeospatialPosition:
    """
    Geospatial position of a satellite.
    
    Attributes
    ----------
    latitude : float
        Latitude in radians (-π/2 to π/2, positive north)
    longitude : float
        Longitude in radians (-π to π, positive east)
    altitude : float
        Altitude above Earth's surface in kilometers
    """
    latitude: float
    longitude: float
    altitude: float
    
    @property
    def latitude_deg(self) -> float:
        """Latitude in degrees."""
        return math.degrees(self.latitude)
    
    @property
    def longitude_deg(self) -> float:
        """Longitude in degrees."""
        return math.degrees(self.longitude)
    
    def __repr__(self) -> str:
        return (f"GeospatialPosition(lat={self.latitude_deg:.2f}°, "
                f"lon={self.longitude_deg:.2f}°, alt={self.altitude:.1f} km)")


class Satellite:
    """
    A satellite in orbit around Earth.
    
    The satellite's position in its orbit is tracked as a value from 0 to 1,
    where both 0 and 1 correspond to periapsis. This maps directly to mean
    anomaly (0 to 2π), which increases linearly with time.
    
    Parameters
    ----------
    orbit : EllipticalOrbit
        The orbital parameters for this satellite
    initial_position : float
        Initial position in orbit from 0 to 1 (default 0, at periapsis)
    satellite_id : Optional[str]
        Unique identifier for this satellite
    
    Attributes
    ----------
    orbit : EllipticalOrbit
        The satellite's orbital parameters
    position : float
        Current position in orbit (0 to 1, where 0 and 1 are periapsis)
    satellite_id : str
        Unique identifier
    elapsed_time : float
        Total elapsed simulation time in seconds
    """
    
    _id_counter = 0
    
    def __init__(
        self,
        orbit: EllipticalOrbit,
        initial_position: float = 0.0,
        satellite_id: Optional[str] = None
    ):
        self.orbit = orbit
        self._position = initial_position % 1.0  # Ensure in [0, 1)
        
        if satellite_id is None:
            Satellite._id_counter += 1
            self.satellite_id = f"SAT-{Satellite._id_counter:04d}"
        else:
            self.satellite_id = satellite_id
        
        self.elapsed_time = 0.0
    
    @property
    def position(self) -> float:
        """Current position in orbit (0 to 1)."""
        return self._position
    
    @position.setter
    def position(self, value: float):
        """Set position, automatically wrapping to [0, 1)."""
        self._position = value % 1.0
    
    @property
    def mean_anomaly(self) -> float:
        """Current mean anomaly in radians (0 to 2π)."""
        return self._position * 2 * math.pi
    
    @property
    def eccentric_anomaly(self) -> float:
        """Current eccentric anomaly in radians."""
        return self.orbit.eccentric_anomaly_from_mean(self.mean_anomaly)
    
    @property
    def true_anomaly(self) -> float:
        """Current true anomaly in radians."""
        return self.orbit.true_anomaly_from_eccentric(self.eccentric_anomaly)
    
    def step(self, timestep: float) -> None:
        """
        Advance the satellite's position by the given timestep.
        
        Parameters
        ----------
        timestep : float
            Time to advance in seconds
        """
        # Calculate fraction of orbit completed during this timestep
        orbit_fraction = timestep / self.orbit.period
        
        # Advance position
        self._position = (self._position + orbit_fraction) % 1.0
        
        # Track elapsed time
        self.elapsed_time += timestep
    
    def get_position_eci(self) -> np.ndarray:
        """
        Get current position in Earth-Centered Inertial (ECI) coordinates.
        
        Returns
        -------
        np.ndarray
            Position vector [x, y, z] in kilometers
        """
        return self.orbit.position_eci(self.true_anomaly)
    
    def get_velocity_eci(self) -> np.ndarray:
        """
        Get current velocity in Earth-Centered Inertial (ECI) coordinates.
        
        Returns
        -------
        np.ndarray
            Velocity vector [vx, vy, vz] in km/s
        """
        return self.orbit.velocity_eci(self.true_anomaly)
    
    def get_radius(self) -> float:
        """
        Get current distance from Earth's center.
        
        Returns
        -------
        float
            Distance in kilometers
        """
        return self.orbit.radius_at_true_anomaly(self.true_anomaly)
    
    def get_altitude(self) -> float:
        """
        Get current altitude above Earth's surface.
        
        Returns
        -------
        float
            Altitude in kilometers
        """
        return self.get_radius() - self.orbit.earth_radius
    
    def get_speed(self) -> float:
        """
        Get current orbital speed.
        
        Returns
        -------
        float
            Speed in km/s
        """
        return self.orbit.velocity_at_radius(self.get_radius())
    
    def get_geospatial_position(self, earth_rotation_angle: float = 0.0) -> GeospatialPosition:
        """
        Get current geospatial position (latitude, longitude, altitude).
        
        This converts from ECI coordinates to geographic coordinates,
        accounting for Earth's rotation if specified.
        
        Parameters
        ----------
        earth_rotation_angle : float
            Current rotation angle of Earth in radians (default 0).
            This is the angle the Prime Meridian has rotated from
            the ECI x-axis (vernal equinox direction).
            For real-time simulation: angle = GMST (Greenwich Mean Sidereal Time)
        
        Returns
        -------
        GeospatialPosition
            Latitude, longitude, and altitude
        """
        # Get ECI position
        pos_eci = self.get_position_eci()
        
        # Calculate radius and altitude
        radius = np.linalg.norm(pos_eci)
        altitude = radius - self.orbit.earth_radius
        
        # Calculate latitude (angle from equatorial plane)
        latitude = math.asin(pos_eci[2] / radius)
        
        # Calculate longitude in ECI frame
        longitude_eci = math.atan2(pos_eci[1], pos_eci[0])
        
        # Convert to geographic longitude by subtracting Earth's rotation
        longitude = (longitude_eci - earth_rotation_angle + math.pi) % (2 * math.pi) - math.pi
        
        return GeospatialPosition(
            latitude=latitude,
            longitude=longitude,
            altitude=altitude
        )
    
    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get full state (position and velocity) in ECI coordinates.
        
        Returns
        -------
        tuple
            (position, velocity) where each is [x, y, z] in km and km/s
        """
        return self.get_position_eci(), self.get_velocity_eci()
    
    def distance_to(self, other: 'Satellite') -> float:
        """
        Calculate distance to another satellite.
        
        Parameters
        ----------
        other : Satellite
            Another satellite
        
        Returns
        -------
        float
            Distance in kilometers
        """
        pos_self = self.get_position_eci()
        pos_other = other.get_position_eci()
        return np.linalg.norm(pos_self - pos_other)
    
    def has_line_of_sight(self, other: 'Satellite') -> bool:
        """
        Check if this satellite has line of sight to another satellite.
        
        Line of sight is blocked if the straight line between satellites
        passes through Earth.
        
        Parameters
        ----------
        other : Satellite
            Another satellite
        
        Returns
        -------
        bool
            True if line of sight exists
        """
        pos_self = self.get_position_eci()
        pos_other = other.get_position_eci()
        
        # Vector from self to other
        to_other = pos_other - pos_self
        distance = np.linalg.norm(to_other)
        
        if distance < 1e-6:
            return True  # Same position
        
        to_other_normalized = to_other / distance
        
        # Find closest approach to Earth center along the line segment
        t = -np.dot(pos_self, to_other_normalized)
        
        # Clamp t to the line segment [0, distance]
        t_clamped = max(0, min(distance, t))
        
        # Closest point on segment to Earth center
        closest_point = pos_self + t_clamped * to_other_normalized
        closest_dist = np.linalg.norm(closest_point)
        
        # Line of sight exists if closest point is outside Earth
        return closest_dist > self.orbit.earth_radius
    
    def __repr__(self) -> str:
        geo = self.get_geospatial_position()
        return (
            f"Satellite({self.satellite_id}, "
            f"pos={self.position:.4f}, "
            f"alt={geo.altitude:.0f} km, "
            f"lat={geo.latitude_deg:.1f}°, "
            f"lon={geo.longitude_deg:.1f}°)"
        )