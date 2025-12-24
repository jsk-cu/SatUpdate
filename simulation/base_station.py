#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base Station Class for Satellite Constellation Simulator

A ground station that can communicate with satellites when they have
line of sight and are within communication range.
"""

import math
import numpy as np
from typing import Optional, List, TYPE_CHECKING
from dataclasses import dataclass

from .orbit import EARTH_RADIUS_KM

if TYPE_CHECKING:
    from .satellite import Satellite


@dataclass
class BaseStationConfig:
    """
    Configuration for a base station.
    
    Attributes
    ----------
    latitude : float
        Latitude in radians (-π/2 to π/2, positive north)
    longitude : float
        Longitude in radians (-π to π, positive east)
    altitude : float
        Altitude above sea level in km (default 0)
    communication_range : float
        Maximum communication range in km (default 2000)
    name : str
        Name/identifier for the base station
    """
    latitude: float = 0.0
    longitude: float = 0.0
    altitude: float = 0.0
    communication_range: float = 2000.0
    name: str = "BASE-1"


class BaseStation:
    """
    A ground-based station that can communicate with satellites.
    
    The base station has a fixed position on Earth's surface and rotates
    with the Earth. It can communicate with satellites when:
    1. The satellite is within line of sight (not blocked by Earth)
    2. The satellite is within communication range
    """
    
    def __init__(
        self,
        config: Optional[BaseStationConfig] = None,
        earth_radius: float = EARTH_RADIUS_KM
    ):
        if config is None:
            config = BaseStationConfig()
        
        self.latitude = config.latitude
        self.longitude = config.longitude
        self.altitude = config.altitude
        self.communication_range = config.communication_range
        self.name = config.name
        self.earth_radius = earth_radius
    
    @classmethod
    def at_coordinates(
        cls,
        latitude_deg: float = 0.0,
        longitude_deg: float = 0.0,
        communication_range: float = 2000.0,
        name: str = "BASE-1"
    ) -> 'BaseStation':
        """
        Create a base station at the given geographic coordinates.
        
        Parameters
        ----------
        latitude_deg : float
            Latitude in degrees
        longitude_deg : float
            Longitude in degrees
        communication_range : float
            Communication range in km
        name : str
            Station name
        """
        config = BaseStationConfig(
            latitude=math.radians(latitude_deg),
            longitude=math.radians(longitude_deg),
            communication_range=communication_range,
            name=name
        )
        return cls(config)
    
    @property
    def latitude_deg(self) -> float:
        """Latitude in degrees."""
        return math.degrees(self.latitude)
    
    @property
    def longitude_deg(self) -> float:
        """Longitude in degrees."""
        return math.degrees(self.longitude)
    
    def get_position_ecef(self) -> np.ndarray:
        """
        Get position in Earth-Centered Earth-Fixed (ECEF) coordinates.
        """
        r = self.earth_radius + self.altitude
        
        x = r * math.cos(self.latitude) * math.cos(self.longitude)
        y = r * math.cos(self.latitude) * math.sin(self.longitude)
        z = r * math.sin(self.latitude)
        
        return np.array([x, y, z])
    
    def get_position_eci(self, earth_rotation_angle: float = 0.0) -> np.ndarray:
        """
        Get position in Earth-Centered Inertial (ECI) coordinates.
        
        Parameters
        ----------
        earth_rotation_angle : float
            Current rotation angle of Earth in radians.
        """
        r = self.earth_radius + self.altitude
        
        # Longitude in ECI frame = geographic longitude + Earth rotation
        lon_eci = self.longitude + earth_rotation_angle
        
        x = r * math.cos(self.latitude) * math.cos(lon_eci)
        y = r * math.cos(self.latitude) * math.sin(lon_eci)
        z = r * math.sin(self.latitude)
        
        return np.array([x, y, z])
    
    def distance_to_satellite(
        self,
        satellite: 'Satellite',
        earth_rotation_angle: float = 0.0
    ) -> float:
        """Calculate distance to a satellite."""
        pos_station = self.get_position_eci(earth_rotation_angle)
        pos_satellite = satellite.get_position_eci()
        return np.linalg.norm(pos_satellite - pos_station)
    
    def has_line_of_sight(
        self,
        satellite: 'Satellite',
        earth_rotation_angle: float = 0.0
    ) -> bool:
        """
        Check if the base station has line of sight to a satellite.
        """
        pos_station = self.get_position_eci(earth_rotation_angle)
        pos_satellite = satellite.get_position_eci()
        
        to_satellite = pos_satellite - pos_station
        distance = np.linalg.norm(to_satellite)
        
        if distance < 1e-6:
            return True
        
        to_satellite_normalized = to_satellite / distance
        t = -np.dot(pos_station, to_satellite_normalized)
        t_clamped = max(0, min(distance, t))
        closest_point = pos_station + t_clamped * to_satellite_normalized
        closest_dist = np.linalg.norm(closest_point)
        
        return closest_dist > self.earth_radius
    
    def is_in_range(
        self,
        satellite: 'Satellite',
        earth_rotation_angle: float = 0.0
    ) -> bool:
        """Check if a satellite is within communication range."""
        distance = self.distance_to_satellite(satellite, earth_rotation_angle)
        return distance <= self.communication_range
    
    def can_communicate(
        self,
        satellite: 'Satellite',
        earth_rotation_angle: float = 0.0
    ) -> bool:
        """
        Check if the base station can communicate with a satellite.
        
        Communication requires both line of sight AND being within range.
        """
        if not self.has_line_of_sight(satellite, earth_rotation_angle):
            return False
        if not self.is_in_range(satellite, earth_rotation_angle):
            return False
        return True
    
    def get_communicable_satellites(
        self,
        satellites: List['Satellite'],
        earth_rotation_angle: float = 0.0
    ) -> List['Satellite']:
        """
        Get list of satellites that can communicate with this station.
        
        Parameters
        ----------
        satellites : List[Satellite]
            List of satellites to check
        earth_rotation_angle : float
            Current Earth rotation angle
        
        Returns
        -------
        List[Satellite]
            Satellites that can communicate
        """
        return [
            sat for sat in satellites
            if self.can_communicate(sat, earth_rotation_angle)
        ]
    
    def __repr__(self) -> str:
        return (
            f"BaseStation({self.name}, "
            f"lat={self.latitude_deg:.2f}°, "
            f"lon={self.longitude_deg:.2f}°, "
            f"range={self.communication_range:.0f} km)"
        )