#!/usr/bin/env python3
"""
Base Station Module

Ground station implementation for satellite communication. Base stations
have fixed geographic positions on Earth's surface and rotate with the Earth.
They can communicate with satellites when:
1. Line of sight exists (signal path not blocked by Earth)
2. Distance is within communication range

The base station serves as the source of software update packets in the
satellite constellation simulation.
"""

import math
import numpy as np
from typing import Optional, List, Tuple, TYPE_CHECKING
from dataclasses import dataclass

from .orbit import EARTH_RADIUS_KM

if TYPE_CHECKING:
    from .satellite import Satellite


@dataclass
class BaseStationConfig:
    """
    Configuration for a ground station.

    Attributes
    ----------
    latitude : float
        Latitude in radians (-π/2 to π/2, positive north).
    longitude : float
        Longitude in radians (-π to π, positive east).
    altitude : float
        Altitude above sea level (km).
    communication_range : float
        Maximum communication range (km).
    min_elevation_angle : float
        Minimum elevation angle for communication (radians).
        Satellites below this angle from horizon are not reachable.
    name : str
        Station identifier.
    """

    latitude: float = 0.0
    longitude: float = 0.0
    altitude: float = 0.0
    communication_range: float = 2000.0
    min_elevation_angle: float = 0.0  # radians, 0 = horizon
    name: str = "BASE-1"


class BaseStation:
    """
    Ground station for satellite communication.

    The station has a fixed geographic position and rotates with Earth.
    It can communicate with satellites when:
    1. Line of sight exists (not blocked by Earth)
    2. Distance is within communication range
    3. Satellite elevation angle is above minimum threshold

    Parameters
    ----------
    config : BaseStationConfig, optional
        Station configuration. Uses defaults if not provided.
    earth_radius : float
        Earth's radius (km).

    Attributes
    ----------
    latitude : float
        Station latitude (radians).
    longitude : float
        Station longitude (radians).
    altitude : float
        Station altitude above sea level (km).
    communication_range : float
        Maximum communication range (km).
    min_elevation_angle : float
        Minimum elevation angle for communication (radians).
    name : str
        Station identifier.
    """

    def __init__(
        self,
        config: Optional[BaseStationConfig] = None,
        earth_radius: float = EARTH_RADIUS_KM,
    ):
        if config is None:
            config = BaseStationConfig()

        self.latitude = config.latitude
        self.longitude = config.longitude
        self.altitude = config.altitude
        self.communication_range = config.communication_range
        self.min_elevation_angle = config.min_elevation_angle
        self.name = config.name
        self.earth_radius = earth_radius

    @classmethod
    def at_coordinates(
        cls,
        latitude_deg: float = 0.0,
        longitude_deg: float = 0.0,
        altitude: float = 0.0,
        communication_range: float = 2000.0,
        min_elevation_deg: float = 0.0,
        name: str = "BASE-1",
    ) -> "BaseStation":
        """
        Create a base station at geographic coordinates.

        Parameters
        ----------
        latitude_deg : float
            Latitude in degrees.
        longitude_deg : float
            Longitude in degrees.
        altitude : float
            Altitude above sea level (km).
        communication_range : float
            Communication range (km).
        min_elevation_deg : float
            Minimum elevation angle in degrees (0 = horizon).
        name : str
            Station name.

        Returns
        -------
        BaseStation
            New base station instance.
        """
        config = BaseStationConfig(
            latitude=math.radians(latitude_deg),
            longitude=math.radians(longitude_deg),
            altitude=altitude,
            communication_range=communication_range,
            min_elevation_angle=math.radians(min_elevation_deg),
            name=name,
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

        Returns
        -------
        np.ndarray
            Position vector [x, y, z] in km.
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
            Current Earth rotation angle (radians).

        Returns
        -------
        np.ndarray
            Position vector [x, y, z] in km.
        """
        r = self.earth_radius + self.altitude
        lon_eci = self.longitude + earth_rotation_angle

        x = r * math.cos(self.latitude) * math.cos(lon_eci)
        y = r * math.cos(self.latitude) * math.sin(lon_eci)
        z = r * math.sin(self.latitude)

        return np.array([x, y, z])

    def distance_to_satellite(
        self, satellite: "Satellite", earth_rotation_angle: float = 0.0
    ) -> float:
        """
        Calculate distance to a satellite.

        Parameters
        ----------
        satellite : Satellite
            Target satellite.
        earth_rotation_angle : float
            Current Earth rotation angle (radians).

        Returns
        -------
        float
            Distance in km.
        """
        pos_station = self.get_position_eci(earth_rotation_angle)
        pos_satellite = satellite.get_position_eci()
        return np.linalg.norm(pos_satellite - pos_station)

    def elevation_angle(
        self, satellite: "Satellite", earth_rotation_angle: float = 0.0
    ) -> float:
        """
        Calculate elevation angle to a satellite.

        The elevation angle is the angle from the local horizon plane
        to the satellite. 0° = on horizon, 90° = directly overhead.

        Parameters
        ----------
        satellite : Satellite
            Target satellite.
        earth_rotation_angle : float
            Current Earth rotation angle (radians).

        Returns
        -------
        float
            Elevation angle in radians (-π/2 to π/2).
            Negative values indicate satellite is below horizon.
        """
        pos_station = self.get_position_eci(earth_rotation_angle)
        pos_satellite = satellite.get_position_eci()

        # Vector from station to satellite
        to_satellite = pos_satellite - pos_station
        distance = np.linalg.norm(to_satellite)

        if distance < 1e-6:
            return math.pi / 2  # Satellite at same position = directly overhead

        # Local "up" vector at station (points away from Earth center)
        up_vector = pos_station / np.linalg.norm(pos_station)

        # Direction to satellite
        to_satellite_normalized = to_satellite / distance

        # Elevation angle is complement of angle between up and to_satellite
        # sin(elevation) = cos(90° - elevation) = dot(up, to_sat)
        sin_elevation = np.dot(up_vector, to_satellite_normalized)

        # Clamp for numerical stability
        sin_elevation = max(-1.0, min(1.0, sin_elevation))

        return math.asin(sin_elevation)

    def elevation_angle_deg(
        self, satellite: "Satellite", earth_rotation_angle: float = 0.0
    ) -> float:
        """
        Calculate elevation angle to a satellite in degrees.

        Parameters
        ----------
        satellite : Satellite
            Target satellite.
        earth_rotation_angle : float
            Current Earth rotation angle (radians).

        Returns
        -------
        float
            Elevation angle in degrees.
        """
        return math.degrees(self.elevation_angle(satellite, earth_rotation_angle))

    def is_above_horizon(
        self, satellite: "Satellite", earth_rotation_angle: float = 0.0
    ) -> bool:
        """
        Check if satellite is above the minimum elevation angle.

        Parameters
        ----------
        satellite : Satellite
            Target satellite.
        earth_rotation_angle : float
            Current Earth rotation angle (radians).

        Returns
        -------
        bool
            True if satellite elevation >= min_elevation_angle.
        """
        elevation = self.elevation_angle(satellite, earth_rotation_angle)
        return elevation >= self.min_elevation_angle

    def has_line_of_sight(
        self, satellite: "Satellite", earth_rotation_angle: float = 0.0
    ) -> bool:
        """
        Check line of sight to a satellite.

        Parameters
        ----------
        satellite : Satellite
            Target satellite.
        earth_rotation_angle : float
            Current Earth rotation angle (radians).

        Returns
        -------
        bool
            True if line of sight exists.
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

        # Station is on Earth's surface; use tolerance for floating point
        return closest_dist >= self.earth_radius - 1.0

    def is_in_range(
        self, satellite: "Satellite", earth_rotation_angle: float = 0.0
    ) -> bool:
        """
        Check if satellite is within communication range.

        Parameters
        ----------
        satellite : Satellite
            Target satellite.
        earth_rotation_angle : float
            Current Earth rotation angle (radians).

        Returns
        -------
        bool
            True if within range.
        """
        distance = self.distance_to_satellite(satellite, earth_rotation_angle)
        return distance <= self.communication_range

    def can_communicate(
        self, satellite: "Satellite", earth_rotation_angle: float = 0.0
    ) -> bool:
        """
        Check if communication with satellite is possible.

        Requires:
        1. Line of sight (not blocked by Earth)
        2. Within communication range
        3. Above minimum elevation angle

        Parameters
        ----------
        satellite : Satellite
            Target satellite.
        earth_rotation_angle : float
            Current Earth rotation angle (radians).

        Returns
        -------
        bool
            True if communication is possible.
        """
        if not self.has_line_of_sight(satellite, earth_rotation_angle):
            return False
        if not self.is_in_range(satellite, earth_rotation_angle):
            return False
        if not self.is_above_horizon(satellite, earth_rotation_angle):
            return False
        return True

    def get_communicable_satellites(
        self, satellites: List["Satellite"], earth_rotation_angle: float = 0.0
    ) -> List["Satellite"]:
        """
        Get all satellites that can communicate with this station.

        Parameters
        ----------
        satellites : list
            List of satellites to check.
        earth_rotation_angle : float
            Current Earth rotation angle (radians).

        Returns
        -------
        list
            Satellites that can communicate.
        """
        return [
            sat
            for sat in satellites
            if self.can_communicate(sat, earth_rotation_angle)
        ]

    def __repr__(self) -> str:
        return (
            f"BaseStation({self.name}, "
            f"lat={self.latitude_deg:.2f}°, "
            f"lon={self.longitude_deg:.2f}°, "
            f"range={self.communication_range:.0f} km)"
        )