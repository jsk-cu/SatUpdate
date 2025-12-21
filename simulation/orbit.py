#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Elliptical Orbit Class for Satellite Constellation Simulator

Defines orbital mechanics for satellites using Keplerian orbital elements.
All distances in kilometers, angles in radians, time in seconds.
"""

import math
import numpy as np
from typing import Tuple

# Gravitational constant in km³/(kg·s²)
G = 6.67430e-20

# Standard Earth parameters
EARTH_RADIUS_KM = 6371.0  # Mean radius in km
EARTH_MASS_KG = 5.972e24  # Mass in kg


class EllipticalOrbit:
    """
    Defines an elliptical orbit using Keplerian orbital elements.
    
    Parameters
    ----------
    apoapsis : float
        Distance from Earth's center at the farthest point (km)
    periapsis : float
        Distance from Earth's center at the closest point (km)
    inclination : float
        Orbital plane tilt from equatorial plane (radians, 0 to π)
    longitude_of_ascending_node : float
        Angle from reference direction (vernal equinox) to ascending node (radians, 0 to 2π)
        Also known as RAAN (Right Ascension of Ascending Node) or Ω
    argument_of_periapsis : float
        Angle from ascending node to periapsis measured in orbital plane (radians, 0 to 2π)
        Also known as ω
    earth_radius : float
        Radius of Earth (km)
    earth_mass : float
        Mass of Earth (kg)
    
    Attributes
    ----------
    semi_major_axis : float
        Half the longest diameter of the ellipse (km)
    semi_minor_axis : float
        Half the shortest diameter of the ellipse (km)
    eccentricity : float
        Shape of ellipse (0 = circular, 0 < e < 1 = elliptical)
    period : float
        Time for one complete orbit (seconds)
    gravitational_parameter : float
        μ = G * M, the standard gravitational parameter (km³/s²)
    specific_angular_momentum : float
        Angular momentum per unit mass (km²/s)
    apoapsis_altitude : float
        Altitude above Earth's surface at apoapsis (km)
    periapsis_altitude : float
        Altitude above Earth's surface at periapsis (km)
    """
    
    def __init__(
        self,
        apoapsis: float,
        periapsis: float,
        inclination: float,
        longitude_of_ascending_node: float,
        argument_of_periapsis: float,
        earth_radius: float = EARTH_RADIUS_KM,
        earth_mass: float = EARTH_MASS_KG
    ):
        # Validate inputs
        if periapsis <= 0:
            raise ValueError("Periapsis must be positive")
        if apoapsis < periapsis:
            raise ValueError("Apoapsis must be greater than or equal to periapsis")
        if periapsis < earth_radius:
            raise ValueError("Periapsis cannot be below Earth's surface")
        if not 0 <= inclination <= math.pi:
            raise ValueError("Inclination must be between 0 and π radians")
        if earth_radius <= 0:
            raise ValueError("Earth radius must be positive")
        if earth_mass <= 0:
            raise ValueError("Earth mass must be positive")
        
        # Store primary orbital elements
        self.apoapsis = apoapsis
        self.periapsis = periapsis
        self.inclination = inclination
        self.longitude_of_ascending_node = longitude_of_ascending_node % (2 * math.pi)
        self.argument_of_periapsis = argument_of_periapsis % (2 * math.pi)
        
        # Store Earth parameters
        self.earth_radius = earth_radius
        self.earth_mass = earth_mass
        
        # Calculate derived parameters
        self._calculate_derived_parameters()
    
    def _calculate_derived_parameters(self) -> None:
        """Calculate all derived orbital parameters."""
        # Gravitational parameter μ = G * M
        self.gravitational_parameter = G * self.earth_mass
        
        # Semi-major axis: average of apoapsis and periapsis
        self.semi_major_axis = (self.apoapsis + self.periapsis) / 2
        
        # Eccentricity: measure of how elliptical the orbit is
        self.eccentricity = (self.apoapsis - self.periapsis) / (self.apoapsis + self.periapsis)
        
        # Semi-minor axis: b = a * sqrt(1 - e²)
        self.semi_minor_axis = self.semi_major_axis * math.sqrt(1 - self.eccentricity**2)
        
        # Orbital period from Kepler's third law: T = 2π * sqrt(a³/μ)
        self.period = 2 * math.pi * math.sqrt(self.semi_major_axis**3 / self.gravitational_parameter)
        
        # Specific angular momentum: h = sqrt(μ * a * (1 - e²))
        self.specific_angular_momentum = math.sqrt(
            self.gravitational_parameter * self.semi_major_axis * (1 - self.eccentricity**2)
        )
        
        # Altitudes above Earth's surface
        self.apoapsis_altitude = self.apoapsis - self.earth_radius
        self.periapsis_altitude = self.periapsis - self.earth_radius
    
    @property
    def mean_motion(self) -> float:
        """Mean angular velocity (radians/second)."""
        return 2 * math.pi / self.period
    
    def mean_anomaly_at_time(self, t: float, t0: float = 0, M0: float = 0) -> float:
        """
        Calculate mean anomaly at time t.
        
        Parameters
        ----------
        t : float
            Time (seconds)
        t0 : float
            Reference time (seconds)
        M0 : float
            Mean anomaly at reference time (radians)
        
        Returns
        -------
        float
            Mean anomaly at time t (radians)
        """
        return (M0 + self.mean_motion * (t - t0)) % (2 * math.pi)
    
    def eccentric_anomaly_from_mean(self, M: float, tolerance: float = 1e-10) -> float:
        """
        Calculate eccentric anomaly from mean anomaly using Newton-Raphson iteration.
        
        Solves Kepler's equation: M = E - e * sin(E)
        
        Parameters
        ----------
        M : float
            Mean anomaly (radians)
        tolerance : float
            Convergence tolerance
        
        Returns
        -------
        float
            Eccentric anomaly (radians)
        """
        e = self.eccentricity
        
        # Initial guess
        E = M if e < 0.8 else math.pi
        
        # Newton-Raphson iteration
        for _ in range(50):
            f = E - e * math.sin(E) - M
            f_prime = 1 - e * math.cos(E)
            E_new = E - f / f_prime
            
            if abs(E_new - E) < tolerance:
                return E_new
            E = E_new
        
        return E  # Return best estimate if not converged
    
    def true_anomaly_from_eccentric(self, E: float) -> float:
        """
        Calculate true anomaly from eccentric anomaly.
        
        Parameters
        ----------
        E : float
            Eccentric anomaly (radians)
        
        Returns
        -------
        float
            True anomaly (radians)
        """
        e = self.eccentricity
        
        # Using the half-angle formula for numerical stability
        return 2 * math.atan2(
            math.sqrt(1 + e) * math.sin(E / 2),
            math.sqrt(1 - e) * math.cos(E / 2)
        )
    
    def radius_at_true_anomaly(self, nu: float) -> float:
        """
        Calculate orbital radius at a given true anomaly.
        
        Parameters
        ----------
        nu : float
            True anomaly (radians)
        
        Returns
        -------
        float
            Distance from Earth's center (km)
        """
        a = self.semi_major_axis
        e = self.eccentricity
        return a * (1 - e**2) / (1 + e * math.cos(nu))
    
    def velocity_at_radius(self, r: float) -> float:
        """
        Calculate orbital velocity magnitude at a given radius (vis-viva equation).
        
        Parameters
        ----------
        r : float
            Distance from Earth's center (km)
        
        Returns
        -------
        float
            Orbital velocity magnitude (km/s)
        """
        mu = self.gravitational_parameter
        a = self.semi_major_axis
        return math.sqrt(mu * (2/r - 1/a))
    
    def position_in_orbital_plane(self, nu: float) -> Tuple[float, float]:
        """
        Calculate position in the orbital plane (perifocal coordinates).
        
        Parameters
        ----------
        nu : float
            True anomaly (radians)
        
        Returns
        -------
        tuple
            (x, y) position in orbital plane (km), x points to periapsis
        """
        r = self.radius_at_true_anomaly(nu)
        x = r * math.cos(nu)
        y = r * math.sin(nu)
        return x, y
    
    def velocity_in_orbital_plane(self, nu: float) -> Tuple[float, float]:
        """
        Calculate velocity in the orbital plane (perifocal coordinates).
        
        Parameters
        ----------
        nu : float
            True anomaly (radians)
        
        Returns
        -------
        tuple
            (vx, vy) velocity in orbital plane (km/s)
        """
        mu = self.gravitational_parameter
        h = self.specific_angular_momentum
        e = self.eccentricity
        
        vx = -mu / h * math.sin(nu)
        vy = mu / h * (e + math.cos(nu))
        return vx, vy
    
    def perifocal_to_eci_matrix(self) -> np.ndarray:
        """
        Get rotation matrix from perifocal (orbital plane) to ECI coordinates.
        
        Returns
        -------
        np.ndarray
            3x3 rotation matrix
        """
        i = self.inclination
        omega = self.argument_of_periapsis
        Omega = self.longitude_of_ascending_node
        
        cos_O = math.cos(Omega)
        sin_O = math.sin(Omega)
        cos_i = math.cos(i)
        sin_i = math.sin(i)
        cos_w = math.cos(omega)
        sin_w = math.sin(omega)
        
        # Rotation matrix components
        R = np.array([
            [cos_O * cos_w - sin_O * sin_w * cos_i,
             -cos_O * sin_w - sin_O * cos_w * cos_i,
             sin_O * sin_i],
            [sin_O * cos_w + cos_O * sin_w * cos_i,
             -sin_O * sin_w + cos_O * cos_w * cos_i,
             -cos_O * sin_i],
            [sin_w * sin_i,
             cos_w * sin_i,
             cos_i]
        ])
        
        return R
    
    def position_eci(self, nu: float) -> np.ndarray:
        """
        Calculate position in Earth-Centered Inertial (ECI) coordinates.
        
        Parameters
        ----------
        nu : float
            True anomaly (radians)
        
        Returns
        -------
        np.ndarray
            Position vector [x, y, z] in ECI frame (km)
        """
        x_pf, y_pf = self.position_in_orbital_plane(nu)
        r_perifocal = np.array([x_pf, y_pf, 0])
        R = self.perifocal_to_eci_matrix()
        return R @ r_perifocal
    
    def velocity_eci(self, nu: float) -> np.ndarray:
        """
        Calculate velocity in Earth-Centered Inertial (ECI) coordinates.
        
        Parameters
        ----------
        nu : float
            True anomaly (radians)
        
        Returns
        -------
        np.ndarray
            Velocity vector [vx, vy, vz] in ECI frame (km/s)
        """
        vx_pf, vy_pf = self.velocity_in_orbital_plane(nu)
        v_perifocal = np.array([vx_pf, vy_pf, 0])
        R = self.perifocal_to_eci_matrix()
        return R @ v_perifocal
    
    def state_at_time(
        self, 
        t: float, 
        t0: float = 0, 
        M0: float = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate position and velocity at a given time in ECI coordinates.
        
        Parameters
        ----------
        t : float
            Time (seconds)
        t0 : float
            Reference time (seconds)
        M0 : float
            Mean anomaly at reference time (radians)
        
        Returns
        -------
        tuple
            (position, velocity) where each is a numpy array [x, y, z] in km and km/s
        """
        M = self.mean_anomaly_at_time(t, t0, M0)
        E = self.eccentric_anomaly_from_mean(M)
        nu = self.true_anomaly_from_eccentric(E)
        
        position = self.position_eci(nu)
        velocity = self.velocity_eci(nu)
        
        return position, velocity
    
    def __repr__(self) -> str:
        return (
            f"EllipticalOrbit(\n"
            f"  apoapsis={self.apoapsis:.2f} km,\n"
            f"  periapsis={self.periapsis:.2f} km,\n"
            f"  inclination={math.degrees(self.inclination):.2f}°,\n"
            f"  RAAN={math.degrees(self.longitude_of_ascending_node):.2f}°,\n"
            f"  arg_periapsis={math.degrees(self.argument_of_periapsis):.2f}°,\n"
            f"  eccentricity={self.eccentricity:.6f},\n"
            f"  period={self.period:.2f} s ({self.period/60:.2f} min)\n"
            f")"
        )


if __name__ == "__main__":
    # Example: Create a GPS-like orbit
    altitude = 20200  # km
    
    gps_orbit = EllipticalOrbit(
        apoapsis=EARTH_RADIUS_KM + altitude,
        periapsis=EARTH_RADIUS_KM + altitude,
        inclination=math.radians(55),
        longitude_of_ascending_node=0,
        argument_of_periapsis=0,
    )
    
    print("GPS-like orbit:")
    print(gps_orbit)
    print(f"Period: {gps_orbit.period / 3600:.2f} hours")