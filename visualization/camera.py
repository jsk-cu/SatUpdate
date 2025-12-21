#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Camera Module for Satellite Visualization

Provides a 3D camera with spherical coordinates for viewing
satellite constellations around Earth.
"""

import math
import numpy as np
from typing import Tuple


class Camera:
    """
    3D camera with spherical coordinate controls.
    
    The camera orbits around the origin (Earth's center) and always
    looks toward the origin. Position is controlled by:
    - theta: longitude angle (horizontal rotation)
    - phi: latitude angle (vertical angle from equator)
    - distance: distance from origin
    
    Parameters
    ----------
    theta : float
        Initial longitude angle in radians (default 0.3)
    phi : float
        Initial latitude angle in radians (default π/5)
    distance : float
        Initial distance from origin (default 6.0)
    min_distance : float
        Minimum zoom distance (default 1.5)
    max_distance : float
        Maximum zoom distance (default 30.0)
    rotation_speed : float
        Speed of rotation in radians per input (default 0.03)
    zoom_speed : float
        Speed of zoom per input (default 0.15)
    
    Attributes
    ----------
    theta : float
        Current longitude angle (radians)
    phi : float
        Current latitude angle (radians)
    distance : float
        Current distance from origin
    """
    
    def __init__(
        self,
        theta: float = 0.3,
        phi: float = math.pi / 5,
        distance: float = 6.0,
        min_distance: float = 1.5,
        max_distance: float = 30.0,
        rotation_speed: float = 0.03,
        zoom_speed: float = 0.15
    ):
        self.theta = theta
        self.phi = phi
        self.distance = distance
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.rotation_speed = rotation_speed
        self.zoom_speed = zoom_speed
    
    def get_position(self) -> np.ndarray:
        """
        Get camera position in Cartesian coordinates.
        
        Returns
        -------
        np.ndarray
            Position vector [x, y, z]
        """
        x = self.distance * math.cos(self.phi) * math.cos(self.theta)
        y = self.distance * math.cos(self.phi) * math.sin(self.theta)
        z = self.distance * math.sin(self.phi)
        return np.array([x, y, z])
    
    def get_view_matrix(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the view direction and up/right vectors.
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (forward, up, right) unit vectors
        """
        pos = self.get_position()
        forward = -pos / np.linalg.norm(pos)
        
        world_up = np.array([0, 0, 1])
        right = np.cross(forward, world_up)
        
        if np.linalg.norm(right) < 0.001:
            # Camera is looking straight up or down
            right = np.array([1, 0, 0])
        else:
            right = right / np.linalg.norm(right)
        
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)
        
        return forward, up, right
    
    def rotate_left(self) -> None:
        """Rotate camera left (decrease theta)."""
        self.theta -= self.rotation_speed
        self._normalize_theta()
    
    def rotate_right(self) -> None:
        """Rotate camera right (increase theta)."""
        self.theta += self.rotation_speed
        self._normalize_theta()
    
    def rotate_up(self) -> None:
        """Rotate camera up (increase phi)."""
        self.phi = min(math.pi/2 - 0.01, self.phi + self.rotation_speed)
    
    def rotate_down(self) -> None:
        """Rotate camera down (decrease phi)."""
        self.phi = max(-math.pi/2 + 0.01, self.phi - self.rotation_speed)
    
    def zoom_in(self) -> None:
        """Zoom camera in (decrease distance)."""
        self.distance = max(self.min_distance, self.distance - self.zoom_speed)
    
    def zoom_out(self) -> None:
        """Zoom camera out (increase distance)."""
        self.distance = min(self.max_distance, self.distance + self.zoom_speed)
    
    def _normalize_theta(self) -> None:
        """Keep theta in [0, 2π) range."""
        self.theta = self.theta % (2 * math.pi)
    
    def set_position_spherical(
        self,
        theta: float,
        phi: float,
        distance: float
    ) -> None:
        """
        Set camera position using spherical coordinates.
        
        Parameters
        ----------
        theta : float
            Longitude angle in radians
        phi : float
            Latitude angle in radians
        distance : float
            Distance from origin
        """
        self.theta = theta % (2 * math.pi)
        self.phi = max(-math.pi/2 + 0.01, min(math.pi/2 - 0.01, phi))
        self.distance = max(self.min_distance, min(self.max_distance, distance))
    
    @property
    def theta_degrees(self) -> float:
        """Current longitude angle in degrees."""
        return math.degrees(self.theta) % 360
    
    @property
    def phi_degrees(self) -> float:
        """Current latitude angle in degrees."""
        return math.degrees(self.phi)
    
    def __repr__(self) -> str:
        return (
            f"Camera(θ={self.theta_degrees:.1f}°, "
            f"φ={self.phi_degrees:.1f}°, "
            f"dist={self.distance:.2f})"
        )