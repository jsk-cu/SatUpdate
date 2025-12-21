#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulation Class for Satellite Constellation Simulator

Provides a unified interface for running satellite constellation simulations.
The simulation can be run independently of any visualization.
"""

import math
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

from .orbit import EllipticalOrbit, EARTH_RADIUS_KM, EARTH_MASS_KG
from .satellite import Satellite, GeospatialPosition
from .constellation import (
    ConstellationFactory,
    create_random_constellation,
    create_walker_delta_constellation,
    create_walker_star_constellation,
)


class ConstellationType(Enum):
    """Enumeration of constellation types."""
    RANDOM = "random"
    WALKER_DELTA = "walker_delta"
    WALKER_STAR = "walker_star"
    CUSTOM = "custom"


@dataclass
class SimulationConfig:
    """
    Configuration for a simulation.
    
    Attributes
    ----------
    constellation_type : ConstellationType
        Type of constellation to generate
    num_planes : int
        Number of orbital planes (for Walker constellations)
    sats_per_plane : int
        Satellites per plane (for Walker constellations)
    num_satellites : int
        Total satellites (for random constellations)
    altitude : float
        Orbital altitude in km
    inclination : float
        Orbital inclination in radians
    phasing_parameter : int
        Walker phasing parameter F
    min_periapsis_altitude : float
        Minimum periapsis for random orbits
    max_periapsis_altitude : float
        Maximum periapsis for random orbits
    max_apoapsis_altitude : float
        Maximum apoapsis for random orbits
    earth_radius : float
        Earth radius in km
    earth_mass : float
        Earth mass in kg
    random_seed : Optional[int]
        Random seed for reproducibility
    """
    constellation_type: ConstellationType = ConstellationType.WALKER_DELTA
    num_planes: int = 3
    sats_per_plane: int = 4
    num_satellites: int = 12  # Used for random constellations
    altitude: float = 550.0
    inclination: float = math.radians(53)
    phasing_parameter: int = 1
    min_periapsis_altitude: float = 300.0
    max_periapsis_altitude: float = 2000.0
    max_apoapsis_altitude: float = 40000.0
    earth_radius: float = EARTH_RADIUS_KM
    earth_mass: float = EARTH_MASS_KG
    random_seed: Optional[int] = None


@dataclass
class SimulationState:
    """
    Current state of the simulation.
    
    Attributes
    ----------
    time : float
        Current simulation time in seconds
    step_count : int
        Number of simulation steps executed
    satellite_positions : Dict[str, GeospatialPosition]
        Current geospatial positions of all satellites
    """
    time: float = 0.0
    step_count: int = 0
    satellite_positions: Dict[str, GeospatialPosition] = field(default_factory=dict)


class Simulation:
    """
    Main simulation class for satellite constellation simulations.
    
    This class encapsulates all the logic for running a simulation,
    independent of any visualization. It can be used programmatically
    for batch simulations, testing, or analysis.
    
    Parameters
    ----------
    config : SimulationConfig
        Configuration for the simulation
    
    Attributes
    ----------
    config : SimulationConfig
        The simulation configuration
    orbits : List[EllipticalOrbit]
        List of orbital elements for each orbit
    satellites : List[Satellite]
        List of satellites in the constellation
    state : SimulationState
        Current state of the simulation
    earth_rotation_rate : float
        Earth's rotation rate in radians/second
    """
    
    # Earth's rotation rate: 360° per sidereal day (23h 56m 4s)
    EARTH_ROTATION_RATE = 2 * math.pi / 86164.0905  # rad/s
    
    def __init__(self, config: Optional[SimulationConfig] = None):
        """
        Initialize the simulation.
        
        Parameters
        ----------
        config : Optional[SimulationConfig]
            Configuration for the simulation. If None, uses default config.
        """
        self.config = config or SimulationConfig()
        self.orbits: List[EllipticalOrbit] = []
        self.satellites: List[Satellite] = []
        self.state = SimulationState()
        self.earth_rotation_rate = self.EARTH_ROTATION_RATE
        
        self._initialized = False
    
    def initialize(self) -> None:
        """
        Initialize the simulation by creating the constellation.
        
        This must be called before stepping the simulation.
        """
        self._create_constellation()
        self._update_state()
        self._initialized = True
    
    def _create_constellation(self) -> None:
        """Create the satellite constellation based on configuration."""
        config = self.config
        
        if config.constellation_type == ConstellationType.RANDOM:
            self.orbits, self.satellites = create_random_constellation(
                num_satellites=config.num_satellites,
                min_periapsis_altitude=config.min_periapsis_altitude,
                max_periapsis_altitude=config.max_periapsis_altitude,
                max_apoapsis_altitude=config.max_apoapsis_altitude,
                earth_radius=config.earth_radius,
                earth_mass=config.earth_mass,
                seed=config.random_seed
            )
        
        elif config.constellation_type == ConstellationType.WALKER_DELTA:
            self.orbits, self.satellites = create_walker_delta_constellation(
                num_planes=config.num_planes,
                sats_per_plane=config.sats_per_plane,
                altitude=config.altitude,
                inclination=config.inclination,
                phasing_parameter=config.phasing_parameter,
                earth_radius=config.earth_radius,
                earth_mass=config.earth_mass
            )
        
        elif config.constellation_type == ConstellationType.WALKER_STAR:
            self.orbits, self.satellites = create_walker_star_constellation(
                num_planes=config.num_planes,
                sats_per_plane=config.sats_per_plane,
                altitude=config.altitude,
                inclination=config.inclination,
                phasing_parameter=config.phasing_parameter,
                earth_radius=config.earth_radius,
                earth_mass=config.earth_mass
            )
        
        elif config.constellation_type == ConstellationType.CUSTOM:
            # For custom constellations, orbits and satellites should be set externally
            pass
    
    def _update_state(self) -> None:
        """Update the simulation state with current satellite positions."""
        earth_rotation = self.state.time * self.earth_rotation_rate
        
        self.state.satellite_positions = {
            sat.satellite_id: sat.get_geospatial_position(earth_rotation)
            for sat in self.satellites
        }
    
    def step(self, timestep: float) -> SimulationState:
        """
        Advance the simulation by one timestep.
        
        Parameters
        ----------
        timestep : float
            Time to advance in seconds
        
        Returns
        -------
        SimulationState
            The updated simulation state
        
        Raises
        ------
        RuntimeError
            If the simulation has not been initialized
        """
        if not self._initialized:
            raise RuntimeError("Simulation not initialized. Call initialize() first.")
        
        # Update all satellites
        for satellite in self.satellites:
            satellite.step(timestep)
        
        # Update simulation time
        self.state.time += timestep
        self.state.step_count += 1
        
        # Update state
        self._update_state()
        
        return self.state
    
    def run(self, duration: float, timestep: float) -> List[SimulationState]:
        """
        Run the simulation for a specified duration.
        
        Parameters
        ----------
        duration : float
            Total simulation time in seconds
        timestep : float
            Time step in seconds
        
        Returns
        -------
        List[SimulationState]
            List of simulation states at each timestep
        """
        if not self._initialized:
            self.initialize()
        
        states = []
        elapsed = 0.0
        
        while elapsed < duration:
            self.step(timestep)
            states.append(SimulationState(
                time=self.state.time,
                step_count=self.state.step_count,
                satellite_positions=dict(self.state.satellite_positions)
            ))
            elapsed += timestep
        
        return states
    
    def reset(self) -> None:
        """
        Reset the simulation to initial state.
        
        Re-creates the constellation with the same configuration.
        """
        self.state = SimulationState()
        self._create_constellation()
        self._update_state()
    
    def regenerate(self, new_seed: Optional[int] = None) -> None:
        """
        Regenerate the constellation with a new random seed.
        
        Parameters
        ----------
        new_seed : Optional[int]
            New random seed. If None, uses a random seed.
        """
        if new_seed is not None:
            self.config.random_seed = new_seed
        else:
            import random
            self.config.random_seed = random.randint(0, 2**31)
        
        self.reset()
    
    def set_custom_constellation(
        self,
        orbits: List[EllipticalOrbit],
        satellites: List[Satellite]
    ) -> None:
        """
        Set a custom constellation.
        
        Parameters
        ----------
        orbits : List[EllipticalOrbit]
            List of orbits
        satellites : List[Satellite]
            List of satellites
        """
        self.config.constellation_type = ConstellationType.CUSTOM
        self.orbits = orbits
        self.satellites = satellites
        self.state = SimulationState()
        self._update_state()
        self._initialized = True
    
    def get_satellite(self, satellite_id: str) -> Optional[Satellite]:
        """
        Get a satellite by its ID.
        
        Parameters
        ----------
        satellite_id : str
            The satellite's ID
        
        Returns
        -------
        Optional[Satellite]
            The satellite if found, None otherwise
        """
        for sat in self.satellites:
            if sat.satellite_id == satellite_id:
                return sat
        return None
    
    def get_inter_satellite_distances(self) -> Dict[Tuple[str, str], float]:
        """
        Calculate distances between all satellite pairs.
        
        Returns
        -------
        Dict[Tuple[str, str], float]
            Dictionary mapping (sat1_id, sat2_id) to distance in km
        """
        distances = {}
        n = len(self.satellites)
        
        for i in range(n):
            for j in range(i + 1, n):
                sat_i = self.satellites[i]
                sat_j = self.satellites[j]
                dist = sat_i.distance_to(sat_j)
                distances[(sat_i.satellite_id, sat_j.satellite_id)] = dist
        
        return distances
    
    def get_line_of_sight_matrix(self) -> Dict[Tuple[str, str], bool]:
        """
        Calculate line-of-sight between all satellite pairs.
        
        Returns
        -------
        Dict[Tuple[str, str], bool]
            Dictionary mapping (sat1_id, sat2_id) to LOS status
        """
        los_matrix = {}
        n = len(self.satellites)
        
        for i in range(n):
            for j in range(i + 1, n):
                sat_i = self.satellites[i]
                sat_j = self.satellites[j]
                has_los = sat_i.has_line_of_sight(sat_j)
                los_matrix[(sat_i.satellite_id, sat_j.satellite_id)] = has_los
        
        return los_matrix
    
    @property
    def num_satellites(self) -> int:
        """Number of satellites in the simulation."""
        return len(self.satellites)
    
    @property
    def num_orbits(self) -> int:
        """Number of unique orbits in the simulation."""
        return len(self.orbits)
    
    @property
    def simulation_time(self) -> float:
        """Current simulation time in seconds."""
        return self.state.time
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the simulation configuration and state.
        
        Returns
        -------
        Dict[str, Any]
            Summary dictionary
        """
        return {
            "constellation_type": self.config.constellation_type.value,
            "num_satellites": self.num_satellites,
            "num_orbits": self.num_orbits,
            "simulation_time": self.state.time,
            "step_count": self.state.step_count,
            "initialized": self._initialized,
        }
    
    def __repr__(self) -> str:
        return (
            f"Simulation(\n"
            f"  type={self.config.constellation_type.value},\n"
            f"  satellites={self.num_satellites},\n"
            f"  orbits={self.num_orbits},\n"
            f"  time={self.state.time:.2f}s,\n"
            f"  steps={self.state.step_count}\n"
            f")"
        )


# Convenience functions for quick simulation creation

def create_simulation(
    constellation_type: str = "walker_delta",
    **kwargs
) -> Simulation:
    """
    Create a simulation with the specified constellation type.
    
    Parameters
    ----------
    constellation_type : str
        One of "random", "walker_delta", "walker_star"
    **kwargs
        Additional configuration parameters
    
    Returns
    -------
    Simulation
        Configured simulation (not yet initialized)
    """
    type_map = {
        "random": ConstellationType.RANDOM,
        "walker_delta": ConstellationType.WALKER_DELTA,
        "walker_star": ConstellationType.WALKER_STAR,
    }
    
    if constellation_type not in type_map:
        raise ValueError(f"Unknown constellation type: {constellation_type}")
    
    config = SimulationConfig(constellation_type=type_map[constellation_type])
    
    # Apply any additional configuration
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return Simulation(config)


if __name__ == "__main__":
    print("Simulation Class Demo")
    print("=" * 60)
    
    # Create a Walker-Delta simulation
    config = SimulationConfig(
        constellation_type=ConstellationType.WALKER_DELTA,
        num_planes=3,
        sats_per_plane=4,
        altitude=550,
        inclination=math.radians(53)
    )
    
    sim = Simulation(config)
    sim.initialize()
    
    print(f"\nCreated simulation: {sim}")
    print(f"\nInitial satellite positions:")
    for sat_id, geo in sim.state.satellite_positions.items():
        print(f"  {sat_id}: {geo}")
    
    # Run simulation for 10 minutes
    print(f"\nRunning simulation for 10 minutes...")
    timestep = 60  # 1 minute steps
    for _ in range(10):
        sim.step(timestep)
    
    print(f"\nAfter 10 minutes:")
    print(f"  Simulation time: {sim.simulation_time} seconds")
    print(f"  Step count: {sim.state.step_count}")
    
    # Get inter-satellite distances
    print(f"\nSample inter-satellite distances:")
    distances = sim.get_inter_satellite_distances()
    for (sat1, sat2), dist in list(distances.items())[:5]:
        print(f"  {sat1} <-> {sat2}: {dist:.1f} km")
    
    # Test convenience function
    print("\n" + "=" * 60)
    print("Testing convenience function:")
    sim2 = create_simulation("random", num_satellites=5, random_seed=42)
    sim2.initialize()
    print(f"\nRandom simulation: {sim2}")