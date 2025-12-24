#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulation Class for Satellite Constellation Simulator

Provides a unified interface for running satellite constellation simulations
with integrated agent-based packet distribution protocol.
The simulation can be run independently of any visualization.
"""

import math
from typing import List, Optional, Tuple, Dict, Any, Set, Type
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
from .base_station import BaseStation, BaseStationConfig


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
    communication_range : Optional[float]
        Maximum communication range in km. If None, range is unlimited.
        Satellites must have line-of-sight AND be within this range to communicate.
    num_packets : int
        Number of packets in the software update to distribute (default 100)
    agent_class : Optional[Type]
        Agent class to use for packet distribution. If None, uses default Agent.
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
    communication_range: Optional[float] = None  # km, None = unlimited
    num_packets: int = 100  # Number of packets in software update
    agent_class: Optional[Type] = None  # Agent class to use


@dataclass
class AgentStatistics:
    """
    Statistics about agent packet distribution.
    
    Attributes
    ----------
    total_packets : int
        Total number of packets in the update
    packets_per_agent : Dict[int, int]
        Number of packets each agent has (agent_id -> count)
    completion_percentage : Dict[int, float]
        Completion percentage for each agent (agent_id -> percentage)
    fully_updated_count : int
        Number of agents that have all packets
    average_completion : float
        Average completion percentage across all satellite agents
    """
    total_packets: int = 0
    packets_per_agent: Dict[int, int] = field(default_factory=dict)
    completion_percentage: Dict[int, float] = field(default_factory=dict)
    fully_updated_count: int = 0
    average_completion: float = 0.0


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
    active_links : Set[Tuple[str, str]]
        Set of satellite ID pairs that currently have an active communication link.
        A link is active when both satellites have line-of-sight AND are within
        communication range. Pairs are stored as (sat1_id, sat2_id) where
        sat1_id < sat2_id alphabetically.
    base_station_links : Set[Tuple[str, str]]
        Set of (base_station_name, satellite_id) pairs with active communication.
    agent_statistics : AgentStatistics
        Statistics about agent packet distribution.
    """
    time: float = 0.0
    step_count: int = 0
    satellite_positions: Dict[str, GeospatialPosition] = field(default_factory=dict)
    active_links: Set[Tuple[str, str]] = field(default_factory=set)
    base_station_links: Set[Tuple[str, str]] = field(default_factory=set)
    agent_statistics: AgentStatistics = field(default_factory=AgentStatistics)


class Simulation:
    """
    Main simulation class for satellite constellation simulations.
    
    This class encapsulates all the logic for running a simulation,
    independent of any visualization. It can be used programmatically
    for batch simulations, testing, or analysis.
    
    The simulation includes an agent-based packet distribution protocol
    where each satellite and base station has an agent that manages
    packet requests and transfers.
    
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
    base_stations : List[BaseStation]
        List of base stations
    state : SimulationState
        Current state of the simulation
    earth_rotation_rate : float
        Earth's rotation rate in radians/second
    agents : Dict[int, Agent]
        Dictionary mapping agent IDs to Agent instances
    satellite_id_to_agent_id : Dict[str, int]
        Mapping from satellite string IDs to agent integer IDs
    agent_id_to_satellite_id : Dict[int, str]
        Mapping from agent integer IDs to satellite string IDs
    base_station_agent_id : int
        Agent ID for the base station (always 0)
    """
    
    # Earth's rotation rate: 360° per sidereal day (23h 56m 4s)
    EARTH_ROTATION_RATE = 2 * math.pi / 86164.0905  # rad/s
    
    # Base station always gets agent ID 0
    BASE_STATION_AGENT_ID = 0
    
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
        self.base_stations: List[BaseStation] = []
        self.state = SimulationState()
        self.earth_rotation_rate = self.EARTH_ROTATION_RATE
        
        # Agent-related attributes
        self.agents: Dict[int, Any] = {}  # agent_id -> Agent instance
        self.satellite_id_to_agent_id: Dict[str, int] = {}
        self.agent_id_to_satellite_id: Dict[int, str] = {}
        self.base_station_agent_id = self.BASE_STATION_AGENT_ID
        
        self._initialized = False
    
    def initialize(self) -> None:
        """
        Initialize the simulation by creating the constellation and agents.
        
        This must be called before stepping the simulation.
        """
        self._create_constellation()
        self._create_base_stations()
        self._create_agents()
        self._update_state()
        self._update_active_links()
        self._update_base_station_links()
        self._update_agent_statistics()
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
    
    def _create_base_stations(self) -> None:
        """Create default base station at 0,0 coordinates."""
        # Create a base station at 0 latitude, 0 longitude
        base_station = BaseStation.at_coordinates(
            latitude_deg=0.0,
            longitude_deg=0.0,
            communication_range=10000.0,
            name="BASE-1"
        )
        self.base_stations = [base_station]
    
    def _create_agents(self) -> None:
        """
        Create agents for the base station and all satellites.
        
        Agent ID assignment:
        - ID 0: Base station
        - IDs 1 to N: Satellites (in order they appear in self.satellites)
        """
        # Import Agent class (use provided class or default)
        if self.config.agent_class is not None:
            AgentClass = self.config.agent_class
        else:
            # Import default Agent class
            from agents import Agent
            AgentClass = Agent
        
        num_satellites = len(self.satellites)
        num_packets = self.config.num_packets
        
        # Clear existing agent mappings
        self.agents = {}
        self.satellite_id_to_agent_id = {}
        self.agent_id_to_satellite_id = {}
        
        # Create base station agent (ID 0)
        self.agents[self.BASE_STATION_AGENT_ID] = AgentClass(
            agent_id=self.BASE_STATION_AGENT_ID,
            num_packets=num_packets,
            num_satellites=num_satellites,
            is_base_station=True
        )
        
        # Create satellite agents (IDs 1 to N)
        for idx, satellite in enumerate(self.satellites):
            agent_id = idx + 1  # Satellite agents start at ID 1
            
            self.agents[agent_id] = AgentClass(
                agent_id=agent_id,
                num_packets=num_packets,
                num_satellites=num_satellites,
                is_base_station=False
            )
            
            # Create bidirectional mapping
            self.satellite_id_to_agent_id[satellite.satellite_id] = agent_id
            self.agent_id_to_satellite_id[agent_id] = satellite.satellite_id
    
    def _get_agent_neighbors(self, agent_id: int) -> Set[int]:
        """
        Get the set of neighbor agent IDs for a given agent.
        
        Neighbors are determined by:
        - For satellites: other satellites with active links + base station if in range
        - For base station: all satellites it can communicate with
        
        Parameters
        ----------
        agent_id : int
            The agent ID to get neighbors for
        
        Returns
        -------
        Set[int]
            Set of neighbor agent IDs
        """
        neighbors = set()
        
        if agent_id == self.BASE_STATION_AGENT_ID:
            # Base station's neighbors are satellites it can communicate with
            for bs_name, sat_id in self.state.base_station_links:
                if sat_id in self.satellite_id_to_agent_id:
                    neighbors.add(self.satellite_id_to_agent_id[sat_id])
        else:
            # Satellite's neighbors
            sat_id = self.agent_id_to_satellite_id.get(agent_id)
            if sat_id is None:
                return neighbors
            
            # Add other satellites with active links
            for id1, id2 in self.state.active_links:
                if id1 == sat_id:
                    if id2 in self.satellite_id_to_agent_id:
                        neighbors.add(self.satellite_id_to_agent_id[id2])
                elif id2 == sat_id:
                    if id1 in self.satellite_id_to_agent_id:
                        neighbors.add(self.satellite_id_to_agent_id[id1])
            
            # Add base station if in range
            for bs_name, linked_sat_id in self.state.base_station_links:
                if linked_sat_id == sat_id:
                    neighbors.add(self.BASE_STATION_AGENT_ID)
                    break
        
        return neighbors
    
    def _run_agent_protocol(self) -> None:
        """
        Run the 4-phase agent communication protocol.
        
        Phase 1: All agents broadcast their state
        Phase 2: All agents make requests based on neighbor broadcasts
        Phase 3: All agents receive requests and decide what to send
        Phase 4: All agents receive packets and update state
        """
        # Phase 1: Broadcast state
        broadcasts: Dict[int, Any] = {}
        for agent_id, agent in self.agents.items():
            broadcasts[agent_id] = agent.broadcast_state()
        
        # Phase 2: Make requests
        all_requests: Dict[int, Dict[int, int]] = {}  # agent_id -> {neighbor_id: packet_idx}
        for agent_id, agent in self.agents.items():
            neighbors = self._get_agent_neighbors(agent_id)
            neighbor_broadcasts = {
                neighbor_id: broadcasts[neighbor_id]
                for neighbor_id in neighbors
                if neighbor_id in broadcasts
            }
            all_requests[agent_id] = agent.make_requests(neighbor_broadcasts)
        
        # Phase 3: Receive requests and decide what to send
        # First, collect requests directed at each agent
        requests_to_agent: Dict[int, Dict[int, int]] = {
            agent_id: {} for agent_id in self.agents
        }
        for requester_id, requests in all_requests.items():
            for requestee_id, packet_idx in requests.items():
                if requestee_id in requests_to_agent:
                    requests_to_agent[requestee_id][requester_id] = packet_idx
        
        # Each agent processes requests and decides what to send
        responses: Dict[int, Dict[int, Optional[int]]] = {}
        for agent_id, agent in self.agents.items():
            responses[agent_id] = agent.receive_requests_and_update(
                requests_to_agent[agent_id]
            )
        
        # Phase 4: Deliver packets to requesters
        # Collect packets sent to each agent
        packets_to_agent: Dict[int, Dict[int, Optional[int]]] = {
            agent_id: {} for agent_id in self.agents
        }
        for requester_id, requests in all_requests.items():
            for requestee_id, requested_packet in requests.items():
                if requestee_id in responses:
                    sent_packet = responses[requestee_id].get(requester_id)
                    packets_to_agent[requester_id][requestee_id] = sent_packet
        
        # Each agent receives its packets
        for agent_id, agent in self.agents.items():
            agent.receive_packets_and_update(packets_to_agent[agent_id])
    
    def _update_state(self) -> None:
        """Update the simulation state with current satellite positions."""
        earth_rotation = self.state.time * self.earth_rotation_rate
        
        self.state.satellite_positions = {
            sat.satellite_id: sat.get_geospatial_position(earth_rotation)
            for sat in self.satellites
        }
    
    def _update_active_links(self) -> None:
        """
        Update the set of active communication links between satellites.
        
        A link is active when:
        1. Both satellites have line-of-sight to each other (not blocked by Earth)
        2. The distance between them is within communication_range (if set)
        
        The active_links set contains tuples of (sat1_id, sat2_id) where
        sat1_id < sat2_id alphabetically for consistent ordering.
        """
        active_links: Set[Tuple[str, str]] = set()
        comm_range = self.config.communication_range
        n = len(self.satellites)
        
        for i in range(n):
            for j in range(i + 1, n):
                sat_i = self.satellites[i]
                sat_j = self.satellites[j]
                
                # Check line of sight first (computationally cheaper to fail fast)
                if not sat_i.has_line_of_sight(sat_j):
                    continue
                
                # Check communication range if set
                if comm_range is not None:
                    distance = sat_i.distance_to(sat_j)
                    if distance > comm_range:
                        continue
                
                # Both conditions met - link is active
                # Store with consistent ordering (alphabetically by ID)
                id_i, id_j = sat_i.satellite_id, sat_j.satellite_id
                if id_i < id_j:
                    active_links.add((id_i, id_j))
                else:
                    active_links.add((id_j, id_i))
        
        self.state.active_links = active_links
    
    def _update_base_station_links(self) -> None:
        """
        Update the set of active communication links between base stations and satellites.
        """
        base_station_links: Set[Tuple[str, str]] = set()
        earth_rotation = self.state.time * self.earth_rotation_rate
        
        for base_station in self.base_stations:
            for satellite in self.satellites:
                if base_station.can_communicate(satellite, earth_rotation):
                    base_station_links.add((base_station.name, satellite.satellite_id))
        
        self.state.base_station_links = base_station_links
    
    def _update_agent_statistics(self) -> None:
        """Update statistics about agent packet distribution."""
        stats = AgentStatistics()
        stats.total_packets = self.config.num_packets
        
        satellite_completions = []
        fully_updated = 0
        
        for agent_id, agent in self.agents.items():
            packet_count = agent.get_packet_count()
            completion = agent.get_completion_percentage()
            
            stats.packets_per_agent[agent_id] = packet_count
            stats.completion_percentage[agent_id] = completion
            
            # Only count satellites (not base station) for completion stats
            if agent_id != self.BASE_STATION_AGENT_ID:
                satellite_completions.append(completion)
                if agent.has_all_packets():
                    fully_updated += 1
        
        stats.fully_updated_count = fully_updated
        if satellite_completions:
            stats.average_completion = sum(satellite_completions) / len(satellite_completions)
        
        self.state.agent_statistics = stats
    
    def step(self, timestep: float) -> SimulationState:
        """
        Advance the simulation by one timestep.
        
        This includes:
        1. Updating satellite positions
        2. Updating communication links
        3. Running the agent protocol
        4. Updating statistics
        
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
        
        # Update active communication links
        self._update_active_links()
        
        # Update base station links
        self._update_base_station_links()
        
        # Run agent protocol
        self._run_agent_protocol()
        
        # Update agent statistics
        self._update_agent_statistics()
        
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
                satellite_positions=dict(self.state.satellite_positions),
                active_links=set(self.state.active_links),
                base_station_links=set(self.state.base_station_links),
                agent_statistics=AgentStatistics(
                    total_packets=self.state.agent_statistics.total_packets,
                    packets_per_agent=dict(self.state.agent_statistics.packets_per_agent),
                    completion_percentage=dict(self.state.agent_statistics.completion_percentage),
                    fully_updated_count=self.state.agent_statistics.fully_updated_count,
                    average_completion=self.state.agent_statistics.average_completion
                )
            ))
            elapsed += timestep
        
        return states
    
    def reset(self) -> None:
        """
        Reset the simulation to initial state.
        
        Re-creates the constellation and agents with the same configuration.
        """
        self.state = SimulationState()
        self._create_constellation()
        self._create_base_stations()
        self._create_agents()
        self._update_state()
        self._update_active_links()
        self._update_base_station_links()
        self._update_agent_statistics()
    
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
        self._create_base_stations()
        self._create_agents()
        self._update_state()
        self._update_active_links()
        self._update_base_station_links()
        self._update_agent_statistics()
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
    
    def get_agent(self, agent_id: int) -> Optional[Any]:
        """
        Get an agent by its ID.
        
        Parameters
        ----------
        agent_id : int
            The agent's ID
        
        Returns
        -------
        Optional[Agent]
            The agent if found, None otherwise
        """
        return self.agents.get(agent_id)
    
    def get_satellite_agent(self, satellite_id: str) -> Optional[Any]:
        """
        Get the agent for a specific satellite.
        
        Parameters
        ----------
        satellite_id : str
            The satellite's string ID
        
        Returns
        -------
        Optional[Agent]
            The agent if found, None otherwise
        """
        agent_id = self.satellite_id_to_agent_id.get(satellite_id)
        if agent_id is not None:
            return self.agents.get(agent_id)
        return None
    
    def get_base_station_agent(self) -> Optional[Any]:
        """
        Get the base station's agent.
        
        Returns
        -------
        Optional[Agent]
            The base station agent
        """
        return self.agents.get(self.BASE_STATION_AGENT_ID)
    
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
    
    def is_update_complete(self) -> bool:
        """
        Check if all satellites have received all packets.
        
        Returns
        -------
        bool
            True if all satellites have complete update
        """
        return self.state.agent_statistics.fully_updated_count == len(self.satellites)
    
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
            "num_base_stations": len(self.base_stations),
            "num_packets": self.config.num_packets,
            "simulation_time": self.state.time,
            "step_count": self.state.step_count,
            "average_completion": self.state.agent_statistics.average_completion,
            "fully_updated_satellites": self.state.agent_statistics.fully_updated_count,
            "update_complete": self.is_update_complete(),
            "initialized": self._initialized,
        }
    
    def __repr__(self) -> str:
        return (
            f"Simulation(\n"
            f"  type={self.config.constellation_type.value},\n"
            f"  satellites={self.num_satellites},\n"
            f"  orbits={self.num_orbits},\n"
            f"  base_stations={len(self.base_stations)},\n"
            f"  packets={self.config.num_packets},\n"
            f"  time={self.state.time:.2f}s,\n"
            f"  steps={self.state.step_count},\n"
            f"  avg_completion={self.state.agent_statistics.average_completion:.1f}%\n"
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
    print("Simulation Class Demo with Agents")
    print("=" * 60)
    
    # Create a Walker-Delta simulation
    config = SimulationConfig(
        constellation_type=ConstellationType.WALKER_DELTA,
        num_planes=3,
        sats_per_plane=4,
        altitude=550,
        inclination=math.radians(53),
        num_packets=50  # 50 packets in the software update
    )
    
    sim = Simulation(config)
    sim.initialize()
    
    print(f"\nCreated simulation: {sim}")
    print(f"\nBase stations:")
    for bs in sim.base_stations:
        print(f"  {bs}")
    
    print(f"\nAgents:")
    print(f"  Base station agent: {sim.get_base_station_agent()}")
    print(f"  Satellite agents: {len(sim.satellites)}")
    
    print(f"\nInitial agent statistics:")
    stats = sim.state.agent_statistics
    print(f"  Total packets: {stats.total_packets}")
    print(f"  Average completion: {stats.average_completion:.1f}%")
    print(f"  Fully updated: {stats.fully_updated_count}/{len(sim.satellites)}")
    
    # Run simulation for 10 minutes
    print(f"\nRunning simulation for 10 minutes...")
    timestep = 60  # 1 minute steps
    for _ in range(10):
        sim.step(timestep)
    
    print(f"\nAfter 10 minutes:")
    print(f"  Simulation time: {sim.simulation_time} seconds")
    print(f"  Step count: {sim.state.step_count}")
    print(f"  Active links: {len(sim.state.active_links)}")
    print(f"  Base station links: {len(sim.state.base_station_links)}")
    
    stats = sim.state.agent_statistics
    print(f"\nAgent statistics after 10 steps:")
    print(f"  Average completion: {stats.average_completion:.1f}%")
    print(f"  Fully updated: {stats.fully_updated_count}/{len(sim.satellites)}")
    
    # Test convenience function
    print("\n" + "=" * 60)
    print("Testing convenience function:")
    sim2 = create_simulation("random", num_satellites=5, random_seed=42, num_packets=20)
    sim2.initialize()
    print(f"\nRandom simulation: {sim2}")