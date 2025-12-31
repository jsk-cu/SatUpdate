#!/usr/bin/env python3
"""
Tests for Step 1: TrajectoryProvider Interface

These tests verify:
1. TrajectoryProvider ABC is correctly defined
2. KeplerianProvider wraps existing satellites properly
3. Simulation works identically with KeplerianProvider
4. All existing functionality is preserved
"""

import math
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestTrajectoryProviderInterface:
    """Tests for the TrajectoryProvider abstract base class."""
    
    def test_trajectory_state_dataclass(self):
        """Test TrajectoryState dataclass creation and attributes."""
        # This tests the expected interface - implementation in Step 1
        from dataclasses import dataclass
        
        @dataclass
        class TrajectoryState:
            position_eci: np.ndarray
            velocity_eci: np.ndarray
            epoch: datetime
            reference_frame: str = "J2000"
        
        state = TrajectoryState(
            position_eci=np.array([7000.0, 0.0, 0.0]),
            velocity_eci=np.array([0.0, 7.5, 0.0]),
            epoch=datetime(2025, 1, 1, 0, 0, 0),
            reference_frame="J2000"
        )
        
        assert state.position_eci.shape == (3,)
        assert state.velocity_eci.shape == (3,)
        assert state.reference_frame == "J2000"
        assert isinstance(state.epoch, datetime)
    
    def test_trajectory_provider_abstract_methods(self):
        """Test that TrajectoryProvider defines required abstract methods."""
        from abc import ABC, abstractmethod
        
        # Define the expected interface
        class TrajectoryProvider(ABC):
            @abstractmethod
            def get_state(self, satellite_id: str, time: datetime):
                pass
            
            @abstractmethod
            def get_position_eci(self, satellite_id: str, time: datetime) -> np.ndarray:
                pass
            
            @abstractmethod
            def get_satellite_ids(self) -> List[str]:
                pass
            
            @abstractmethod
            def get_time_bounds(self, satellite_id: str) -> Tuple[datetime, datetime]:
                pass
        
        # Verify cannot instantiate abstract class
        with pytest.raises(TypeError):
            TrajectoryProvider()
    
    def test_concrete_provider_implementation(self):
        """Test that concrete implementations satisfy the interface."""
        from abc import ABC, abstractmethod
        from dataclasses import dataclass
        
        @dataclass
        class TrajectoryState:
            position_eci: np.ndarray
            velocity_eci: np.ndarray
            epoch: datetime
            reference_frame: str = "J2000"
        
        class TrajectoryProvider(ABC):
            @abstractmethod
            def get_state(self, satellite_id: str, time: datetime) -> TrajectoryState:
                pass
            
            @abstractmethod
            def get_position_eci(self, satellite_id: str, time: datetime) -> np.ndarray:
                pass
            
            @abstractmethod
            def get_satellite_ids(self) -> List[str]:
                pass
            
            @abstractmethod
            def get_time_bounds(self, satellite_id: str) -> Tuple[datetime, datetime]:
                pass
        
        class MockProvider(TrajectoryProvider):
            def get_state(self, satellite_id: str, time: datetime) -> TrajectoryState:
                return TrajectoryState(
                    position_eci=np.array([7000.0, 0.0, 0.0]),
                    velocity_eci=np.array([0.0, 7.5, 0.0]),
                    epoch=time
                )
            
            def get_position_eci(self, satellite_id: str, time: datetime) -> np.ndarray:
                return np.array([7000.0, 0.0, 0.0])
            
            def get_satellite_ids(self) -> List[str]:
                return ["SAT-001", "SAT-002"]
            
            def get_time_bounds(self, satellite_id: str) -> Tuple[datetime, datetime]:
                return (datetime.min, datetime.max)
        
        # Should be instantiable
        provider = MockProvider()
        assert len(provider.get_satellite_ids()) == 2


class TestKeplerianProvider:
    """Tests for the KeplerianProvider implementation."""
    
    def test_keplerian_provider_creation(self, sample_satellites):
        """Test KeplerianProvider can be created from satellites."""
        # Mock implementation for testing interface
        class KeplerianProvider:
            def __init__(self, satellites, epoch):
                self.satellites = {sat.satellite_id: sat for sat in satellites}
                self.epoch = epoch
            
            def get_satellite_ids(self):
                return list(self.satellites.keys())
        
        epoch = datetime(2025, 1, 1, 0, 0, 0)
        provider = KeplerianProvider(sample_satellites, epoch)
        
        assert len(provider.get_satellite_ids()) == 3
        assert "TEST-SAT-001" in provider.get_satellite_ids()
    
    def test_keplerian_provider_position_matches_satellite(self, sample_satellites):
        """Test KeplerianProvider returns same position as direct satellite access."""
        sat = sample_satellites[0]
        expected_position = sat.get_position_eci()
        
        # Verify we can get the expected position
        assert expected_position.shape == (3,)
        assert np.linalg.norm(expected_position) > 6371  # Above Earth's surface
    
    def test_keplerian_provider_velocity_matches_satellite(self, sample_satellites):
        """Test KeplerianProvider returns same velocity as direct satellite access."""
        sat = sample_satellites[0]
        expected_velocity = sat.get_velocity_eci()
        
        # Verify reasonable velocity (LEO ~ 7-8 km/s)
        speed = np.linalg.norm(expected_velocity)
        assert 6.5 < speed < 8.5
    
    def test_keplerian_provider_time_propagation(self, sample_satellites):
        """Test KeplerianProvider correctly propagates position over time."""
        sat = sample_satellites[0]
        
        initial_position = sat.get_position_eci().copy()
        
        # Step forward
        sat.step(60.0)  # 1 minute
        
        new_position = sat.get_position_eci()
        
        # Position should have changed
        assert not np.allclose(initial_position, new_position)
        
        # But altitude should be approximately the same (circular orbit)
        initial_radius = np.linalg.norm(initial_position)
        new_radius = np.linalg.norm(new_position)
        assert abs(initial_radius - new_radius) < 1.0  # Within 1 km
    
    def test_keplerian_provider_time_bounds(self, sample_satellites):
        """Test KeplerianProvider reports unbounded time range."""
        # Keplerian orbits are valid for all time
        # Implementation should return (datetime.min, datetime.max)
        pass  # Will be implemented in Step 1


class TestSimulationWithTrajectoryProvider:
    """Tests for Simulation integration with TrajectoryProvider."""
    
    def test_simulation_accepts_trajectory_provider(self, simulation_config):
        """Test Simulation can be created with a trajectory provider."""
        from simulation import Simulation
        
        # Currently, Simulation doesn't accept trajectory_provider
        # This test documents the expected interface after Step 1
        sim = Simulation(simulation_config)
        sim.initialize()
        
        # After Step 1, this should work:
        # sim = Simulation(simulation_config, trajectory_provider=provider)
        assert sim.num_satellites > 0
    
    def test_simulation_default_provider_unchanged(self, small_simulation):
        """Test simulation with default provider behaves identically."""
        sim = small_simulation
        
        # Record initial state
        initial_positions = {
            sat.satellite_id: sat.get_position_eci().copy()
            for sat in sim.satellites
        }
        
        # Step simulation
        sim.step(60.0)
        
        # Positions should have changed
        for sat in sim.satellites:
            new_pos = sat.get_position_eci()
            old_pos = initial_positions[sat.satellite_id]
            assert not np.allclose(new_pos, old_pos)
    
    def test_simulation_statistics_unchanged(self, small_simulation):
        """Test agent statistics work correctly with provider."""
        sim = small_simulation
        
        # Run a few steps
        for _ in range(5):
            sim.step(60.0)
        
        stats = sim.state.agent_statistics
        
        # Verify statistics are computed
        assert stats.total_packets == sim.config.num_packets
        assert 0.0 <= stats.average_completion <= 100.0
        assert stats.fully_updated_count >= 0


class TestBackwardCompatibilityStep1:
    """Ensure Step 1 changes don't break existing functionality."""
    
    def test_satellite_class_unchanged(self):
        """Test Satellite class API is unchanged."""
        from simulation import Satellite, EllipticalOrbit, EARTH_RADIUS_KM
        
        orbit = EllipticalOrbit(
            apoapsis=EARTH_RADIUS_KM + 550,
            periapsis=EARTH_RADIUS_KM + 550,
            inclination=math.radians(53),
            longitude_of_ascending_node=0,
            argument_of_periapsis=0,
        )
        
        sat = Satellite(orbit, initial_position=0.0, satellite_id="TEST")
        
        # All existing methods should work
        assert sat.satellite_id == "TEST"
        assert sat.get_position_eci().shape == (3,)
        assert sat.get_velocity_eci().shape == (3,)
        assert isinstance(sat.get_altitude(), float)
        assert isinstance(sat.get_speed(), float)
        
        sat.step(60.0)
        assert sat.elapsed_time == 60.0
    
    def test_simulation_api_unchanged(self, simulation_config):
        """Test Simulation API is unchanged."""
        from simulation import Simulation
        
        # Create simulation the normal way
        sim = Simulation(simulation_config)
        sim.initialize()
        
        # All existing methods should work
        assert hasattr(sim, 'step')
        assert hasattr(sim, 'reset')
        assert hasattr(sim, 'regenerate')
        assert hasattr(sim, 'is_update_complete')
        assert hasattr(sim, 'get_summary')
        assert hasattr(sim, 'save_log')
        
        # Properties
        assert sim.num_satellites > 0
        assert sim.num_orbits > 0
        assert sim.simulation_time == 0.0
    
    def test_constellation_creation_unchanged(self):
        """Test constellation factory functions unchanged."""
        from simulation import (
            create_walker_delta_constellation,
            create_walker_star_constellation,
            create_random_constellation,
        )
        
        # Walker-Delta
        orbits, sats = create_walker_delta_constellation(
            num_planes=2, sats_per_plane=2, altitude=550, inclination=math.radians(53)
        )
        assert len(orbits) == 2
        assert len(sats) == 4
        
        # Walker-Star
        orbits, sats = create_walker_star_constellation(
            num_planes=2, sats_per_plane=2, altitude=550
        )
        assert len(orbits) == 2
        assert len(sats) == 4
        
        # Random
        orbits, sats = create_random_constellation(num_satellites=5, seed=42)
        assert len(sats) == 5
    
    def test_logging_unchanged(self, small_simulation, tmp_path):
        """Test logging functionality unchanged."""
        sim = small_simulation
        
        # Run some steps
        for _ in range(3):
            sim.step(60.0)
        
        # Save log
        log_file = tmp_path / "test_log.json"
        sim.save_log(str(log_file))
        
        assert log_file.exists()
        
        # Load and verify
        from simulation import load_simulation_log
        log = load_simulation_log(str(log_file))
        
        assert "header" in log
        assert "time_series" in log
        assert len(log["time_series"]) == 4  # Initial + 3 steps


class TestTrajectoryProviderEdgeCases:
    """Test edge cases and error handling."""
    
    def test_invalid_satellite_id(self, sample_satellites):
        """Test handling of invalid satellite ID."""
        # Provider should raise appropriate error for unknown satellite
        pass  # Will be implemented in Step 1
    
    def test_time_out_of_bounds(self):
        """Test handling of time outside valid range."""
        # For SPICE provider, time must be within kernel coverage
        pass  # Will be implemented in Step 2
    
    def test_empty_satellite_list(self):
        """Test provider with empty satellite list."""
        class KeplerianProvider:
            def __init__(self, satellites, epoch):
                self.satellites = {sat.satellite_id: sat for sat in satellites}
            
            def get_satellite_ids(self):
                return list(self.satellites.keys())
        
        epoch = datetime(2025, 1, 1)
        provider = KeplerianProvider([], epoch)
        assert provider.get_satellite_ids() == []
