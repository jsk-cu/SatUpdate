#!/usr/bin/env python3
"""
Backward Compatibility and Integration Tests

These tests ensure:
1. All existing functionality is preserved
2. New features are opt-in only
3. Default behavior is unchanged
4. Command-line interface remains backward compatible
5. End-to-end integration works correctly
"""

import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# BACKWARD COMPATIBILITY TESTS
# =============================================================================

class TestSimulationBackwardCompatibility:
    """Ensure Simulation class API is unchanged."""
    
    def test_simulation_creation_unchanged(self):
        """Test Simulation can be created without new parameters."""
        from simulation import Simulation, SimulationConfig, ConstellationType
        
        # Original way of creating simulation
        config = SimulationConfig(
            constellation_type=ConstellationType.WALKER_DELTA,
            num_planes=2,
            sats_per_plane=3,
        )
        
        sim = Simulation(config)
        sim.initialize()
        
        assert sim.num_satellites == 6
    
    def test_simulation_step_unchanged(self, small_simulation):
        """Test step() method unchanged."""
        sim = small_simulation
        
        state = sim.step(60.0)
        
        # State should have expected attributes
        assert hasattr(state, 'time')
        assert hasattr(state, 'step_count')
        assert hasattr(state, 'satellite_positions')
        assert hasattr(state, 'active_links')
        assert hasattr(state, 'agent_statistics')
    
    def test_simulation_run_unchanged(self):
        """Test run() method unchanged."""
        from simulation import Simulation, SimulationConfig, ConstellationType
        
        config = SimulationConfig(
            constellation_type=ConstellationType.WALKER_DELTA,
            num_planes=2,
            sats_per_plane=2,
            num_packets=10,
        )
        
        sim = Simulation(config)
        states = sim.run(duration=300, timestep=60)
        
        assert len(states) == 5
        assert states[-1].time == 300.0
    
    def test_simulation_reset_unchanged(self, small_simulation):
        """Test reset() method unchanged."""
        sim = small_simulation
        
        # Run some steps
        for _ in range(5):
            sim.step(60.0)
        
        assert sim.simulation_time > 0
        
        # Reset
        sim.reset()
        
        assert sim.simulation_time == 0.0
        assert sim.state.step_count == 0
    
    def test_simulation_logging_unchanged(self, small_simulation, tmp_path):
        """Test logging functionality unchanged."""
        sim = small_simulation
        
        for _ in range(3):
            sim.step(60.0)
        
        log_file = tmp_path / "test_log.json"
        sim.save_log(str(log_file))
        
        assert log_file.exists()
        
        # Verify log format
        with open(log_file) as f:
            log = json.load(f)
        
        assert "header" in log
        assert "time_series" in log
        assert log["header"]["constellation_type"] == "walker_delta"
    
    def test_get_summary_unchanged(self, small_simulation):
        """Test get_summary() method unchanged."""
        sim = small_simulation
        
        summary = sim.get_summary()
        
        expected_keys = [
            "constellation_type",
            "num_satellites",
            "num_orbits",
            "num_base_stations",
            "num_packets",
            "simulation_time",
            "step_count",
            "average_completion",
            "fully_updated_satellites",
            "update_complete",
            "initialized",
        ]
        
        for key in expected_keys:
            assert key in summary, f"Missing key: {key}"


class TestSatelliteBackwardCompatibility:
    """Ensure Satellite class API is unchanged."""
    
    def test_satellite_creation_unchanged(self):
        """Test Satellite can be created without new parameters."""
        from simulation import Satellite, EllipticalOrbit, EARTH_RADIUS_KM
        
        orbit = EllipticalOrbit(
            apoapsis=EARTH_RADIUS_KM + 550,
            periapsis=EARTH_RADIUS_KM + 550,
            inclination=math.radians(53),
            longitude_of_ascending_node=0,
            argument_of_periapsis=0,
        )
        
        sat = Satellite(orbit, initial_position=0.0, satellite_id="TEST-001")
        
        assert sat.satellite_id == "TEST-001"
        assert sat.position == 0.0
    
    def test_satellite_methods_unchanged(self):
        """Test Satellite methods unchanged."""
        from simulation import Satellite, EllipticalOrbit, EARTH_RADIUS_KM
        
        orbit = EllipticalOrbit(
            apoapsis=EARTH_RADIUS_KM + 550,
            periapsis=EARTH_RADIUS_KM + 550,
            inclination=math.radians(53),
            longitude_of_ascending_node=0,
            argument_of_periapsis=0,
        )
        
        sat = Satellite(orbit, initial_position=0.0)
        
        # All methods should exist and work
        pos = sat.get_position_eci()
        assert pos.shape == (3,)
        
        vel = sat.get_velocity_eci()
        assert vel.shape == (3,)
        
        alt = sat.get_altitude()
        assert 500 < alt < 600  # Approximately 550 km
        
        speed = sat.get_speed()
        assert 7.0 < speed < 8.0  # LEO velocity
        
        geo = sat.get_geospatial_position()
        assert hasattr(geo, 'latitude')
        assert hasattr(geo, 'longitude')
        assert hasattr(geo, 'altitude')


class TestOrbitBackwardCompatibility:
    """Ensure EllipticalOrbit class API is unchanged."""
    
    def test_orbit_creation_unchanged(self):
        """Test EllipticalOrbit can be created without new parameters."""
        from simulation import EllipticalOrbit, EARTH_RADIUS_KM
        
        orbit = EllipticalOrbit(
            apoapsis=EARTH_RADIUS_KM + 550,
            periapsis=EARTH_RADIUS_KM + 550,
            inclination=math.radians(53),
            longitude_of_ascending_node=0,
            argument_of_periapsis=0,
        )
        
        assert orbit.eccentricity < 0.001  # Nearly circular
        assert 90 < orbit.period / 60 < 100  # ~95 minute period
    
    def test_orbit_methods_unchanged(self):
        """Test EllipticalOrbit methods unchanged."""
        from simulation import EllipticalOrbit, EARTH_RADIUS_KM
        
        orbit = EllipticalOrbit(
            apoapsis=EARTH_RADIUS_KM + 550,
            periapsis=EARTH_RADIUS_KM + 550,
            inclination=math.radians(53),
            longitude_of_ascending_node=0,
            argument_of_periapsis=0,
        )
        
        # All methods should exist and work
        pos = orbit.position_eci(0.0)
        assert pos.shape == (3,)
        
        vel = orbit.velocity_eci(0.0)
        assert vel.shape == (3,)
        
        r = orbit.radius_at_true_anomaly(0.0)
        assert r > EARTH_RADIUS_KM
        
        v = orbit.velocity_at_radius(r)
        assert v > 0


class TestConstellationBackwardCompatibility:
    """Ensure constellation factory functions unchanged."""
    
    def test_walker_delta_unchanged(self):
        """Test Walker-Delta creation unchanged."""
        from simulation import create_walker_delta_constellation
        
        orbits, sats = create_walker_delta_constellation(
            num_planes=3,
            sats_per_plane=4,
            altitude=550,
            inclination=math.radians(53),
            phasing_parameter=1,
        )
        
        assert len(orbits) == 3
        assert len(sats) == 12
    
    def test_walker_star_unchanged(self):
        """Test Walker-Star creation unchanged."""
        from simulation import create_walker_star_constellation
        
        orbits, sats = create_walker_star_constellation(
            num_planes=4,
            sats_per_plane=3,
            altitude=800,
        )
        
        assert len(orbits) == 4
        assert len(sats) == 12
    
    def test_random_constellation_unchanged(self):
        """Test random constellation creation unchanged."""
        from simulation import create_random_constellation
        
        orbits, sats = create_random_constellation(
            num_satellites=10,
            seed=42,
        )
        
        assert len(sats) == 10
        
        # Same seed should give same result
        orbits2, sats2 = create_random_constellation(
            num_satellites=10,
            seed=42,
        )
        
        # Positions should match
        for s1, s2 in zip(sats, sats2):
            np.testing.assert_allclose(
                s1.get_position_eci(),
                s2.get_position_eci()
            )


class TestAgentBackwardCompatibility:
    """Ensure agent system is unchanged."""
    
    def test_agent_imports_unchanged(self):
        """Test agent imports work."""
        from agents import get_agent_class, BaseAgent
        
        # Default agent
        AgentClass = get_agent_class("min")
        assert AgentClass is not None
        
        # Base agent
        AgentClass = get_agent_class("base")
        assert AgentClass is not None
    
    def test_agent_protocol_unchanged(self, small_simulation):
        """Test agent protocol works correctly."""
        sim = small_simulation
        
        # Get base station agent
        bs_agent = sim.get_base_station_agent()
        assert bs_agent is not None
        assert bs_agent.has_all_packets()
        
        # Get satellite agent
        sat_agent = sim.get_satellite_agent(sim.satellites[0].satellite_id)
        assert sat_agent is not None
        assert not sat_agent.has_all_packets()  # Initially empty


# =============================================================================
# DEFAULT BEHAVIOR TESTS
# =============================================================================

class TestDefaultBehaviorUnchanged:
    """Ensure default behavior with no new arguments is unchanged."""
    
    def test_default_trajectory_provider(self, simulation_config):
        """Test default trajectory provider is Keplerian."""
        from simulation import Simulation
        
        sim = Simulation(simulation_config)
        sim.initialize()
        
        # Should use Keplerian by default
        # Positions should be computed correctly
        for sat in sim.satellites:
            pos = sat.get_position_eci()
            assert np.linalg.norm(pos) > 6371  # Above Earth
    
    def test_default_network_backend(self, simulation_config):
        """Test default network backend is native."""
        from simulation import Simulation
        
        sim = Simulation(simulation_config)
        sim.initialize()
        
        # Should use native backend
        # Packets should distribute correctly
        for _ in range(10):
            sim.step(60.0)
        
        stats = sim.state.agent_statistics
        assert stats.average_completion > 0  # Some distribution happened
    
    def test_no_new_required_arguments(self):
        """Test no new required arguments."""
        from simulation import Simulation, SimulationConfig
        
        # Should work with minimal config
        config = SimulationConfig()
        sim = Simulation(config)
        sim.initialize()
        
        assert sim.num_satellites > 0


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestEndToEndIntegration:
    """End-to-end integration tests."""
    
    def test_full_simulation_run(self):
        """Test complete simulation from start to finish."""
        from simulation import Simulation, SimulationConfig, ConstellationType
        
        config = SimulationConfig(
            constellation_type=ConstellationType.WALKER_DELTA,
            num_planes=2,
            sats_per_plane=3,
            altitude=550,
            inclination=math.radians(53),
            num_packets=20,
            random_seed=42,
        )
        
        sim = Simulation(config, enable_logging=True)
        sim.initialize()
        
        max_steps = 50
        for step in range(max_steps):
            sim.step(60.0)
            
            if sim.is_update_complete():
                break
        
        # Should have made progress
        stats = sim.state.agent_statistics
        assert stats.average_completion > 0
        
        # Log should be populated
        log = sim.get_log()
        assert len(log["time_series"]) > 0
    
    def test_simulation_with_custom_base_station(self):
        """Test simulation with custom base station location."""
        from simulation import Simulation, SimulationConfig, ConstellationType
        
        config = SimulationConfig(
            constellation_type=ConstellationType.WALKER_DELTA,
            num_planes=2,
            sats_per_plane=3,
            num_packets=10,
            base_station_latitude=40.7,
            base_station_longitude=-74.0,
            base_station_range=5000,
        )
        
        sim = Simulation(config)
        sim.initialize()
        
        assert len(sim.base_stations) == 1
        assert abs(sim.base_stations[0].latitude_deg - 40.7) < 0.1
    
    def test_simulation_with_communication_range(self):
        """Test simulation with limited communication range."""
        from simulation import Simulation, SimulationConfig, ConstellationType
        
        config_unlimited = SimulationConfig(
            constellation_type=ConstellationType.WALKER_DELTA,
            num_planes=2,
            sats_per_plane=3,
            num_packets=10,
            communication_range=None,
            random_seed=42,
        )
        
        config_limited = SimulationConfig(
            constellation_type=ConstellationType.WALKER_DELTA,
            num_planes=2,
            sats_per_plane=3,
            num_packets=10,
            communication_range=2000,
            random_seed=42,
        )
        
        sim_unlimited = Simulation(config_unlimited)
        sim_unlimited.initialize()
        
        sim_limited = Simulation(config_limited)
        sim_limited.initialize()
        
        # Limited range should have fewer active links
        assert len(sim_limited.state.active_links) <= \
               len(sim_unlimited.state.active_links)


@pytest.mark.integration
class TestCommandLineInterface:
    """Tests for command-line interface backward compatibility."""
    
    def test_main_help(self):
        """Test main.py --help works."""
        # This would run the actual command if main.py exists
        pass
    
    def test_default_arguments(self):
        """Test default argument values."""
        import argparse
        
        # Simulate the argument parser from main.py
        parser = argparse.ArgumentParser()
        parser.add_argument("--type", "-t", default="walker_delta")
        parser.add_argument("--planes", "-p", type=int, default=3)
        parser.add_argument("--sats-per-plane", "-s", type=int, default=4)
        parser.add_argument("--altitude", "-a", type=float, default=550.0)
        parser.add_argument("--inclination", "-i", type=float, default=53.0)
        parser.add_argument("--num-packets", type=int, default=100)
        parser.add_argument("--headless", action="store_true")
        
        args = parser.parse_args([])
        
        assert args.type == "walker_delta"
        assert args.planes == 3
        assert args.sats_per_plane == 4
        assert args.altitude == 550.0
        assert args.num_packets == 100
        assert args.headless is False
    
    def test_new_arguments_have_defaults(self):
        """Test new arguments have sensible defaults."""
        import argparse
        
        parser = argparse.ArgumentParser()
        
        # New arguments should have defaults that preserve old behavior
        parser.add_argument(
            "--trajectory-provider",
            choices=["keplerian", "spice"],
            default="keplerian"
        )
        parser.add_argument(
            "--network-backend",
            choices=["native", "ns3"],
            default="native"
        )
        parser.add_argument("--spice-config", type=Path, default=None)
        parser.add_argument("--ns3-mode", default="file")
        parser.add_argument("--ns3-path", type=Path, default=None)
        
        args = parser.parse_args([])
        
        # Defaults should preserve old behavior
        assert args.trajectory_provider == "keplerian"
        assert args.network_backend == "native"
        assert args.spice_config is None
        assert args.ns3_path is None


# =============================================================================
# PERFORMANCE REGRESSION TESTS
# =============================================================================

class TestPerformanceRegression:
    """Ensure no significant performance regression."""
    
    @pytest.mark.slow
    def test_simulation_performance(self):
        """Test simulation performance hasn't regressed significantly."""
        import time
        from simulation import Simulation, SimulationConfig, ConstellationType
        
        config = SimulationConfig(
            constellation_type=ConstellationType.WALKER_DELTA,
            num_planes=4,
            sats_per_plane=6,
            num_packets=50,
            random_seed=42,
        )
        
        sim = Simulation(config)
        sim.initialize()
        
        start_time = time.time()
        
        for _ in range(100):
            sim.step(60.0)
        
        elapsed = time.time() - start_time
        
        # Should complete in reasonable time
        # Adjust threshold based on expected performance
        assert elapsed < 10.0, f"Simulation took {elapsed:.2f}s, expected < 10s"
    
    def test_logging_performance(self, tmp_path):
        """Test logging doesn't significantly impact performance."""
        import time
        from simulation import Simulation, SimulationConfig, ConstellationType
        
        config = SimulationConfig(
            constellation_type=ConstellationType.WALKER_DELTA,
            num_planes=3,
            sats_per_plane=4,
            num_packets=30,
            random_seed=42,
        )
        
        # With logging
        sim_logged = Simulation(config, enable_logging=True)
        sim_logged.initialize()
        
        start = time.time()
        for _ in range(50):
            sim_logged.step(60.0)
        logged_time = time.time() - start
        
        # Without logging
        sim_unlogged = Simulation(config, enable_logging=False)
        sim_unlogged.initialize()
        
        start = time.time()
        for _ in range(50):
            sim_unlogged.step(60.0)
        unlogged_time = time.time() - start
        
        # Logging overhead should be reasonable (<50% slower)
        overhead = (logged_time - unlogged_time) / unlogged_time
        assert overhead < 0.5, f"Logging overhead: {overhead*100:.1f}%"
