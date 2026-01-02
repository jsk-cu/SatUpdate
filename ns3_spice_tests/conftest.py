#!/usr/bin/env python3
"""
Pytest Configuration and Shared Fixtures

Provides common fixtures, markers, and utilities for testing the
NS-3 and SPICE integration components.
"""

import json
import math
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

import pytest
import numpy as np

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# PYTEST MARKERS
# =============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "requires_spice: mark test as requiring SpiceyPy installation"
    )
    config.addinivalue_line(
        "markers", "requires_ns3: mark test as requiring NS-3 installation"
    )
    config.addinivalue_line(
        "markers", "requires_ns3_bindings: mark test as requiring NS-3 Python bindings"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: mark as integration test"
    )


def pytest_collection_modifyitems(config, items):
    """Skip tests based on available dependencies."""
    # Check for SpiceyPy
    try:
        import spiceypy
        spice_available = True
    except ImportError:
        spice_available = False
    
    # Check for NS-3 (basic check)
    ns3_available = Path("/usr/local/ns3/ns3").exists() or \
                    Path("/opt/ns3/ns3").exists()
    
    # Check for NS-3 Python bindings
    try:
        import ns.core
        ns3_bindings_available = True
    except ImportError:
        ns3_bindings_available = False
    
    skip_spice = pytest.mark.skip(reason="SpiceyPy not installed")
    skip_ns3 = pytest.mark.skip(reason="NS-3 not installed")
    skip_ns3_bindings = pytest.mark.skip(reason="NS-3 Python bindings not available")
    
    for item in items:
        if "requires_spice" in item.keywords and not spice_available:
            item.add_marker(skip_spice)
        if "requires_ns3" in item.keywords and not ns3_available:
            item.add_marker(skip_ns3)
        if "requires_ns3_bindings" in item.keywords and not ns3_bindings_available:
            item.add_marker(skip_ns3_bindings)


# =============================================================================
# SIMULATION FIXTURES
# =============================================================================

@pytest.fixture
def simulation_config():
    """Default simulation configuration for testing."""
    return {
        "num_orbits": 3,
        "satellites_per_orbit": 4,
        "altitude_km": 550,
        "inclination_deg": 53.0,
        "epoch": "2025-01-01T00:00:00Z",
    }


@pytest.fixture
def sample_satellites():
    """Sample satellite orbital elements."""
    return [
        {
            "id": "SAT-001",
            "semi_major_axis": 6928.0,  # km
            "eccentricity": 0.001,
            "inclination": 53.0,  # degrees
            "raan": 0.0,
            "arg_perigee": 0.0,
            "true_anomaly": 0.0,
        },
        {
            "id": "SAT-002",
            "semi_major_axis": 6928.0,
            "eccentricity": 0.001,
            "inclination": 53.0,
            "raan": 120.0,
            "arg_perigee": 0.0,
            "true_anomaly": 30.0,
        },
        {
            "id": "SAT-003",
            "semi_major_axis": 6928.0,
            "eccentricity": 0.001,
            "inclination": 53.0,
            "raan": 240.0,
            "arg_perigee": 0.0,
            "true_anomaly": 60.0,
        },
    ]


# =============================================================================
# NS-3 MOCK FIXTURES
# =============================================================================

@pytest.fixture
def mock_ns3_subprocess():
    """Mock subprocess for NS-3 file mode testing."""
    
    def create_mock_result(transfers=None):
        if transfers is None:
            transfers = [
                {
                    "source": "SAT-001",
                    "destination": "SAT-002",
                    "packet_id": 0,
                    "timestamp": 0.023,
                    "success": True,
                    "latency_ms": 23.4
                }
            ]
        
        output = {
            "status": "success",
            "simulation_time": 60.0,
            "transfers": transfers,
            "statistics": {
                "total_packets_sent": len(transfers),
                "total_packets_received": sum(1 for t in transfers if t["success"]),
                "average_latency_ms": 23.4
            }
        }
        
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(output)
        mock_result.stderr = ""
        
        return mock_result
    
    return create_mock_result


@pytest.fixture
def mock_ns3_socket():
    """Mock socket for NS-3 socket mode testing."""
    import socket as socket_module
    import time
    
    class MockSocket:
        def __init__(self):
            self.sent_data = []
            self.response_queue = []
            self.connected = False
            self._timeout = None
            self._blocking = True
        
        def connect(self, address):
            self.connected = True
        
        def settimeout(self, timeout):
            self._timeout = timeout
            self._blocking = (timeout is None or timeout > 0)
        
        def sendall(self, data):
            self.sent_data.append(data)
            # Auto-generate response
            response = {
                "status": "success",
                "transfers": [
                    {
                        "source": "SAT-001",
                        "destination": "SAT-002",
                        "packet_id": 0,
                        "timestamp": 0.023,
                        "success": True,
                        "latency_ms": 23.4
                    }
                ],
                "statistics": {
                    "total_packets_sent": 1,
                    "total_packets_received": 1,
                    "average_latency_ms": 23.4
                }
            }
            self.response_queue.append(json.dumps(response) + "\n")
        
        def recv(self, bufsize):
            if self.response_queue:
                return self.response_queue.pop(0).encode()
            # Simulate timeout instead of returning empty (which signals close)
            if self._timeout is not None and self._timeout < 1:
                raise socket_module.timeout("timed out")
            # For longer timeouts, sleep briefly and raise timeout
            time.sleep(0.05)
            raise socket_module.timeout("timed out")
        
        def close(self):
            self.connected = False
    
    return MockSocket()


@pytest.fixture
def mock_ns3_bindings():
    """Mock NS-3 Python bindings for testing."""
    mock_core = MagicMock()
    mock_network = MagicMock()
    mock_internet = MagicMock()
    mock_p2p = MagicMock()
    mock_mobility = MagicMock()
    mock_apps = MagicMock()
    
    mock_core.Simulator = MagicMock()
    mock_core.Simulator.Stop = MagicMock()
    mock_core.Simulator.Run = MagicMock()
    mock_core.Simulator.Destroy = MagicMock()
    mock_core.Seconds = lambda x: x
    mock_core.Vector = lambda x, y, z: (x, y, z)
    
    mock_network.NodeContainer = MagicMock
    
    mock_internet.InternetStackHelper = MagicMock
    mock_p2p.PointToPointHelper = MagicMock
    mock_mobility.MobilityHelper = MagicMock
    mock_mobility.ListPositionAllocator = MagicMock
    
    return {
        "core": mock_core,
        "network": mock_network,
        "internet": mock_internet,
        "point_to_point": mock_p2p,
        "mobility": mock_mobility,
        "applications": mock_apps
    }


# =============================================================================
# NETWORK BACKEND FIXTURES
# =============================================================================

@pytest.fixture
def sample_topology():
    """Sample network topology for testing."""
    return {
        "nodes": [
            {"id": "SAT-001", "type": "satellite", "position": [7000000, 0, 0]},
            {"id": "SAT-002", "type": "satellite", "position": [0, 7000000, 0]},
            {"id": "SAT-003", "type": "satellite", "position": [-7000000, 0, 0]},
            {"id": "BASE-1", "type": "ground", "position": [6371000, 0, 0]},
        ],
        "links": [
            ("SAT-001", "SAT-002"),
            ("SAT-002", "SAT-003"),
            ("BASE-1", "SAT-001"),
        ],
        "config": {
            "data_rate": "10Mbps",
            "propagation_model": "constant_speed"
        }
    }


@pytest.fixture
def active_links_set():
    """Sample set of active links."""
    return {
        ("SAT-001", "SAT-002"),
        ("SAT-002", "SAT-003"),
    }


# =============================================================================
# FILE SYSTEM FIXTURES
# =============================================================================

@pytest.fixture
def temp_output_dir(tmp_path):
    """Temporary directory for output files."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def temp_work_dir(tmp_path):
    """Temporary working directory."""
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    return work_dir


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def assert_positions_close(pos1: np.ndarray, pos2: np.ndarray, rtol: float = 1e-5):
    """Assert two position vectors are close."""
    np.testing.assert_allclose(pos1, pos2, rtol=rtol)


def run_simulation_steps(sim, num_steps: int, timestep: float = 60.0):
    """Helper to run simulation for several steps."""
    states = []
    for _ in range(num_steps):
        state = sim.step(timestep)
        states.append(state)
    return states