#!/usr/bin/env python3
"""
Tests for Step 7: NS-3 Backend - Python Bindings Mode

These tests verify:
1. NS-3 Python bindings detection
2. SNS3 satellite extensions detection
3. NS3BindingsWrapper functionality
4. NS3Backend bindings mode integration
5. Fallback behavior when bindings unavailable
6. Trace callbacks and packet handling
7. Position updates and mobility model
8. Comparison with file/socket modes
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation import (
    NS3Backend,
    NS3Config,
    NS3Mode,
    NS3Node,
    NS3BindingsWrapper,
    NS3BindingsError,
    PacketTransfer,
    check_ns3_bindings,
    check_sns3_bindings,
    create_ns3_backend,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_topology():
    """Sample network topology for testing."""
    return {
        "nodes": [
            {"id": "SAT-001", "type": "satellite", "position": [7000000, 0, 0]},
            {"id": "SAT-002", "type": "satellite", "position": [0, 7000000, 0]},
            {"id": "SAT-003", "type": "satellite", "position": [0, 0, 7000000]},
        ],
        "links": [
            ("SAT-001", "SAT-002"),
            ("SAT-002", "SAT-003"),
        ],
        "config": {
            "data_rate": "10Mbps",
            "fixed_delay_ms": 50.0,
        }
    }


@pytest.fixture
def mock_ns3_modules():
    """Mock NS-3 Python modules."""
    mock_core = MagicMock()
    mock_network = MagicMock()
    mock_internet = MagicMock()
    mock_p2p = MagicMock()
    mock_applications = MagicMock()
    mock_mobility = MagicMock()
    
    # Configure core module
    mock_core.Simulator = MagicMock()
    mock_core.Simulator.Stop = MagicMock()
    mock_core.Simulator.Run = MagicMock()
    mock_core.Simulator.Destroy = MagicMock()
    mock_core.Seconds = MagicMock(return_value=MagicMock())
    mock_core.StringValue = MagicMock(return_value=MagicMock())
    mock_core.Vector = MagicMock(return_value=MagicMock())
    
    # Configure network module
    mock_node_container = MagicMock()
    mock_node_container.Create = MagicMock()
    mock_node_container.Get = MagicMock(return_value=MagicMock())
    mock_network.NodeContainer = MagicMock(return_value=mock_node_container)
    mock_network.Ipv4Address = MagicMock(return_value=MagicMock())
    mock_network.Ipv4Mask = MagicMock(return_value=MagicMock())
    
    # Configure internet module
    mock_internet.InternetStackHelper = MagicMock(return_value=MagicMock())
    mock_internet.Ipv4AddressHelper = MagicMock(return_value=MagicMock())
    
    # Configure point-to-point module
    mock_p2p.PointToPointHelper = MagicMock(return_value=MagicMock())
    
    # Configure mobility module
    mock_mobility.MobilityHelper = MagicMock(return_value=MagicMock())
    mock_mobility.ListPositionAllocator = MagicMock(return_value=MagicMock())
    mock_mobility.MobilityModel = MagicMock()
    mock_mobility.MobilityModel.GetTypeId = MagicMock(return_value=MagicMock())
    
    return {
        "core": mock_core,
        "network": mock_network,
        "internet": mock_internet,
        "point_to_point": mock_p2p,
        "applications": mock_applications,
        "mobility": mock_mobility,
    }


@pytest.fixture
def mock_ns3_bindings(mock_ns3_modules):
    """Patch NS-3 imports to return mocks."""
    with patch.dict('sys.modules', {
        'ns': MagicMock(),
        'ns.core': mock_ns3_modules["core"],
        'ns.network': mock_ns3_modules["network"],
        'ns.internet': mock_ns3_modules["internet"],
        'ns.point_to_point': mock_ns3_modules["point_to_point"],
        'ns.applications': mock_ns3_modules["applications"],
        'ns.mobility': mock_ns3_modules["mobility"],
    }):
        yield mock_ns3_modules


# =============================================================================
# Test: Bindings Detection Functions
# =============================================================================

class TestBindingsDetection:
    """Tests for NS-3 bindings detection functions."""
    
    def test_check_ns3_bindings_unavailable(self):
        """Test that check_ns3_bindings returns False when bindings not installed."""
        # Without mocking, ns.core import should fail
        result = check_ns3_bindings()
        assert isinstance(result, bool)
        # In test environment, bindings are typically not available
        assert result is False
    
    def test_check_sns3_bindings_unavailable(self):
        """Test that check_sns3_bindings returns False when SNS3 not installed."""
        result = check_sns3_bindings()
        assert isinstance(result, bool)
        assert result is False
    
    def test_check_ns3_bindings_available_with_mock(self, mock_ns3_bindings):
        """Test check_ns3_bindings returns True with mocked bindings."""
        # The function checks at module level, so we need to test differently
        # This verifies the mock setup works
        import ns.core
        assert ns.core is not None
    
    def test_bindings_check_returns_boolean(self):
        """Test that detection functions always return boolean."""
        ns3_result = check_ns3_bindings()
        sns3_result = check_sns3_bindings()
        
        assert type(ns3_result) is bool
        assert type(sns3_result) is bool


# =============================================================================
# Test: NS3BindingsError
# =============================================================================

class TestNS3BindingsError:
    """Tests for NS3BindingsError exception."""
    
    def test_error_creation(self):
        """Test NS3BindingsError can be created."""
        error = NS3BindingsError("Test error message")
        assert str(error) == "Test error message"
    
    def test_error_inheritance(self):
        """Test NS3BindingsError inherits from Exception."""
        error = NS3BindingsError("test")
        assert isinstance(error, Exception)
    
    def test_error_raised(self):
        """Test NS3BindingsError can be raised and caught."""
        with pytest.raises(NS3BindingsError) as exc_info:
            raise NS3BindingsError("Bindings not found")
        
        assert "Bindings not found" in str(exc_info.value)


# =============================================================================
# Test: NS3BindingsWrapper Creation
# =============================================================================

class TestNS3BindingsWrapperCreation:
    """Tests for NS3BindingsWrapper creation."""
    
    def test_wrapper_creation_fails_without_bindings(self):
        """Test wrapper creation fails when bindings not available."""
        config = NS3Config()
        
        with pytest.raises(NS3BindingsError) as exc_info:
            NS3BindingsWrapper(config)
        
        assert "Python bindings not available" in str(exc_info.value)
    
    def test_error_message_includes_instructions(self):
        """Test error message includes build instructions."""
        config = NS3Config()
        
        with pytest.raises(NS3BindingsError) as exc_info:
            NS3BindingsWrapper(config)
        
        error_msg = str(exc_info.value)
        assert "enable-python-bindings" in error_msg or "build" in error_msg.lower()


# =============================================================================
# Test: NS3BindingsWrapper with Mocks
# =============================================================================

class TestNS3BindingsWrapperWithMocks:
    """Tests for NS3BindingsWrapper using mocked NS-3 modules."""
    
    def test_wrapper_creation_with_mock(self, mock_ns3_bindings):
        """Test wrapper creation succeeds with mocked bindings."""
        config = NS3Config()
        
        # Patch check_ns3_bindings to return True
        with patch('simulation.ns3_backend.check_ns3_bindings', return_value=True):
            wrapper = NS3BindingsWrapper(config)
            assert wrapper is not None
            assert wrapper.initialized is False
    
    def test_wrapper_imports_modules(self, mock_ns3_bindings):
        """Test wrapper imports NS-3 modules correctly."""
        config = NS3Config()
        
        with patch('simulation.ns3_backend.check_ns3_bindings', return_value=True):
            wrapper = NS3BindingsWrapper(config)
            assert wrapper._ns is not None
            assert "core" in wrapper._ns
            assert "network" in wrapper._ns
    
    def test_wrapper_sns3_detection(self, mock_ns3_bindings):
        """Test wrapper detects SNS3 availability."""
        config = NS3Config()
        
        with patch('simulation.ns3_backend.check_ns3_bindings', return_value=True):
            with patch('simulation.ns3_backend.check_sns3_bindings', return_value=True):
                wrapper = NS3BindingsWrapper(config, use_sns3=True)
                # SNS3 check happens at import time
                assert isinstance(wrapper.has_sns3, bool)
    
    def test_wrapper_initialization(self, mock_ns3_bindings, sample_topology):
        """Test wrapper initialization with nodes and links."""
        config = NS3Config()
        
        with patch('simulation.ns3_backend.check_ns3_bindings', return_value=True):
            wrapper = NS3BindingsWrapper(config)
            
            # Create NS3Node objects
            nodes = {}
            for node_data in sample_topology["nodes"]:
                nodes[node_data["id"]] = NS3Node(
                    id=node_data["id"],
                    node_type=node_data["type"],
                    position=np.array(node_data["position"])
                )
            
            links = set(tuple(link) for link in sample_topology["links"])
            
            wrapper.initialize(nodes, links)
            
            assert wrapper.initialized is True
            assert len(wrapper._node_map) == 3
    
    def test_wrapper_send_packet(self, mock_ns3_bindings, sample_topology):
        """Test wrapper packet sending."""
        config = NS3Config()
        
        with patch('simulation.ns3_backend.check_ns3_bindings', return_value=True):
            wrapper = NS3BindingsWrapper(config)
            
            nodes = {}
            for node_data in sample_topology["nodes"]:
                nodes[node_data["id"]] = NS3Node(
                    id=node_data["id"],
                    node_type=node_data["type"],
                    position=np.array(node_data["position"])
                )
            
            links = set(tuple(link) for link in sample_topology["links"])
            wrapper.initialize(nodes, links)
            
            result = wrapper.send_packet("SAT-001", "SAT-002", packet_id=1, size_bytes=1024)
            assert result is True
            assert 1 in wrapper._pending_packets
    
    def test_wrapper_step(self, mock_ns3_bindings, sample_topology):
        """Test wrapper simulation step."""
        config = NS3Config()
        
        with patch('simulation.ns3_backend.check_ns3_bindings', return_value=True):
            wrapper = NS3BindingsWrapper(config)
            
            nodes = {}
            for node_data in sample_topology["nodes"]:
                nodes[node_data["id"]] = NS3Node(
                    id=node_data["id"],
                    node_type=node_data["type"],
                    position=np.array(node_data["position"])
                )
            
            links = set(tuple(link) for link in sample_topology["links"])
            wrapper.initialize(nodes, links)
            
            wrapper.send_packet("SAT-001", "SAT-002", packet_id=1, size_bytes=1024)
            transfers = wrapper.step(60.0)
            
            assert isinstance(transfers, list)
            assert len(transfers) == 1
            assert transfers[0]["packet_id"] == 1
            assert transfers[0]["success"] is True
    
    def test_wrapper_shutdown(self, mock_ns3_bindings):
        """Test wrapper shutdown."""
        config = NS3Config()
        
        with patch('simulation.ns3_backend.check_ns3_bindings', return_value=True):
            wrapper = NS3BindingsWrapper(config)
            
            # Create minimal initialization
            nodes = {"A": NS3Node(id="A", node_type="satellite", position=np.array([0, 0, 0]))}
            wrapper.initialize(nodes, set())
            
            wrapper.shutdown()
            
            assert wrapper.initialized is False
            assert len(wrapper._node_map) == 0


# =============================================================================
# Test: NS3Backend Bindings Mode
# =============================================================================

class TestNS3BackendBindingsMode:
    """Tests for NS3Backend in bindings mode."""
    
    def test_backend_bindings_mode_creation(self):
        """Test backend creation with bindings mode."""
        backend = NS3Backend(mode="bindings")
        
        # Should fall back to mock since bindings not available
        assert backend.mode == NS3Mode.MOCK
    
    def test_backend_bindings_available_property(self):
        """Test bindings_available property."""
        backend = NS3Backend(mode="mock")
        assert isinstance(backend.bindings_available, bool)
    
    def test_backend_bindings_wrapper_property(self):
        """Test bindings_wrapper property."""
        backend = NS3Backend(mode="mock")
        # In mock mode, wrapper should be None
        assert backend.bindings_wrapper is None
    
    def test_backend_fallback_to_mock(self, sample_topology):
        """Test backend falls back to mock when bindings unavailable."""
        backend = NS3Backend(mode="bindings")
        backend.initialize(sample_topology)
        
        # Should work in mock mode
        backend.send_packet("SAT-001", "SAT-002", packet_id=1)
        transfers = backend.step(60.0)
        
        assert isinstance(transfers, list)
        backend.shutdown()


class TestNS3BackendBindingsModeWithMocks:
    """Tests for NS3Backend bindings mode using mocked NS-3."""
    
    def test_backend_initialization_with_bindings(self, mock_ns3_bindings, sample_topology):
        """Test backend initialization in bindings mode."""
        with patch('simulation.ns3_backend.check_ns3_bindings', return_value=True):
            backend = NS3Backend(mode="bindings")
            assert backend.mode == NS3Mode.BINDINGS
            
            backend.initialize(sample_topology)
            assert backend._bindings_wrapper is not None
            
            backend.shutdown()
    
    def test_backend_step_bindings_mode(self, mock_ns3_bindings, sample_topology):
        """Test backend step in bindings mode."""
        with patch('simulation.ns3_backend.check_ns3_bindings', return_value=True):
            backend = NS3Backend(mode="bindings")
            backend.initialize(sample_topology)
            
            backend.send_packet("SAT-001", "SAT-002", packet_id=1)
            transfers = backend.step(60.0)
            
            assert isinstance(transfers, list)
            assert len(transfers) == 1
            assert transfers[0].packet_id == 1
            
            backend.shutdown()
    
    def test_backend_statistics_bindings_mode(self, mock_ns3_bindings, sample_topology):
        """Test statistics tracking in bindings mode."""
        with patch('simulation.ns3_backend.check_ns3_bindings', return_value=True):
            backend = NS3Backend(mode="bindings")
            backend.initialize(sample_topology)
            
            backend.send_packet("SAT-001", "SAT-002", packet_id=1, size_bytes=2048)
            backend.step(60.0)
            
            stats = backend.get_statistics()
            assert stats.total_packets_sent == 1
            assert stats.total_packets_received == 1
            assert stats.total_bytes_sent == 2048
            
            backend.shutdown()


# =============================================================================
# Test: Factory Function with Bindings Mode
# =============================================================================

class TestFactoryFunctionBindingsMode:
    """Tests for create_ns3_backend with bindings mode."""
    
    def test_factory_creates_bindings_mode(self):
        """Test factory function can create bindings mode backend."""
        backend = create_ns3_backend(mode="bindings")
        
        # Will fall back to mock without real bindings
        assert backend is not None
        assert backend.mode in (NS3Mode.BINDINGS, NS3Mode.MOCK)
    
    def test_factory_with_mock_bindings(self, mock_ns3_bindings):
        """Test factory function with mocked bindings."""
        with patch('simulation.ns3_backend.check_ns3_bindings', return_value=True):
            backend = create_ns3_backend(mode="bindings")
            assert backend.mode == NS3Mode.BINDINGS


# =============================================================================
# Test: Position Updates
# =============================================================================

class TestPositionUpdates:
    """Tests for position update functionality in bindings mode."""
    
    def test_wrapper_update_positions(self, mock_ns3_bindings, sample_topology):
        """Test wrapper can update node positions."""
        config = NS3Config()
        
        with patch('simulation.ns3_backend.check_ns3_bindings', return_value=True):
            wrapper = NS3BindingsWrapper(config)
            
            nodes = {}
            for node_data in sample_topology["nodes"]:
                nodes[node_data["id"]] = NS3Node(
                    id=node_data["id"],
                    node_type=node_data["type"],
                    position=np.array(node_data["position"])
                )
            
            links = set(tuple(link) for link in sample_topology["links"])
            wrapper.initialize(nodes, links)
            
            # Update positions
            nodes["SAT-001"].position = np.array([8000000, 0, 0])
            wrapper.update_positions(nodes)
            
            # Verify no errors (mock doesn't actually update)
            assert wrapper.initialized


# =============================================================================
# Test: Error Handling
# =============================================================================

class TestBindingsModeErrorHandling:
    """Tests for error handling in bindings mode."""
    
    def test_send_to_unknown_node(self, mock_ns3_bindings, sample_topology):
        """Test sending to unknown node returns False."""
        config = NS3Config()
        
        with patch('simulation.ns3_backend.check_ns3_bindings', return_value=True):
            wrapper = NS3BindingsWrapper(config)
            
            nodes = {}
            for node_data in sample_topology["nodes"]:
                nodes[node_data["id"]] = NS3Node(
                    id=node_data["id"],
                    node_type=node_data["type"],
                    position=np.array(node_data["position"])
                )
            
            wrapper.initialize(nodes, set())
            
            result = wrapper.send_packet("UNKNOWN", "SAT-001", packet_id=1, size_bytes=1024)
            assert result is False
    
    def test_backend_handles_wrapper_error(self, mock_ns3_bindings, sample_topology):
        """Test backend handles wrapper errors gracefully."""
        with patch('simulation.ns3_backend.check_ns3_bindings', return_value=True):
            backend = NS3Backend(mode="bindings")
            backend.initialize(sample_topology)
            
            # Force an error by clearing the wrapper
            original_wrapper = backend._bindings_wrapper
            backend._bindings_wrapper = None
            
            # Should fall back to mock mode
            backend.send_packet("SAT-001", "SAT-002", packet_id=1)
            transfers = backend.step(60.0)
            
            assert isinstance(transfers, list)
            
            # Restore for cleanup
            backend._bindings_wrapper = original_wrapper
            backend.shutdown()


# =============================================================================
# Test: Mode Comparison
# =============================================================================

class TestModeComparison:
    """Tests comparing bindings mode with other modes."""
    
    def test_all_modes_produce_transfers(self, sample_topology):
        """Test all modes produce packet transfers."""
        modes = ["mock", "file"]  # bindings falls back to mock without real NS-3
        
        for mode in modes:
            backend = NS3Backend(mode=mode)
            backend.initialize(sample_topology)
            
            backend.send_packet("SAT-001", "SAT-002", packet_id=1)
            transfers = backend.step(60.0)
            
            assert isinstance(transfers, list), f"Mode {mode} failed"
            assert len(transfers) == 1, f"Mode {mode} failed"
            
            backend.shutdown()
    
    def test_bindings_mode_same_interface(self, sample_topology):
        """Test bindings mode has same interface as other modes."""
        backend = NS3Backend(mode="bindings")
        
        # All these should work regardless of actual mode
        assert hasattr(backend, 'initialize')
        assert hasattr(backend, 'send_packet')
        assert hasattr(backend, 'step')
        assert hasattr(backend, 'get_statistics')
        assert hasattr(backend, 'shutdown')
        assert hasattr(backend, 'update_topology')


# =============================================================================
# Test: Backward Compatibility
# =============================================================================

class TestBackwardCompatibilityStep7:
    """Tests ensuring backward compatibility after Step 7."""
    
    def test_file_mode_unchanged(self, sample_topology):
        """Test file mode still works correctly."""
        backend = NS3Backend(mode="file")
        
        # Will fall back to mock without NS-3
        backend.initialize(sample_topology)
        backend.send_packet("SAT-001", "SAT-002", packet_id=1)
        transfers = backend.step(60.0)
        
        assert isinstance(transfers, list)
        backend.shutdown()
    
    def test_socket_mode_unchanged(self, sample_topology):
        """Test socket mode still works correctly."""
        backend = NS3Backend(mode="socket")
        backend.initialize(sample_topology)
        
        # Socket mode should fall back to mock without server
        backend.send_packet("SAT-001", "SAT-002", packet_id=1)
        transfers = backend.step(60.0)
        
        assert isinstance(transfers, list)
        backend.shutdown()
    
    def test_mock_mode_unchanged(self, sample_topology):
        """Test mock mode still works correctly."""
        backend = NS3Backend(mode="mock")
        backend.initialize(sample_topology)
        
        backend.send_packet("SAT-001", "SAT-002", packet_id=1)
        transfers = backend.step(60.0)
        
        assert isinstance(transfers, list)
        assert len(transfers) == 1
        backend.shutdown()
    
    def test_all_modes_available(self):
        """Test all modes are still available."""
        assert NS3Mode.FILE.value == "file"
        assert NS3Mode.SOCKET.value == "socket"
        assert NS3Mode.BINDINGS.value == "bindings"
        assert NS3Mode.MOCK.value == "mock"
    
    def test_new_exports_available(self):
        """Test new Step 7 exports are available."""
        from simulation import (
            NS3BindingsWrapper,
            NS3BindingsError,
            check_ns3_bindings,
            check_sns3_bindings,
        )
        
        assert NS3BindingsWrapper is not None
        assert NS3BindingsError is not None
        assert callable(check_ns3_bindings)
        assert callable(check_sns3_bindings)


# =============================================================================
# Test: Requires Real NS-3 Bindings (Skipped by Default)
# =============================================================================

@pytest.mark.skipif(not check_ns3_bindings(), reason="NS-3 Python bindings not installed")
class TestRealNS3Bindings:
    """Tests requiring actual NS-3 Python bindings."""
    
    def test_real_wrapper_creation(self):
        """Test wrapper creation with real bindings."""
        config = NS3Config()
        wrapper = NS3BindingsWrapper(config)
        assert wrapper is not None
    
    def test_real_simulation_step(self, sample_topology):
        """Test simulation step with real bindings."""
        backend = NS3Backend(mode="bindings")
        backend.initialize(sample_topology)
        
        backend.send_packet("SAT-001", "SAT-002", packet_id=1)
        transfers = backend.step(60.0)
        
        assert len(transfers) == 1
        backend.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])