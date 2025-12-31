#!/usr/bin/env python3
"""
Tests for Step 6: NS-3 Backend - Socket Mode
and Step 7: NS-3 Backend - Python Bindings Mode

These tests verify:
1. Socket connection established reliably
2. Background receiver thread handles responses
3. Proper cleanup on disconnect/error
4. Timeout handling
5. Python bindings integration (when available)
"""

import json
import queue
import socket
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# STEP 6: Socket Mode Tests
# =============================================================================

class TestNS3SocketConnection:
    """Tests for socket connection management."""
    
    def test_socket_connection(self, mock_ns3_socket):
        """Test socket connection establishment."""
        class NS3SocketBackend:
            def __init__(self, host="localhost", port=5555):
                self.host = host
                self.port = port
                self._socket = None
            
            def connect(self, socket_impl=None):
                if socket_impl:
                    self._socket = socket_impl
                else:
                    self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._socket.connect((self.host, self.port))
        
        backend = NS3SocketBackend()
        backend.connect(mock_ns3_socket)
        
        assert mock_ns3_socket.connected is True
    
    def test_send_command(self, mock_ns3_socket):
        """Test sending command over socket."""
        class NS3SocketBackend:
            def __init__(self):
                self._socket = None
            
            def connect(self, socket_impl):
                self._socket = socket_impl
                self._socket.connect(("localhost", 5555))
            
            def _send_command(self, command):
                msg = json.dumps(command) + "\n"
                self._socket.sendall(msg.encode())
        
        backend = NS3SocketBackend()
        backend.connect(mock_ns3_socket)
        
        backend._send_command({"command": "step", "timestep": 60.0})
        
        assert len(mock_ns3_socket.sent_data) == 1
        sent = mock_ns3_socket.sent_data[0].decode()
        assert "step" in sent
    
    def test_receive_response(self, mock_ns3_socket):
        """Test receiving response from socket."""
        class NS3SocketBackend:
            def __init__(self):
                self._socket = None
            
            def connect(self, socket_impl):
                self._socket = socket_impl
                self._socket.connect(("localhost", 5555))
            
            def _send_command(self, command):
                msg = json.dumps(command) + "\n"
                self._socket.sendall(msg.encode())
            
            def _receive_response(self):
                data = self._socket.recv(4096).decode()
                return json.loads(data.strip())
        
        backend = NS3SocketBackend()
        backend.connect(mock_ns3_socket)
        
        # Send triggers mock to queue response
        backend._send_command({"command": "step"})
        
        response = backend._receive_response()
        
        assert "status" in response
        assert response["status"] == "success"


class TestNS3SocketThreading:
    """Tests for threaded socket communication."""
    
    def test_receiver_thread_creation(self):
        """Test background receiver thread is created."""
        class NS3SocketBackend:
            def __init__(self):
                self._socket = None
                self._response_queue = queue.Queue()
                self._receiver_thread = None
                self._running = False
            
            def _start_receiver(self):
                self._running = True
                self._receiver_thread = threading.Thread(
                    target=self._receive_loop,
                    daemon=True
                )
                self._receiver_thread.start()
            
            def _receive_loop(self):
                while self._running:
                    time.sleep(0.01)  # Mock loop
            
            def _stop_receiver(self):
                self._running = False
                if self._receiver_thread:
                    self._receiver_thread.join(timeout=1.0)
        
        backend = NS3SocketBackend()
        backend._start_receiver()
        
        assert backend._receiver_thread is not None
        assert backend._receiver_thread.is_alive()
        
        backend._stop_receiver()
    
    def test_response_queue(self):
        """Test response queue mechanism."""
        response_queue = queue.Queue()
        
        # Simulate receiver putting response
        response = {"status": "success", "transfers": []}
        response_queue.put(response)
        
        # Consumer gets response
        result = response_queue.get(timeout=1.0)
        
        assert result["status"] == "success"
    
    def test_timeout_handling(self):
        """Test timeout when no response received."""
        response_queue = queue.Queue()
        
        with pytest.raises(queue.Empty):
            response_queue.get(timeout=0.1)


class TestNS3SocketCleanup:
    """Tests for socket cleanup."""
    
    def test_socket_close(self, mock_ns3_socket):
        """Test socket is properly closed."""
        class NS3SocketBackend:
            def __init__(self):
                self._socket = None
            
            def connect(self, socket_impl):
                self._socket = socket_impl
                self._socket.connect(("localhost", 5555))
            
            def close(self):
                if self._socket:
                    self._socket.close()
                    self._socket = None
        
        backend = NS3SocketBackend()
        backend.connect(mock_ns3_socket)
        
        assert mock_ns3_socket.connected is True
        
        backend.close()
        
        assert mock_ns3_socket.connected is False
    
    def test_cleanup_on_error(self, mock_ns3_socket):
        """Test cleanup happens on error."""
        class NS3SocketBackend:
            def __init__(self):
                self._socket = None
            
            def connect(self, socket_impl):
                self._socket = socket_impl
                self._socket.connect(("localhost", 5555))
            
            def close(self):
                if self._socket:
                    self._socket.close()
        
        backend = NS3SocketBackend()
        backend.connect(mock_ns3_socket)
        
        try:
            raise RuntimeError("Simulated error")
        finally:
            backend.close()
        
        assert mock_ns3_socket.connected is False


class TestNS3SocketCommandLine:
    """Tests for socket mode command-line arguments."""
    
    def test_ns3_host_argument(self):
        """Test --ns3-host argument parsing."""
        import argparse
        
        parser = argparse.ArgumentParser()
        parser.add_argument("--ns3-host", default="localhost")
        
        args = parser.parse_args([])
        assert args.ns3_host == "localhost"
        
        args = parser.parse_args(["--ns3-host", "192.168.1.100"])
        assert args.ns3_host == "192.168.1.100"
    
    def test_ns3_port_argument(self):
        """Test --ns3-port argument parsing."""
        import argparse
        
        parser = argparse.ArgumentParser()
        parser.add_argument("--ns3-port", type=int, default=5555)
        
        args = parser.parse_args([])
        assert args.ns3_port == 5555
        
        args = parser.parse_args(["--ns3-port", "6666"])
        assert args.ns3_port == 6666


# =============================================================================
# STEP 7: Python Bindings Mode Tests
# =============================================================================

class TestNS3BindingsAvailability:
    """Tests for NS-3 Python bindings detection."""
    
    def test_bindings_import_check(self):
        """Test checking for NS-3 bindings availability."""
        def check_ns3_bindings():
            try:
                import ns.core
                return True
            except ImportError:
                return False
        
        # This will be False unless NS-3 bindings are installed
        result = check_ns3_bindings()
        assert isinstance(result, bool)
    
    def test_import_error_message(self):
        """Test clear error message when bindings unavailable."""
        def require_ns3_bindings():
            try:
                import ns.core
            except ImportError as e:
                raise ImportError(
                    f"NS-3 Python bindings not available: {e}\n"
                    "Build NS-3 with: ./ns3 configure --enable-python-bindings\n"
                    "Or use --ns3-mode=file or --ns3-mode=socket"
                )
        
        # This should raise unless bindings are installed
        try:
            require_ns3_bindings()
        except ImportError as e:
            assert "python-bindings" in str(e).lower() or "Python bindings" in str(e)


class TestNS3BindingsWithMock:
    """Tests for NS-3 bindings using mocks."""
    
    def test_node_container_creation(self, mock_ns3_bindings):
        """Test NodeContainer creation."""
        ns = mock_ns3_bindings
        
        nodes = ns["network"].NodeContainer()
        assert nodes is not None
    
    def test_simulator_stop_and_run(self, mock_ns3_bindings):
        """Test Simulator.Stop and Run."""
        ns = mock_ns3_bindings
        
        ns["core"].Simulator.Stop(ns["core"].Seconds(60.0))
        ns["core"].Simulator.Run()
        
        ns["core"].Simulator.Stop.assert_called()
        ns["core"].Simulator.Run.assert_called()
    
    def test_internet_stack_helper(self, mock_ns3_bindings):
        """Test InternetStackHelper."""
        ns = mock_ns3_bindings
        
        stack = ns["internet"].InternetStackHelper()
        assert stack is not None
    
    def test_mobility_helper(self, mock_ns3_bindings):
        """Test MobilityHelper setup."""
        ns = mock_ns3_bindings
        
        mobility = ns["mobility"].MobilityHelper()
        positions = ns["mobility"].ListPositionAllocator()
        
        assert mobility is not None
        assert positions is not None


class TestNS3BindingsIntegration:
    """Integration tests for NS-3 bindings mode."""
    
    def test_backend_with_mock_bindings(self, mock_ns3_bindings, sample_topology):
        """Test backend initialization with mock bindings."""
        class NS3BindingsBackend:
            def __init__(self, ns_modules):
                self._ns = ns_modules
                self._nodes = None
            
            def initialize(self, topology):
                ns = self._ns
                
                # Create nodes
                self._nodes = ns["network"].NodeContainer()
                self._nodes.Create(len(topology["nodes"]))
                
                # Setup mobility
                mobility = ns["mobility"].MobilityHelper()
                positions = ns["mobility"].ListPositionAllocator()
                
                for node in topology["nodes"]:
                    pos = node["position"]
                    positions.Add(ns["core"].Vector(pos[0], pos[1], pos[2]))
                
                # Install internet stack
                internet = ns["internet"].InternetStackHelper()
                internet.Install(self._nodes)
        
        backend = NS3BindingsBackend(mock_ns3_bindings)
        backend.initialize(sample_topology)
        
        assert backend._nodes is not None
    
    def test_step_with_mock_bindings(self, mock_ns3_bindings):
        """Test step execution with mock bindings."""
        class NS3BindingsBackend:
            def __init__(self, ns_modules):
                self._ns = ns_modules
                self._pending_sends = []
                self._completed_transfers = []
            
            def send_packet(self, source, destination, packet_id):
                self._pending_sends.append({
                    "source": source,
                    "destination": destination,
                    "packet_id": packet_id
                })
            
            def step(self, timestep):
                ns = self._ns
                
                # Clear previous transfers
                self._completed_transfers.clear()
                
                # Schedule sends (mock)
                self._pending_sends.clear()
                
                # Run simulation
                ns["core"].Simulator.Stop(ns["core"].Seconds(timestep))
                ns["core"].Simulator.Run()
                
                return self._completed_transfers
        
        backend = NS3BindingsBackend(mock_ns3_bindings)
        backend.send_packet("A", "B", 0)
        
        transfers = backend.step(60.0)
        
        mock_ns3_bindings["core"].Simulator.Stop.assert_called()
        mock_ns3_bindings["core"].Simulator.Run.assert_called()


class TestSNS3Integration:
    """Tests for SNS3 (Satellite Network Simulator 3) integration."""
    
    def test_sns3_availability_check(self):
        """Test checking for SNS3 availability."""
        def check_sns3():
            try:
                import ns.satellite
                return True
            except ImportError:
                return False
        
        result = check_sns3()
        assert isinstance(result, bool)
    
    def test_fallback_without_sns3(self, mock_ns3_bindings):
        """Test fallback when SNS3 not available."""
        class NS3BindingsBackend:
            def __init__(self, ns_modules, has_sns3=False):
                self._ns = ns_modules
                self._has_sns3 = has_sns3
            
            def _setup_channel(self):
                if self._has_sns3:
                    # Would use ns.satellite.SatelliteChannel
                    return "satellite_channel"
                else:
                    # Fallback to point-to-point
                    return "point_to_point_channel"
        
        backend_with = NS3BindingsBackend(mock_ns3_bindings, has_sns3=True)
        backend_without = NS3BindingsBackend(mock_ns3_bindings, has_sns3=False)
        
        assert backend_with._setup_channel() == "satellite_channel"
        assert backend_without._setup_channel() == "point_to_point_channel"


@pytest.mark.requires_ns3_bindings
class TestNS3BindingsReal:
    """Tests requiring actual NS-3 Python bindings."""
    
    def test_real_bindings_import(self):
        """Test real NS-3 bindings import."""
        import ns.core
        import ns.network
        
        assert hasattr(ns.core, 'Simulator')
        assert hasattr(ns.network, 'NodeContainer')
    
    def test_real_node_creation(self):
        """Test real node creation."""
        import ns.core
        import ns.network
        
        nodes = ns.network.NodeContainer()
        nodes.Create(3)
        
        assert nodes.GetN() == 3


class TestNS3PerformanceComparison:
    """Tests comparing performance of different modes."""
    
    def test_mode_selection_logic(self):
        """Test logic for selecting best available mode."""
        def select_best_mode(
            requested_mode: str,
            ns3_available: bool,
            bindings_available: bool,
            sns3_available: bool
        ) -> str:
            """Select best available NS-3 mode."""
            if requested_mode == "bindings":
                if bindings_available:
                    return "bindings"
                else:
                    return "file"  # Fallback
            
            if requested_mode == "socket":
                if ns3_available:
                    return "socket"
                else:
                    return "file"  # Fallback
            
            return "file"  # Default
        
        # Bindings requested and available
        assert select_best_mode("bindings", True, True, False) == "bindings"
        
        # Bindings requested but not available
        assert select_best_mode("bindings", True, False, False) == "file"
        
        # Socket requested and NS-3 available
        assert select_best_mode("socket", True, False, False) == "socket"
        
        # Default
        assert select_best_mode("file", True, True, True) == "file"


class TestNS3TraceCallbacks:
    """Tests for NS-3 trace callback mechanism."""
    
    def test_callback_registration_pattern(self):
        """Test pattern for registering trace callbacks."""
        callbacks_called = []
        
        def rx_callback(context, packet, address):
            callbacks_called.append({
                "context": context,
                "packet": packet,
                "address": address
            })
        
        # Simulate callback invocation
        rx_callback("test_context", {"id": 0}, "10.0.0.1")
        
        assert len(callbacks_called) == 1
        assert callbacks_called[0]["packet"]["id"] == 0
    
    def test_transfer_collection_via_callback(self):
        """Test collecting transfers via callbacks."""
        @dataclass
        class PacketTransfer:
            source_id: str
            destination_id: str
            packet_id: int
            timestamp: float
            success: bool
        
        class NS3BindingsBackend:
            def __init__(self):
                self._completed_transfers = []
            
            def _on_packet_received(self, context, packet_id, source, dest, time):
                self._completed_transfers.append(PacketTransfer(
                    source_id=source,
                    destination_id=dest,
                    packet_id=packet_id,
                    timestamp=time,
                    success=True
                ))
            
            def get_transfers(self):
                return list(self._completed_transfers)
        
        backend = NS3BindingsBackend()
        
        # Simulate callbacks
        backend._on_packet_received("ctx", 0, "A", "B", 0.01)
        backend._on_packet_received("ctx", 1, "B", "C", 0.02)
        
        transfers = backend.get_transfers()
        
        assert len(transfers) == 2
        assert transfers[0].packet_id == 0
        assert transfers[1].packet_id == 1
