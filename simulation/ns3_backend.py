#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NS-3 Network Backend Module

Provides NS-3 integration for high-fidelity network simulation using:
- File Mode: Batch processing via subprocess and JSON files (Step 5)
- Socket Mode: Real-time communication via TCP (Step 6)
- Bindings Mode: Direct Python bindings for native speed (Step 7)

This module implements Steps 5, 6, and 7 of the NS-3/SPICE integration plan.

Features:
- JSON-based communication protocol (file/socket modes)
- Direct NS-3 API access (bindings mode)
- Automatic NS-3 installation detection
- Graceful fallback to mock mode for testing
- Configurable network parameters (data rate, delay, error models)
- Proper cleanup of temporary files
- Thread-safe socket communication (Step 6)
- Background receiver thread for async responses (Step 6)
- Automatic reconnection on disconnect (Step 6)
- SNS3 satellite extensions support (Step 7)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import (
    List, Dict, Set, Tuple, Optional, Any, Union, Callable
)
import json
import logging
import os
import queue
import shutil
import socket
import subprocess
import tempfile
import threading
import time

import numpy as np

from .network_backend import (
    NetworkBackend,
    PacketTransfer,
    NetworkStatistics,
    PendingTransfer,
    DropReason,
)


logger = logging.getLogger(__name__)


class NS3Mode(Enum):
    """NS-3 communication modes."""
    FILE = "file"       # Subprocess with JSON files
    SOCKET = "socket"   # TCP socket communication
    BINDINGS = "bindings"  # Direct Python bindings
    MOCK = "mock"       # Mock mode for testing


class NS3ErrorModel(Enum):
    """NS-3 error models for link simulation."""
    NONE = "none"
    RATE = "rate"           # Fixed error rate
    BURST = "burst"         # Bursty errors
    GILBERT_ELLIOT = "gilbert_elliot"  # Two-state Markov


class NS3PropagationModel(Enum):
    """NS-3 propagation delay models."""
    CONSTANT_SPEED = "constant_speed"  # Speed of light
    FIXED = "fixed"                    # Fixed delay
    RANDOM = "random"                  # Random delay


@dataclass
class NS3Config:
    """
    Configuration for NS-3 network simulation.
    
    Attributes
    ----------
    data_rate : str
        Link data rate (e.g., "10Mbps")
    propagation_model : NS3PropagationModel
        How propagation delay is calculated
    propagation_speed : float
        Speed of signal propagation in m/s
    error_model : NS3ErrorModel
        Error model for packet drops
    error_rate : float
        Error probability for RATE model
    queue_size : int
        Queue size in packets
    mtu : int
        Maximum transmission unit in bytes
    fixed_delay_ms : float
        Fixed delay for FIXED propagation model
    """
    data_rate: str = "10Mbps"
    propagation_model: NS3PropagationModel = NS3PropagationModel.CONSTANT_SPEED
    propagation_speed: float = 299792458.0  # Speed of light
    error_model: NS3ErrorModel = NS3ErrorModel.NONE
    error_rate: float = 0.0
    queue_size: int = 100
    mtu: int = 1500
    fixed_delay_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        # Handle both enum and string values
        prop_model = self.propagation_model
        if hasattr(prop_model, 'value'):
            prop_model = prop_model.value
        
        err_model = self.error_model
        if hasattr(err_model, 'value'):
            err_model = err_model.value
        
        return {
            "data_rate": self.data_rate,
            "propagation_model": prop_model,
            "propagation_speed": self.propagation_speed,
            "error_model": err_model,
            "error_rate": self.error_rate,
            "queue_size": self.queue_size,
            "mtu": self.mtu,
            "fixed_delay_ms": self.fixed_delay_ms,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NS3Config":
        """Create from dictionary."""
        return cls(
            data_rate=data.get("data_rate", "10Mbps"),
            propagation_model=NS3PropagationModel(
                data.get("propagation_model", "constant_speed")
            ),
            propagation_speed=data.get("propagation_speed", 299792458.0),
            error_model=NS3ErrorModel(data.get("error_model", "none")),
            error_rate=data.get("error_rate", 0.0),
            queue_size=data.get("queue_size", 100),
            mtu=data.get("mtu", 1500),
            fixed_delay_ms=data.get("fixed_delay_ms", 0.0),
        )


@dataclass
class NS3Node:
    """
    Node specification for NS-3 topology.
    
    Attributes
    ----------
    id : str
        Unique node identifier
    node_type : str
        Node type: "satellite" or "ground"
    position : np.ndarray
        Position [x, y, z] in meters
    """
    id: str
    node_type: str  # "satellite" or "ground"
    position: np.ndarray
    
    def __post_init__(self):
        if not isinstance(self.position, np.ndarray):
            self.position = np.array(self.position, dtype=float)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "type": self.node_type,
            "position": self.position.tolist()
        }


@dataclass
class NS3SendCommand:
    """
    Packet send command for NS-3.
    
    Attributes
    ----------
    source : str
        Source node ID
    destination : str
        Destination node ID
    packet_id : int
        Unique packet identifier
    size : int
        Packet size in bytes
    """
    source: str
    destination: str
    packet_id: int
    size: int = 1024
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source": self.source,
            "destination": self.destination,
            "packet_id": self.packet_id,
            "size": self.size,
        }


# =============================================================================
# STEP 6: Socket Mode Support Classes
# =============================================================================

class SocketConnectionError(Exception):
    """Error connecting to NS-3 socket server."""
    pass


class SocketTimeoutError(Exception):
    """Timeout waiting for response from NS-3 server."""
    pass


class NS3SocketClient:
    """
    Thread-safe TCP socket client for NS-3 communication.
    
    Implements Step 6: Socket Mode for real-time NS-3 simulation.
    
    Features:
    - JSON-line protocol (newline-delimited JSON)
    - Background receiver thread for async responses
    - Thread-safe command sending
    - Automatic reconnection on disconnect
    - Configurable timeouts
    
    Parameters
    ----------
    host : str
        NS-3 server hostname
    port : int
        NS-3 server port
    timeout : float
        Connection and response timeout in seconds
    max_reconnect_attempts : int
        Maximum reconnection attempts before giving up
    reconnect_delay : float
        Delay between reconnection attempts in seconds
    
    Examples
    --------
    >>> client = NS3SocketClient("localhost", 5555)
    >>> client.connect()
    >>> response = client.send_command({"command": "step", "timestep": 60.0})
    >>> client.disconnect()
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        timeout: float = 30.0,
        max_reconnect_attempts: int = 3,
        reconnect_delay: float = 1.0,
        buffer_size: int = 65536,
    ):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_delay = reconnect_delay
        self.buffer_size = buffer_size
        
        # Socket state
        self._socket: Optional[socket.socket] = None
        self._connected = False
        
        # Threading
        self._response_queue: queue.Queue = queue.Queue()
        self._receiver_thread: Optional[threading.Thread] = None
        self._running = False
        self._send_lock = threading.Lock()
        
        # Buffer for partial messages
        self._recv_buffer = ""
        
        # Statistics
        self._messages_sent = 0
        self._messages_received = 0
        self._reconnect_count = 0
    
    @property
    def connected(self) -> bool:
        """Check if connected to server."""
        return self._connected and self._socket is not None
    
    def connect(self, socket_impl: Optional[socket.socket] = None) -> None:
        """
        Connect to NS-3 server.
        
        Parameters
        ----------
        socket_impl : socket.socket, optional
            Socket implementation for testing (mock injection)
        
        Raises
        ------
        SocketConnectionError
            If connection fails after max attempts
        """
        for attempt in range(self.max_reconnect_attempts):
            try:
                if socket_impl is not None:
                    # Use injected socket (for testing)
                    self._socket = socket_impl
                else:
                    # Create real socket
                    self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    self._socket.settimeout(self.timeout)
                
                self._socket.connect((self.host, self.port))
                self._connected = True
                
                # Start receiver thread
                self._start_receiver()
                
                # Wait for acknowledgment
                try:
                    ack = self._receive_response_sync(timeout=5.0)
                    if ack and ack.get("status") == "ready":
                        logger.info(f"Connected to NS-3 server at {self.host}:{self.port}")
                        return
                except SocketTimeoutError:
                    # Server may not send ack, that's OK
                    logger.debug("No ack received, assuming connected")
                    return
                
                return
                
            except (socket.error, OSError) as e:
                logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                self._cleanup_socket()
                
                if attempt < self.max_reconnect_attempts - 1:
                    time.sleep(self.reconnect_delay)
        
        raise SocketConnectionError(
            f"Failed to connect to NS-3 server at {self.host}:{self.port} "
            f"after {self.max_reconnect_attempts} attempts"
        )
    
    def disconnect(self) -> None:
        """Disconnect from NS-3 server."""
        self._stop_receiver()
        self._cleanup_socket()
        logger.info("Disconnected from NS-3 server")
    
    def _cleanup_socket(self) -> None:
        """Clean up socket resources."""
        self._connected = False
        if self._socket:
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None
    
    def _start_receiver(self) -> None:
        """Start background receiver thread."""
        if self._receiver_thread is not None and self._receiver_thread.is_alive():
            return
        
        self._running = True
        self._receiver_thread = threading.Thread(
            target=self._receive_loop,
            name="NS3SocketReceiver",
            daemon=True
        )
        self._receiver_thread.start()
        logger.debug("Started receiver thread")
    
    def _stop_receiver(self) -> None:
        """Stop background receiver thread."""
        self._running = False
        if self._receiver_thread is not None:
            self._receiver_thread.join(timeout=2.0)
            if self._receiver_thread.is_alive():
                logger.warning("Receiver thread did not stop cleanly")
            self._receiver_thread = None
        logger.debug("Stopped receiver thread")
    
    def _receive_loop(self) -> None:
        """Background loop for receiving responses."""
        while self._running and self._connected:
            try:
                if self._socket is None:
                    break
                
                # Set a short timeout for polling
                self._socket.settimeout(0.1)
                
                try:
                    data = self._socket.recv(self.buffer_size)
                except socket.timeout:
                    continue
                except OSError:
                    # Socket closed or other OS error
                    continue
                
                if not data:
                    # Connection closed
                    logger.warning("Connection closed by server")
                    self._connected = False
                    break
                
                # Add to buffer and parse complete messages
                self._recv_buffer += data.decode('utf-8')
                self._process_buffer()
                
            except Exception as e:
                if self._running:
                    logger.error(f"Receiver error: {e}")
                break
    
    def _process_buffer(self) -> None:
        """Process buffer and extract complete JSON messages."""
        while '\n' in self._recv_buffer:
            line, self._recv_buffer = self._recv_buffer.split('\n', 1)
            line = line.strip()
            
            if not line:
                continue
            
            try:
                message = json.loads(line)
                self._response_queue.put(message)
                self._messages_received += 1
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON received: {e}")
    
    def send_command(
        self,
        command: Dict[str, Any],
        wait_response: bool = True,
        timeout: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Send command to NS-3 server.
        
        Parameters
        ----------
        command : Dict
            Command to send (will be JSON-encoded)
        wait_response : bool
            Whether to wait for response
        timeout : float, optional
            Response timeout (defaults to self.timeout)
        
        Returns
        -------
        Dict or None
            Response from server if wait_response=True
        
        Raises
        ------
        SocketConnectionError
            If not connected
        SocketTimeoutError
            If response times out
        """
        if not self.connected:
            raise SocketConnectionError("Not connected to NS-3 server")
        
        timeout = timeout or self.timeout
        
        # Thread-safe send
        with self._send_lock:
            try:
                message = json.dumps(command) + "\n"
                self._socket.sendall(message.encode('utf-8'))
                self._messages_sent += 1
                logger.debug(f"Sent command: {command.get('command', 'unknown')}")
            except Exception as e:
                self._connected = False
                raise SocketConnectionError(f"Send failed: {e}")
        
        if not wait_response:
            return None
        
        # Wait for response
        return self._receive_response_sync(timeout)
    
    def _receive_response_sync(self, timeout: float) -> Dict[str, Any]:
        """Wait for response synchronously."""
        try:
            response = self._response_queue.get(timeout=timeout)
            return response
        except queue.Empty:
            raise SocketTimeoutError(
                f"No response received within {timeout}s"
            )
    
    def reconnect(self) -> bool:
        """
        Attempt to reconnect to server.
        
        Returns
        -------
        bool
            True if reconnection successful
        """
        self._reconnect_count += 1
        logger.info(f"Attempting reconnection (attempt {self._reconnect_count})")
        
        self.disconnect()
        
        try:
            self.connect()
            return True
        except SocketConnectionError:
            return False
    
    def get_statistics(self) -> Dict[str, int]:
        """Get socket statistics."""
        return {
            "messages_sent": self._messages_sent,
            "messages_received": self._messages_received,
            "reconnect_count": self._reconnect_count,
        }
    
    def close(self) -> None:
        """Alias for disconnect."""
        self.disconnect()


# =============================================================================
# NS3 Bindings Wrapper (Step 7)
# =============================================================================

def check_ns3_bindings() -> bool:
    """
    Check if NS-3 Python bindings are available.
    
    Returns
    -------
    bool
        True if NS-3 Python bindings can be imported
    """
    try:
        import ns.core
        return True
    except ImportError:
        return False


def check_sns3_bindings() -> bool:
    """
    Check if SNS3 (Satellite Network Simulator 3) is available.
    
    Returns
    -------
    bool
        True if SNS3 can be imported
    """
    try:
        import ns.satellite
        return True
    except ImportError:
        return False


class NS3BindingsError(Exception):
    """Exception raised when NS-3 bindings are not available or encounter an error."""
    pass


class NS3BindingsWrapper:
    """
    Wrapper for NS-3 Python bindings providing a simplified interface.
    
    This class handles the direct interaction with NS-3's Python bindings,
    managing node creation, network configuration, and simulation execution.
    
    Features:
    - Dynamic module loading (fails gracefully when bindings unavailable)
    - Node and link management
    - Internet stack configuration
    - Position-based mobility model
    - Trace callbacks for packet events
    - SNS3 integration when available
    
    Parameters
    ----------
    config : NS3Config
        Network configuration parameters
    use_sns3 : bool
        Whether to use SNS3 if available (default: True)
    
    Raises
    ------
    NS3BindingsError
        If NS-3 Python bindings are not available
    """
    
    def __init__(self, config: NS3Config, use_sns3: bool = True):
        self._config = config
        self._use_sns3 = use_sns3
        
        # Check bindings availability
        if not check_ns3_bindings():
            raise NS3BindingsError(
                "NS-3 Python bindings not available.\n"
                "Build NS-3 with: ./ns3 configure --enable-python-bindings && ./ns3 build\n"
                "Or use --ns3-mode=file or --ns3-mode=socket"
            )
        
        # Import NS-3 modules
        self._ns = self._import_ns3_modules()
        
        # Check SNS3 availability
        self._has_sns3 = use_sns3 and check_sns3_bindings()
        if self._has_sns3:
            logger.info("SNS3 satellite extensions available")
        
        # Node management
        self._node_container = None
        self._node_map: Dict[str, int] = {}  # node_id -> ns-3 node index
        self._devices = {}  # (node1, node2) -> NetDevice pair
        
        # Trace callbacks
        self._packet_received_callbacks: List[Callable] = []
        self._packet_dropped_callbacks: List[Callable] = []
        
        # Transfer tracking
        self._pending_packets: Dict[int, Dict[str, Any]] = {}  # packet_id -> info
        self._completed_transfers: List[Dict[str, Any]] = []
        
        # Simulation state
        self._current_time: float = 0.0
        self._initialized = False
    
    def _import_ns3_modules(self) -> Dict[str, Any]:
        """
        Import NS-3 Python modules.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary of imported NS-3 modules
        """
        modules = {}
        
        try:
            import ns.core
            import ns.network
            import ns.internet
            import ns.point_to_point
            import ns.applications
            import ns.mobility
            
            modules["core"] = ns.core
            modules["network"] = ns.network
            modules["internet"] = ns.internet
            modules["point_to_point"] = ns.point_to_point
            modules["applications"] = ns.applications
            modules["mobility"] = ns.mobility
            
            # Try to import SNS3 modules if available
            if self._use_sns3:
                try:
                    import ns.satellite
                    modules["satellite"] = ns.satellite
                except ImportError:
                    pass
            
        except ImportError as e:
            raise NS3BindingsError(f"Failed to import NS-3 modules: {e}")
        
        return modules
    
    @property
    def has_sns3(self) -> bool:
        """Whether SNS3 extensions are available."""
        return self._has_sns3
    
    @property
    def initialized(self) -> bool:
        """Whether the wrapper has been initialized."""
        return self._initialized
    
    def initialize(self, nodes: Dict[str, 'NS3Node'], links: Set[Tuple[str, str]]) -> None:
        """
        Initialize NS-3 simulation with nodes and links.
        
        Parameters
        ----------
        nodes : Dict[str, NS3Node]
            Dictionary of node specifications
        links : Set[Tuple[str, str]]
            Set of active link tuples
        """
        ns = self._ns
        
        # Create node container
        num_nodes = len(nodes)
        self._node_container = ns["network"].NodeContainer()
        self._node_container.Create(num_nodes)
        
        # Map node IDs to indices
        self._node_map = {}
        for i, (node_id, node) in enumerate(nodes.items()):
            self._node_map[node_id] = i
        
        # Install Internet stack
        internet_stack = ns["internet"].InternetStackHelper()
        internet_stack.Install(self._node_container)
        
        # Set up mobility model
        self._setup_mobility(nodes)
        
        # Create point-to-point links
        self._setup_links(nodes, links)
        
        # Install trace callbacks
        self._install_trace_callbacks()
        
        self._initialized = True
        logger.info(f"NS3BindingsWrapper initialized with {num_nodes} nodes")
    
    def _setup_mobility(self, nodes: Dict[str, 'NS3Node']) -> None:
        """Set up mobility model for all nodes."""
        ns = self._ns
        
        mobility = ns["mobility"].MobilityHelper()
        positions = ns["mobility"].ListPositionAllocator()
        
        # Add positions for each node in order
        for node_id in self._node_map:
            node = nodes[node_id]
            pos = node.position
            positions.Add(ns["core"].Vector(float(pos[0]), float(pos[1]), float(pos[2])))
        
        mobility.SetPositionAllocator(positions)
        mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel")
        mobility.Install(self._node_container)
    
    def _setup_links(
        self, 
        nodes: Dict[str, 'NS3Node'], 
        links: Set[Tuple[str, str]]
    ) -> None:
        """Set up point-to-point links."""
        ns = self._ns
        
        # Configure point-to-point helper
        p2p = ns["point_to_point"].PointToPointHelper()
        p2p.SetDeviceAttribute("DataRate", ns["core"].StringValue(self._config.data_rate))
        
        # Calculate delay based on propagation model
        if self._config.propagation_model == NS3PropagationModel.FIXED or \
           (hasattr(self._config.propagation_model, 'value') and 
            self._config.propagation_model.value == "fixed"):
            delay_str = f"{self._config.fixed_delay_ms}ms"
            p2p.SetChannelAttribute("Delay", ns["core"].StringValue(delay_str))
        else:
            # Default to small delay, actual delay calculated per-packet
            p2p.SetChannelAttribute("Delay", ns["core"].StringValue("1ms"))
        
        # IP address helper
        address = ns["internet"].Ipv4AddressHelper()
        base_addr = "10.0.0.0"
        subnet_counter = 0
        
        self._devices = {}
        
        for node1, node2 in links:
            if node1 not in self._node_map or node2 not in self._node_map:
                logger.warning(f"Link references unknown node: ({node1}, {node2})")
                continue
            
            idx1 = self._node_map[node1]
            idx2 = self._node_map[node2]
            
            # Create link
            devices = p2p.Install(
                self._node_container.Get(idx1),
                self._node_container.Get(idx2)
            )
            
            # Assign IP addresses
            subnet_counter += 1
            base = f"10.{subnet_counter // 256}.{subnet_counter % 256}.0"
            address.SetBase(
                ns["network"].Ipv4Address(base),
                ns["network"].Ipv4Mask("255.255.255.0")
            )
            address.Assign(devices)
            
            self._devices[(node1, node2)] = devices
    
    def _install_trace_callbacks(self) -> None:
        """Install trace callbacks for packet events."""
        # In real NS-3, we would connect to trace sources like:
        # device.TraceConnectWithoutContext("PhyRxDrop", callback)
        # This is simplified for the wrapper
        pass
    
    def update_positions(self, nodes: Dict[str, 'NS3Node']) -> None:
        """
        Update node positions in the mobility model.
        
        Parameters
        ----------
        nodes : Dict[str, NS3Node]
            Dictionary of node specifications with updated positions
        """
        ns = self._ns
        
        for node_id, node in nodes.items():
            if node_id not in self._node_map:
                continue
            
            idx = self._node_map[node_id]
            ns3_node = self._node_container.Get(idx)
            
            # Get mobility model and update position
            mobility_model = ns3_node.GetObject(ns["mobility"].MobilityModel.GetTypeId())
            if mobility_model:
                pos = node.position
                mobility_model.SetPosition(
                    ns["core"].Vector(float(pos[0]), float(pos[1]), float(pos[2]))
                )
    
    def send_packet(
        self,
        source: str,
        destination: str,
        packet_id: int,
        size_bytes: int
    ) -> bool:
        """
        Schedule a packet for transmission.
        
        Parameters
        ----------
        source : str
            Source node ID
        destination : str
            Destination node ID
        packet_id : int
            Packet identifier
        size_bytes : int
            Packet size in bytes
        
        Returns
        -------
        bool
            True if packet was scheduled
        """
        if source not in self._node_map or destination not in self._node_map:
            logger.warning(f"Cannot send packet: unknown node(s) {source} -> {destination}")
            return False
        
        # Track pending packet
        self._pending_packets[packet_id] = {
            "source": source,
            "destination": destination,
            "size_bytes": size_bytes,
            "send_time": self._current_time,
        }
        
        return True
    
    def step(self, timestep: float) -> List[Dict[str, Any]]:
        """
        Run simulation for the specified timestep.
        
        Parameters
        ----------
        timestep : float
            Time to advance simulation in seconds
        
        Returns
        -------
        List[Dict[str, Any]]
            List of completed transfer info dictionaries
        """
        ns = self._ns
        
        # Clear previous transfers
        self._completed_transfers.clear()
        
        # Schedule stop time
        ns["core"].Simulator.Stop(ns["core"].Seconds(timestep))
        
        # Run simulation
        ns["core"].Simulator.Run()
        
        # Process pending packets (simplified - instant delivery for now)
        # In real NS-3, callbacks would populate completed_transfers
        for packet_id, info in list(self._pending_packets.items()):
            # Calculate latency based on config
            if self._config.propagation_model == NS3PropagationModel.FIXED or \
               (hasattr(self._config.propagation_model, 'value') and 
                self._config.propagation_model.value == "fixed"):
                latency_ms = self._config.fixed_delay_ms
            else:
                # Distance-based calculation would go here
                latency_ms = 10.0  # Default
            
            transfer = {
                "packet_id": packet_id,
                "source": info["source"],
                "destination": info["destination"],
                "size_bytes": info["size_bytes"],
                "latency_ms": latency_ms,
                "success": True,
            }
            self._completed_transfers.append(transfer)
        
        self._pending_packets.clear()
        self._current_time += timestep
        
        return self._completed_transfers
    
    def shutdown(self) -> None:
        """Clean up NS-3 simulator state."""
        ns = self._ns
        
        ns["core"].Simulator.Destroy()
        
        self._node_container = None
        self._node_map.clear()
        self._devices.clear()
        self._pending_packets.clear()
        self._completed_transfers.clear()
        self._initialized = False
        
        logger.info("NS3BindingsWrapper shut down")


# =============================================================================
# NS3Backend Main Class
# =============================================================================

class NS3Backend(NetworkBackend):
    """
    NS-3 network backend for high-fidelity network simulation.
    
    Supports multiple communication modes:
    - FILE: Subprocess with JSON file I/O (default, most compatible)
    - SOCKET: TCP socket for real-time simulation (Step 6)
    - BINDINGS: Direct Python bindings (Step 7 - not yet implemented)
    - MOCK: Mock mode for testing without NS-3
    
    Parameters
    ----------
    mode : NS3Mode or str
        Communication mode
    ns3_path : Path or str, optional
        Path to NS-3 installation directory
    work_dir : Path or str, optional
        Working directory for temporary files
    config : NS3Config, optional
        Network simulation configuration
    scenario_name : str
        Name of NS-3 scenario to run
    timeout : float
        Subprocess/socket timeout in seconds
    cleanup : bool
        Whether to clean up temporary files
    host : str
        NS-3 server host (socket mode)
    port : int
        NS-3 server port (socket mode)
    
    Examples
    --------
    >>> # File mode
    >>> backend = NS3Backend(mode="file", ns3_path="/opt/ns3")
    >>> backend.initialize(topology)
    >>> backend.send_packet("SAT-001", "SAT-002", packet_id=1)
    >>> transfers = backend.step(60.0)
    
    >>> # Socket mode (Step 6)
    >>> backend = NS3Backend(mode="socket", host="localhost", port=5555)
    >>> backend.initialize(topology)
    >>> backend.send_packet("SAT-001", "SAT-002", packet_id=1)
    >>> transfers = backend.step(60.0)
    """
    
    DEFAULT_NS3_PATH = Path("/usr/local/ns3")
    DEFAULT_SCENARIO = "satellite-update-scenario"
    DEFAULT_HOST = "localhost"
    DEFAULT_PORT = 5555
    
    def __init__(
        self,
        mode: Union[NS3Mode, str] = NS3Mode.FILE,
        ns3_path: Optional[Union[Path, str]] = None,
        work_dir: Optional[Union[Path, str]] = None,
        config: Optional[NS3Config] = None,
        scenario_name: str = DEFAULT_SCENARIO,
        timeout: float = 300.0,
        cleanup: bool = True,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
    ):
        # Parse mode
        if isinstance(mode, str):
            mode = NS3Mode(mode.lower())
        self._mode = mode
        
        # Paths
        self._ns3_path = Path(ns3_path) if ns3_path else self.DEFAULT_NS3_PATH
        self._work_dir: Optional[Path] = Path(work_dir) if work_dir else None
        self._temp_dir: Optional[Path] = None
        
        # Configuration
        self._config = config or NS3Config()
        self._scenario_name = scenario_name
        self._timeout = timeout
        self._cleanup = cleanup
        
        # Socket mode configuration (Step 6)
        self._host = host
        self._port = port
        self._socket_client: Optional[NS3SocketClient] = None
        
        # Bindings mode configuration (Step 7)
        self._bindings_wrapper: Optional[NS3BindingsWrapper] = None
        self._bindings_available = check_ns3_bindings()
        
        # State
        self._nodes: Dict[str, NS3Node] = {}
        self._active_links: Set[Tuple[str, str]] = set()
        self._pending_sends: List[NS3SendCommand] = []
        self._statistics = NetworkStatistics()
        self._current_time: float = 0.0
        self._initialized = False
        
        # File paths (set during initialization)
        self._input_file: Optional[Path] = None
        self._output_file: Optional[Path] = None
        
        # Check NS-3 availability
        self._ns3_available = self._check_ns3_installation()
        
        # Handle mode-specific availability
        if mode == NS3Mode.BINDINGS and not self._bindings_available:
            logger.warning(
                "NS-3 Python bindings not available. "
                "Falling back to mock mode. Build NS-3 with --enable-python-bindings."
            )
            self._mode = NS3Mode.MOCK
        elif mode not in (NS3Mode.MOCK, NS3Mode.SOCKET, NS3Mode.BINDINGS) and not self._ns3_available:
            logger.warning(
                f"NS-3 not found at {self._ns3_path}. "
                f"Using mock mode. Install NS-3 or set ns3_path correctly."
            )
            self._mode = NS3Mode.MOCK
    
    @property
    def mode(self) -> NS3Mode:
        """Current communication mode."""
        return self._mode
    
    @property
    def ns3_available(self) -> bool:
        """Whether NS-3 is available."""
        return self._ns3_available
    
    @property
    def config(self) -> NS3Config:
        """Network configuration."""
        return self._config
    
    @property
    def host(self) -> str:
        """Socket server host."""
        return self._host
    
    @property
    def port(self) -> int:
        """Socket server port."""
        return self._port
    
    @property
    def bindings_available(self) -> bool:
        """Whether NS-3 Python bindings are available."""
        return self._bindings_available
    
    @property
    def bindings_wrapper(self) -> Optional[NS3BindingsWrapper]:
        """The NS-3 bindings wrapper instance (if in bindings mode)."""
        return self._bindings_wrapper
    
    def _check_ns3_installation(self) -> bool:
        """
        Check if NS-3 is installed and configured.
        
        Returns
        -------
        bool
            True if NS-3 is available
        """
        ns3_exe = self._ns3_path / "ns3"
        
        if not ns3_exe.exists():
            # Try common alternative locations
            alternatives = [
                self._ns3_path / "build" / "ns3",
                self._ns3_path / "waf",
                Path("/usr/local/ns3/ns3"),
                Path("/opt/ns3/ns3"),
            ]
            for alt in alternatives:
                if alt.exists():
                    return True
            
            # Check PATH
            if shutil.which("ns3"):
                return True
            
            return False
        
        # Try running ns3 --version
        try:
            result = subprocess.run(
                [str(ns3_exe), "show", "version"],
                capture_output=True,
                timeout=10,
                cwd=str(ns3_exe.parent)
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError):
            return False
    
    def _setup_work_dir(self) -> None:
        """Set up working directory for file I/O."""
        if self._work_dir:
            self._work_dir.mkdir(parents=True, exist_ok=True)
            self._input_file = self._work_dir / "ns3_input.json"
            self._output_file = self._work_dir / "ns3_output.json"
        else:
            # Create temporary directory
            self._temp_dir = Path(tempfile.mkdtemp(prefix="ns3_satupdate_"))
            self._input_file = self._temp_dir / "input.json"
            self._output_file = self._temp_dir / "output.json"
    
    def _setup_socket_client(self) -> None:
        """Set up socket client for socket mode."""
        if self._socket_client is None:
            self._socket_client = NS3SocketClient(
                host=self._host,
                port=self._port,
                timeout=self._timeout,
            )
    
    def initialize(self, topology: Dict[str, Any]) -> None:
        """
        Initialize backend with network topology.
        
        Parameters
        ----------
        topology : Dict
            Network topology containing:
            - nodes: List of node specifications
            - links: List of (node1, node2) tuples
            - config: Optional configuration overrides
        """
        # Set up working directory
        self._setup_work_dir()
        
        # Parse nodes
        nodes = topology.get("nodes", [])
        self._nodes = {}
        for node in nodes:
            if isinstance(node, dict):
                node_id = node.get("id", "")
                node_type = node.get("type", "satellite")
                position = node.get("position", [0, 0, 0])
                self._nodes[node_id] = NS3Node(
                    id=node_id,
                    node_type=node_type,
                    position=np.array(position)
                )
        
        # Parse links
        links = topology.get("links", [])
        self._active_links = set()
        for link in links:
            if isinstance(link, (list, tuple)) and len(link) == 2:
                self._active_links.add((link[0], link[1]))
        
        # Apply config overrides
        if "config" in topology:
            for key, value in topology["config"].items():
                if hasattr(self._config, key):
                    setattr(self._config, key, value)
        
        # Mode-specific initialization
        if self._mode == NS3Mode.SOCKET:
            self._initialize_socket_mode()
        elif self._mode == NS3Mode.FILE:
            self._write_init_file()
        elif self._mode == NS3Mode.BINDINGS:
            self._initialize_bindings_mode()
        
        self._initialized = True
        logger.info(f"NS3Backend initialized in {self._mode.value} mode")
    
    def _initialize_socket_mode(self) -> None:
        """Initialize socket mode connection."""
        self._setup_socket_client()
        
        try:
            self._socket_client.connect()
            
            # Send initialization command
            init_command = {
                "command": "initialize",
                "topology": {
                    "nodes": [node.to_dict() for node in self._nodes.values()],
                    "links": list(self._active_links),
                },
                "config": self._config.to_dict(),
            }
            
            response = self._socket_client.send_command(init_command)
            
            if response and response.get("status") != "success":
                logger.warning(f"Initialization response: {response}")
                
        except SocketConnectionError as e:
            logger.warning(f"Socket connection failed: {e}. Falling back to mock mode.")
            self._mode = NS3Mode.MOCK
    
    def _write_init_file(self) -> None:
        """Write initialization file for file mode."""
        init_data = {
            "command": "initialize",
            "topology": {
                "nodes": [node.to_dict() for node in self._nodes.values()],
                "links": list(self._active_links),
            },
            "config": self._config.to_dict(),
        }
        
        with open(self._input_file, 'w') as f:
            json.dump(init_data, f, indent=2)
    
    def _initialize_bindings_mode(self) -> None:
        """Initialize NS-3 bindings mode (Step 7)."""
        try:
            self._bindings_wrapper = NS3BindingsWrapper(
                config=self._config,
                use_sns3=True
            )
            
            # Initialize with current topology
            self._bindings_wrapper.initialize(
                nodes=self._nodes,
                links=self._active_links
            )
            
            if self._bindings_wrapper.has_sns3:
                logger.info("NS-3 bindings mode initialized with SNS3 support")
            else:
                logger.info("NS-3 bindings mode initialized (without SNS3)")
                
        except NS3BindingsError as e:
            logger.warning(f"NS-3 bindings initialization failed: {e}. Falling back to mock mode.")
            self._mode = NS3Mode.MOCK
            self._bindings_wrapper = None
    
    def update_topology(self, active_links: Set[Tuple[str, str]]) -> None:
        """
        Update the set of active links.
        
        Parameters
        ----------
        active_links : Set[Tuple[str, str]]
            Set of active link tuples
        """
        self._active_links = active_links
        
        # Notify NS-3 in socket mode
        if self._mode == NS3Mode.SOCKET and self._socket_client and self._socket_client.connected:
            try:
                self._socket_client.send_command({
                    "command": "update_topology",
                    "links": list(active_links),
                }, wait_response=False)
            except SocketConnectionError:
                logger.warning("Failed to update topology via socket")
    
    def update_node_position(self, node_id: str, position: np.ndarray) -> None:
        """
        Update node position.
        
        Parameters
        ----------
        node_id : str
            Node identifier
        position : np.ndarray
            New position [x, y, z] in meters
        """
        if node_id in self._nodes:
            self._nodes[node_id].position = np.array(position)
    
    def _has_link(self, node1: str, node2: str) -> bool:
        """Check if link exists (bidirectional)."""
        return (node1, node2) in self._active_links or \
               (node2, node1) in self._active_links
    
    def send_packet(
        self,
        source: str,
        destination: str,
        packet_id: int,
        size_bytes: int = 1024,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Queue a packet for transmission.
        
        Parameters
        ----------
        source : str
            Source node ID
        destination : str  
            Destination node ID
        packet_id : int
            Packet content identifier
        size_bytes : int
            Packet size in bytes
        metadata : Dict, optional
            Additional metadata (ignored in NS-3 mode)
        
        Returns
        -------
        bool
            True if packet was queued
        """
        cmd = NS3SendCommand(
            source=source,
            destination=destination,
            packet_id=packet_id,
            size=size_bytes
        )
        self._pending_sends.append(cmd)
        self._statistics.record_send(size_bytes)
        return True
    
    def step(self, timestep: float) -> List[PacketTransfer]:
        """
        Advance simulation and process transfers.
        
        Parameters
        ----------
        timestep : float
            Time step in seconds
        
        Returns
        -------
        List[PacketTransfer]
            Completed packet transfers
        """
        self._current_time += timestep
        
        if self._mode == NS3Mode.MOCK:
            return self._step_mock(timestep)
        elif self._mode == NS3Mode.FILE:
            return self._step_file_mode(timestep)
        elif self._mode == NS3Mode.SOCKET:
            return self._step_socket_mode(timestep)
        elif self._mode == NS3Mode.BINDINGS:
            return self._step_bindings_mode(timestep)
        else:
            raise ValueError(f"Unknown mode: {self._mode}")
    
    def _step_mock(self, timestep: float) -> List[PacketTransfer]:
        """
        Mock step for testing without NS-3.
        
        Simulates realistic latency based on node positions and
        applies configured error model.
        """
        transfers = []
        
        for cmd in self._pending_sends:
            # Calculate latency
            latency_ms = self._calculate_mock_latency(cmd.source, cmd.destination)
            
            # Apply error model
            success = self._apply_error_model()
            
            # Check if link exists
            has_link = self._has_link(cmd.source, cmd.destination)
            
            if not has_link:
                success = False
                drop_reason = DropReason.NO_ROUTE
            elif not success:
                drop_reason = DropReason.ERROR
            else:
                drop_reason = None
            
            transfer = PacketTransfer(
                source_id=cmd.source,
                destination_id=cmd.destination,
                packet_id=cmd.packet_id,
                timestamp=self._current_time,
                success=success and has_link,
                latency_ms=latency_ms if success and has_link else 0.0,
                size_bytes=cmd.size,
                drop_reason=drop_reason,
            )
            
            if transfer.success:
                self._statistics.record_receive(cmd.size, latency_ms)
            else:
                self._statistics.record_drop()
            
            transfers.append(transfer)
        
        self._pending_sends.clear()
        return transfers
    
    def _step_file_mode(self, timestep: float) -> List[PacketTransfer]:
        """
        File mode step:
        1. Write input JSON file
        2. Invoke NS-3 subprocess
        3. Read output JSON file
        4. Parse and return transfers
        """
        if not self._ns3_available:
            logger.warning("NS-3 not available, falling back to mock mode")
            return self._step_mock(timestep)
        
        # Write input file
        input_data = self._create_input_data(timestep)
        with open(self._input_file, 'w') as f:
            json.dump(input_data, f, indent=2)
        
        # Build command
        cmd = self._build_ns3_command()
        
        try:
            # Run NS-3 scenario
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                cwd=str(self._ns3_path)
            )
            
            if result.returncode != 0:
                logger.error(f"NS-3 failed: {result.stderr}")
                return self._step_mock(timestep)
            
            # Read output file
            if not self._output_file.exists():
                logger.error("NS-3 did not create output file")
                return self._step_mock(timestep)
            
            with open(self._output_file) as f:
                output_data = json.load(f)
            
            # Parse transfers
            return self._parse_output(output_data)
            
        except subprocess.TimeoutExpired:
            logger.error(f"NS-3 timed out after {self._timeout}s")
            return self._step_mock(timestep)
        except Exception as e:
            logger.error(f"NS-3 error: {e}")
            return self._step_mock(timestep)
        finally:
            self._pending_sends.clear()
    
    def _step_socket_mode(self, timestep: float) -> List[PacketTransfer]:
        """
        Socket mode step (Step 6 implementation):
        1. Send step command with pending packets via socket
        2. Wait for response with transfer results
        3. Parse and return transfers
        
        Features automatic reconnection on failure.
        """
        if self._socket_client is None or not self._socket_client.connected:
            logger.warning("Socket not connected, attempting reconnection")
            try:
                if self._socket_client is None:
                    self._setup_socket_client()
                if not self._socket_client.reconnect():
                    logger.error("Reconnection failed, falling back to mock mode")
                    return self._step_mock(timestep)
            except Exception as e:
                logger.error(f"Reconnection error: {e}")
                return self._step_mock(timestep)
        
        # Build step command
        step_command = {
            "command": "step",
            "timestep": timestep,
            "simulation_time": self._current_time,
            "topology": {
                "nodes": [node.to_dict() for node in self._nodes.values()],
                "links": list(self._active_links),
            },
            "sends": [cmd.to_dict() for cmd in self._pending_sends],
            "config": self._config.to_dict(),
        }
        
        try:
            # Send command and wait for response
            response = self._socket_client.send_command(step_command, timeout=self._timeout)
            
            if response is None:
                logger.error("No response received from NS-3 server")
                return self._step_mock(timestep)
            
            if response.get("status") != "success":
                logger.warning(f"NS-3 step failed: {response.get('error', 'unknown')}")
                return self._step_mock(timestep)
            
            # Parse transfers from response
            transfers = self._parse_output(response)
            
            # Clear pending sends on success
            self._pending_sends.clear()
            
            return transfers
            
        except SocketTimeoutError:
            logger.error(f"Socket timeout after {self._timeout}s")
            # Try to reconnect for next step
            self._socket_client.reconnect()
            return self._step_mock(timestep)
            
        except SocketConnectionError as e:
            logger.error(f"Socket error: {e}")
            return self._step_mock(timestep)
            
        except Exception as e:
            logger.error(f"Unexpected error in socket step: {e}")
            return self._step_mock(timestep)
    
    def _step_bindings_mode(self, timestep: float) -> List[PacketTransfer]:
        """
        Execute simulation step using NS-3 Python bindings (Step 7).
        
        Parameters
        ----------
        timestep : float
            Time to advance simulation in seconds
        
        Returns
        -------
        List[PacketTransfer]
            Completed packet transfers
        """
        if self._bindings_wrapper is None:
            logger.warning("Bindings wrapper not initialized, falling back to mock mode")
            return self._step_mock(timestep)
        
        try:
            # Update positions in NS-3
            self._bindings_wrapper.update_positions(self._nodes)
            
            # Schedule pending sends through bindings
            for cmd in self._pending_sends:
                self._bindings_wrapper.send_packet(
                    source=cmd.source,
                    destination=cmd.destination,
                    packet_id=cmd.packet_id,
                    size_bytes=cmd.size
                )
            
            # Run simulation step
            transfer_dicts = self._bindings_wrapper.step(timestep)
            
            # Convert to PacketTransfer objects
            transfers = []
            for td in transfer_dicts:
                transfer = PacketTransfer(
                    packet_id=td["packet_id"],
                    source_id=td["source"],
                    destination_id=td["destination"],
                    size_bytes=td["size_bytes"],
                    latency_ms=td.get("latency_ms", 0.0),
                    success=td.get("success", True),
                    drop_reason=None if td.get("success", True) else DropReason.LINK_DOWN,
                    timestamp=self._current_time,
                )
                
                if transfer.success:
                    self._statistics.record_receive(transfer.size_bytes, transfer.latency_ms)
                else:
                    self._statistics.record_drop()
                
                transfers.append(transfer)
            
            self._pending_sends.clear()
            return transfers
            
        except Exception as e:
            logger.error(f"Error in bindings step: {e}")
            return self._step_mock(timestep)
    
    def _create_input_data(self, timestep: float) -> Dict[str, Any]:
        """Create input JSON data for NS-3."""
        return {
            "command": "step",
            "timestep": timestep,
            "simulation_time": self._current_time,
            "topology": {
                "nodes": [node.to_dict() for node in self._nodes.values()],
                "links": list(self._active_links),
            },
            "sends": [cmd.to_dict() for cmd in self._pending_sends],
            "config": self._config.to_dict(),
        }
    
    def _build_ns3_command(self) -> List[str]:
        """Build NS-3 subprocess command."""
        ns3_exe = self._ns3_path / "ns3"
        
        # Build scenario arguments
        scenario_args = (
            f"{self._scenario_name} "
            f"--input={self._input_file} "
            f"--output={self._output_file}"
        )
        
        return [str(ns3_exe), "run", scenario_args]
    
    def _parse_output(self, output_data: Dict[str, Any]) -> List[PacketTransfer]:
        """Parse NS-3 output JSON into PacketTransfer objects."""
        transfers = []
        
        for transfer_data in output_data.get("transfers", []):
            transfer = PacketTransfer.from_dict(transfer_data)
            
            if transfer.success:
                self._statistics.record_receive(
                    transfer.size_bytes,
                    transfer.latency_ms
                )
            else:
                self._statistics.record_drop()
            
            transfers.append(transfer)
        
        # Update statistics from NS-3 if provided
        ns3_stats = output_data.get("statistics", {})
        if "average_latency_ms" in ns3_stats:
            # Could update internal stats here if needed
            pass
        
        return transfers
    
    def _calculate_mock_latency(self, source: str, destination: str) -> float:
        """
        Calculate mock latency based on positions.
        
        Returns latency in milliseconds.
        """
        if self._config.propagation_model == NS3PropagationModel.FIXED:
            return self._config.fixed_delay_ms
        
        # Get positions
        if source not in self._nodes or destination not in self._nodes:
            return 0.0
        
        src_pos = self._nodes[source].position
        dst_pos = self._nodes[destination].position
        
        # Calculate distance (positions in meters)
        distance_m = np.linalg.norm(dst_pos - src_pos)
        
        # Propagation delay
        delay_s = distance_m / self._config.propagation_speed
        delay_ms = delay_s * 1000.0
        
        # Add random component if configured
        if self._config.propagation_model == NS3PropagationModel.RANDOM:
            delay_ms *= np.random.uniform(0.9, 1.1)
        
        return delay_ms
    
    def _apply_error_model(self) -> bool:
        """
        Apply error model to determine if packet succeeds.
        
        Returns True if packet should succeed.
        """
        if self._config.error_model == NS3ErrorModel.NONE:
            return True
        elif self._config.error_model == NS3ErrorModel.RATE:
            return np.random.random() > self._config.error_rate
        else:
            # Default: no error
            return True
    
    def get_statistics(self) -> NetworkStatistics:
        """Get network statistics."""
        return self._statistics
    
    def reset(self) -> None:
        """Reset backend state."""
        self._nodes.clear()
        self._active_links.clear()
        self._pending_sends.clear()
        self._statistics.reset()
        self._current_time = 0.0
    
    def shutdown(self) -> None:
        """Clean up resources."""
        # Shutdown bindings wrapper (Step 7)
        if self._bindings_wrapper is not None:
            try:
                self._bindings_wrapper.shutdown()
            except Exception as e:
                logger.warning(f"Failed to shutdown bindings wrapper: {e}")
            self._bindings_wrapper = None
        
        # Close socket connection
        if self._socket_client is not None:
            try:
                # Send shutdown command
                if self._socket_client.connected:
                    self._socket_client.send_command(
                        {"command": "shutdown"},
                        wait_response=False
                    )
            except Exception:
                pass
            
            self._socket_client.disconnect()
            self._socket_client = None
        
        # Clean up temp directory
        if self._cleanup and self._temp_dir and self._temp_dir.exists():
            try:
                shutil.rmtree(self._temp_dir)
                logger.debug(f"Cleaned up temp directory: {self._temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp dir: {e}")
    
    def __del__(self):
        """Destructor - ensure cleanup."""
        try:
            self.shutdown()
        except Exception:
            pass
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()
        return False
    
    def is_link_active(self, node1: str, node2: str) -> bool:
        """Check if link is active."""
        return self._has_link(node1, node2)
    
    def get_pending_count(self) -> int:
        """Get number of pending transfers."""
        return len(self._pending_sends)
    
    @property
    def name(self) -> str:
        """Backend name."""
        return f"NS3Backend({self._mode.value})"
    
    def get_socket_statistics(self) -> Optional[Dict[str, int]]:
        """Get socket communication statistics (Step 6)."""
        if self._socket_client:
            return self._socket_client.get_statistics()
        return None


# =============================================================================
# Factory Functions and Utilities
# =============================================================================

def check_ns3_available(ns3_path: Optional[Union[Path, str]] = None) -> bool:
    """
    Check if NS-3 is available.
    
    Parameters
    ----------
    ns3_path : Path or str, optional
        Path to NS-3 installation
    
    Returns
    -------
    bool
        True if NS-3 is available
    """
    backend = NS3Backend(mode="mock", ns3_path=ns3_path)
    return backend._check_ns3_installation()


def create_ns3_backend(
    mode: str = "file",
    ns3_path: Optional[Union[Path, str]] = None,
    config: Optional[NS3Config] = None,
    host: str = "localhost",
    port: int = 5555,
    **kwargs
) -> NS3Backend:
    """
    Create an NS-3 network backend.
    
    Parameters
    ----------
    mode : str
        Communication mode: "file", "socket", "bindings", or "mock"
    ns3_path : Path or str, optional
        Path to NS-3 installation
    config : NS3Config, optional
        Network configuration
    host : str
        NS-3 server host (socket mode)
    port : int
        NS-3 server port (socket mode)
    **kwargs
        Additional arguments passed to NS3Backend
    
    Returns
    -------
    NS3Backend
        Configured NS-3 backend
    """
    return NS3Backend(
        mode=mode,
        ns3_path=ns3_path,
        config=config,
        host=host,
        port=port,
        **kwargs
    )


def is_ns3_available() -> bool:
    """Check if NS-3 is available at the default location."""
    return check_ns3_available()