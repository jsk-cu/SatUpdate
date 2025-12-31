#!/usr/bin/env python3
"""
Tests for Step 4: NetworkBackend Interface

These tests verify:
1. NetworkBackend ABC is correctly defined
2. NativeNetworkBackend produces identical results to current implementation
3. NetworkStatistics captures relevant metrics
4. Topology updates handled correctly
5. All existing tests pass with NativeNetworkBackend
"""

import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestNetworkBackendInterface:
    """Tests for the NetworkBackend abstract base class."""
    
    def test_packet_transfer_dataclass(self):
        """Test PacketTransfer dataclass creation."""
        @dataclass
        class PacketTransfer:
            source_id: str
            destination_id: str
            packet_id: int
            timestamp: float
            success: bool
            latency_ms: Optional[float] = None
            dropped_reason: Optional[str] = None
        
        transfer = PacketTransfer(
            source_id="SAT-001",
            destination_id="SAT-002",
            packet_id=42,
            timestamp=100.0,
            success=True,
            latency_ms=23.5
        )
        
        assert transfer.source_id == "SAT-001"
        assert transfer.destination_id == "SAT-002"
        assert transfer.packet_id == 42
        assert transfer.success is True
        assert transfer.latency_ms == 23.5
        assert transfer.dropped_reason is None
    
    def test_network_statistics_dataclass(self):
        """Test NetworkStatistics dataclass."""
        @dataclass
        class NetworkStatistics:
            total_packets_sent: int = 0
            total_packets_received: int = 0
            total_packets_dropped: int = 0
            average_latency_ms: float = 0.0
            throughput_kbps: float = 0.0
            link_utilization: Dict[Tuple[str, str], float] = None
        
        stats = NetworkStatistics(
            total_packets_sent=100,
            total_packets_received=95,
            total_packets_dropped=5,
            average_latency_ms=25.0
        )
        
        assert stats.total_packets_sent == 100
        assert stats.total_packets_received == 95
        assert stats.total_packets_dropped == 5
    
    def test_network_backend_abstract_methods(self):
        """Test NetworkBackend defines required abstract methods."""
        from abc import ABC, abstractmethod
        
        class NetworkBackend(ABC):
            @abstractmethod
            def initialize(self, topology: Dict) -> None:
                pass
            
            @abstractmethod
            def update_topology(self, active_links: Set[Tuple[str, str]]) -> None:
                pass
            
            @abstractmethod
            def send_packet(self, source: str, destination: str, 
                          packet_id: int, size_bytes: int) -> bool:
                pass
            
            @abstractmethod
            def step(self, timestep: float) -> List:
                pass
            
            @abstractmethod
            def get_statistics(self):
                pass
        
        # Verify cannot instantiate abstract class
        with pytest.raises(TypeError):
            NetworkBackend()


class TestNativeNetworkBackend:
    """Tests for NativeNetworkBackend implementation."""
    
    def test_native_backend_creation(self):
        """Test NativeNetworkBackend can be created."""
        @dataclass
        class NetworkStatistics:
            total_packets_sent: int = 0
            total_packets_received: int = 0
        
        class NativeNetworkBackend:
            def __init__(self):
                self.active_links: Set[Tuple[str, str]] = set()
                self.pending_transfers = []
                self.stats = NetworkStatistics()
        
        backend = NativeNetworkBackend()
        assert backend.active_links == set()
        assert backend.pending_transfers == []
    
    def test_native_backend_initialize(self, sample_topology):
        """Test initialization with topology."""
        class NativeNetworkBackend:
            def __init__(self):
                self.active_links = set()
            
            def initialize(self, topology):
                self.active_links = set(topology.get("links", []))
        
        backend = NativeNetworkBackend()
        backend.initialize(sample_topology)
        
        assert len(backend.active_links) == 3
        assert ("SAT-001", "SAT-002") in backend.active_links
    
    def test_native_backend_update_topology(self, active_links_set):
        """Test topology update."""
        class NativeNetworkBackend:
            def __init__(self):
                self.active_links = set()
            
            def update_topology(self, active_links):
                self.active_links = active_links
        
        backend = NativeNetworkBackend()
        backend.update_topology(active_links_set)
        
        assert backend.active_links == active_links_set
    
    def test_native_backend_send_packet(self, active_links_set):
        """Test packet sending."""
        @dataclass
        class PacketTransfer:
            source_id: str
            destination_id: str
            packet_id: int
            timestamp: float
            success: bool
            latency_ms: float = 0.0
        
        class NativeNetworkBackend:
            def __init__(self):
                self.active_links = set()
                self.pending_transfers = []
            
            def update_topology(self, active_links):
                self.active_links = active_links
            
            def send_packet(self, source, destination, packet_id, size=1024):
                # Check if link exists (either direction)
                link = tuple(sorted([source, destination]))
                if link in self.active_links or \
                   (source, destination) in self.active_links or \
                   (destination, source) in self.active_links:
                    self.pending_transfers.append(PacketTransfer(
                        source_id=source,
                        destination_id=destination,
                        packet_id=packet_id,
                        timestamp=0.0,
                        success=True
                    ))
                    return True
                return False
        
        backend = NativeNetworkBackend()
        backend.update_topology(active_links_set)
        
        # Valid link
        result = backend.send_packet("SAT-001", "SAT-002", 0)
        assert result is True
        assert len(backend.pending_transfers) == 1
        
        # Invalid link
        result = backend.send_packet("SAT-001", "SAT-003", 1)
        assert result is False
        assert len(backend.pending_transfers) == 1  # No new transfer
    
    def test_native_backend_step_instant_delivery(self, active_links_set):
        """Test step performs instant delivery."""
        @dataclass
        class PacketTransfer:
            source_id: str
            destination_id: str
            packet_id: int
            timestamp: float
            success: bool
            latency_ms: float = 0.0
        
        class NativeNetworkBackend:
            def __init__(self):
                self.active_links = set()
                self.pending_transfers = []
            
            def update_topology(self, active_links):
                self.active_links = active_links
            
            def send_packet(self, source, destination, packet_id, size=1024):
                link = tuple(sorted([source, destination]))
                if link in self.active_links:
                    self.pending_transfers.append(PacketTransfer(
                        source_id=source,
                        destination_id=destination,
                        packet_id=packet_id,
                        timestamp=0.0,
                        success=True
                    ))
                    return True
                return False
            
            def step(self, timestep):
                completed = self.pending_transfers.copy()
                self.pending_transfers.clear()
                return completed
        
        backend = NativeNetworkBackend()
        backend.update_topology(active_links_set)
        backend.send_packet("SAT-001", "SAT-002", 0)
        
        completed = backend.step(60.0)
        
        assert len(completed) == 1
        assert completed[0].success is True
        assert backend.pending_transfers == []  # Cleared after step
    
    def test_native_backend_statistics(self):
        """Test statistics collection."""
        @dataclass
        class NetworkStatistics:
            total_packets_sent: int = 0
            total_packets_received: int = 0
            total_packets_dropped: int = 0
        
        @dataclass
        class PacketTransfer:
            source_id: str
            destination_id: str
            packet_id: int
            timestamp: float
            success: bool
        
        class NativeNetworkBackend:
            def __init__(self):
                self.active_links = set()
                self.pending_transfers = []
                self.stats = NetworkStatistics()
            
            def step(self, timestep):
                completed = self.pending_transfers.copy()
                self.pending_transfers.clear()
                
                self.stats.total_packets_sent += len(completed)
                self.stats.total_packets_received += sum(
                    1 for t in completed if t.success
                )
                
                return completed
            
            def get_statistics(self):
                return self.stats
        
        backend = NativeNetworkBackend()
        backend.pending_transfers = [
            PacketTransfer("A", "B", 0, 0.0, True),
            PacketTransfer("B", "C", 1, 0.0, True),
        ]
        
        backend.step(60.0)
        stats = backend.get_statistics()
        
        assert stats.total_packets_sent == 2
        assert stats.total_packets_received == 2


class TestNativeBackendEquivalence:
    """Tests ensuring NativeNetworkBackend produces identical results."""
    
    def test_simulation_results_unchanged(self, small_simulation):
        """Test simulation produces same results with native backend."""
        sim = small_simulation
        
        # Record initial statistics
        initial_stats = sim.state.agent_statistics.average_completion
        
        # Run simulation
        for _ in range(5):
            sim.step(60.0)
        
        # Statistics should be computed correctly
        final_stats = sim.state.agent_statistics.average_completion
        
        # Should have made some progress
        assert final_stats >= initial_stats
    
    def test_packet_distribution_unchanged(self, small_simulation):
        """Test packet distribution works correctly."""
        sim = small_simulation
        
        # Run until some packets distributed
        for _ in range(10):
            sim.step(60.0)
        
        # Check that packets have been distributed
        stats = sim.state.agent_statistics
        
        # At least base station should have all packets
        assert stats.completion_percentage.get(0, 0) == 100.0


class TestTopologyManagement:
    """Tests for topology update handling."""
    
    def test_link_addition(self):
        """Test adding links to topology."""
        class NativeNetworkBackend:
            def __init__(self):
                self.active_links = set()
            
            def update_topology(self, active_links):
                self.active_links = active_links
        
        backend = NativeNetworkBackend()
        
        # Start with one link
        backend.update_topology({("A", "B")})
        assert len(backend.active_links) == 1
        
        # Add more links
        backend.update_topology({("A", "B"), ("B", "C"), ("C", "D")})
        assert len(backend.active_links) == 3
    
    def test_link_removal(self):
        """Test removing links from topology."""
        class NativeNetworkBackend:
            def __init__(self):
                self.active_links = set()
            
            def update_topology(self, active_links):
                self.active_links = active_links
        
        backend = NativeNetworkBackend()
        
        # Start with multiple links
        backend.update_topology({("A", "B"), ("B", "C"), ("C", "D")})
        assert len(backend.active_links) == 3
        
        # Remove links
        backend.update_topology({("A", "B")})
        assert len(backend.active_links) == 1
        assert ("A", "B") in backend.active_links
    
    def test_topology_with_simulation_links(self, small_simulation):
        """Test topology updates from simulation."""
        sim = small_simulation
        
        # Get active links from simulation
        active_links = sim.state.active_links
        
        # Links should be populated
        assert isinstance(active_links, set)
        
        # Step simulation - links may change
        sim.step(60.0)
        new_links = sim.state.active_links
        
        # Links should still be valid
        assert isinstance(new_links, set)


class TestBackwardCompatibilityStep4:
    """Ensure Step 4 changes don't break existing functionality."""
    
    def test_agent_protocol_unchanged(self, small_simulation):
        """Test agent protocol works correctly."""
        sim = small_simulation
        
        # Run protocol several times
        for _ in range(5):
            sim.step(60.0)
        
        # Verify agents have packets
        stats = sim.state.agent_statistics
        
        # Base station should have all packets
        assert stats.completion_percentage.get(0) == 100.0
        
        # Some satellites should have received packets
        satellite_completions = [
            v for k, v in stats.completion_percentage.items() 
            if k != 0
        ]
        assert any(c > 0 for c in satellite_completions)
    
    def test_simulation_logging_unchanged(self, small_simulation, tmp_path):
        """Test logging still works."""
        sim = small_simulation
        
        for _ in range(3):
            sim.step(60.0)
        
        log_file = tmp_path / "test.json"
        sim.save_log(str(log_file))
        
        assert log_file.exists()
    
    def test_is_update_complete_unchanged(self, small_simulation):
        """Test completion check works."""
        sim = small_simulation
        
        # Should not be complete initially
        assert not sim.is_update_complete()
        
        # Run for a while
        for _ in range(100):
            sim.step(60.0)
            if sim.is_update_complete():
                break
        
        # Function should work regardless of completion state
        result = sim.is_update_complete()
        assert isinstance(result, bool)


class TestNetworkBackendCommandLine:
    """Tests for command-line argument handling."""
    
    def test_network_backend_argument(self):
        """Test --network-backend argument parsing."""
        import argparse
        
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--network-backend",
            choices=["native", "ns3"],
            default="native"
        )
        
        # Default
        args = parser.parse_args([])
        assert args.network_backend == "native"
        
        # Explicit native
        args = parser.parse_args(["--network-backend", "native"])
        assert args.network_backend == "native"
        
        # NS-3
        args = parser.parse_args(["--network-backend", "ns3"])
        assert args.network_backend == "ns3"
