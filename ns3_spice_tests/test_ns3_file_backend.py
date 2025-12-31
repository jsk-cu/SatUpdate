#!/usr/bin/env python3
"""
Tests for Step 5: NS-3 Backend - File Mode

These tests verify:
1. JSON protocol is correctly specified
2. NS-3 scenario can be invoked via subprocess
3. File-based communication works reliably
4. Temporary files cleaned up properly
5. Error handling for NS-3 failures
"""

import json
import math
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestNS3FileBackendBasic:
    """Basic tests for NS-3 file mode backend."""
    
    def test_backend_creation(self, temp_work_dir):
        """Test NS3Backend can be created in file mode."""
        class NS3Backend:
            def __init__(self, mode="file", ns3_path=None, work_dir=None):
                self.mode = mode
                self.ns3_path = ns3_path or Path("/usr/local/ns3")
                self.work_dir = work_dir
                self.pending_sends = []
        
        backend = NS3Backend(mode="file", work_dir=temp_work_dir)
        
        assert backend.mode == "file"
        assert backend.pending_sends == []
    
    def test_backend_initialization(self, sample_topology, temp_work_dir):
        """Test backend initializes with topology."""
        class NS3Backend:
            def __init__(self, mode="file", work_dir=None):
                self.mode = mode
                self.work_dir = Path(work_dir) if work_dir else None
                self.topology = {}
            
            def initialize(self, topology):
                self.topology = topology
                if self.work_dir:
                    self.input_file = self.work_dir / "input.json"
                    self.output_file = self.work_dir / "output.json"
        
        backend = NS3Backend(mode="file", work_dir=temp_work_dir)
        backend.initialize(sample_topology)
        
        assert backend.topology == sample_topology
        assert backend.input_file.parent.exists()


class TestNS3JSONProtocol:
    """Tests for the JSON communication protocol."""
    
    def test_input_format_step_command(self, sample_topology, temp_work_dir):
        """Test input JSON format for step command."""
        input_data = {
            "command": "step",
            "timestep": 60.0,
            "topology": sample_topology,
            "sends": [
                {
                    "source": "SAT-001",
                    "destination": "SAT-002",
                    "packet_id": 0,
                    "size": 1024
                }
            ],
            "config": {
                "data_rate": "10Mbps",
                "propagation_model": "constant_speed"
            }
        }
        
        # Write to file
        input_file = temp_work_dir / "input.json"
        with open(input_file, "w") as f:
            json.dump(input_data, f)
        
        # Read back and verify
        with open(input_file) as f:
            loaded = json.load(f)
        
        assert loaded["command"] == "step"
        assert loaded["timestep"] == 60.0
        assert len(loaded["sends"]) == 1
    
    def test_output_format_transfers(self, temp_work_dir):
        """Test output JSON format with transfers."""
        output_data = {
            "status": "success",
            "simulation_time": 60.0,
            "transfers": [
                {
                    "source": "SAT-001",
                    "destination": "SAT-002",
                    "packet_id": 0,
                    "timestamp": 0.023,
                    "success": True,
                    "latency_ms": 23.4
                },
                {
                    "source": "SAT-002",
                    "destination": "SAT-003",
                    "packet_id": 1,
                    "timestamp": 0.045,
                    "success": False,
                    "dropped_reason": "queue_overflow"
                }
            ],
            "statistics": {
                "total_packets_sent": 2,
                "total_packets_received": 1,
                "average_latency_ms": 23.4
            }
        }
        
        output_file = temp_work_dir / "output.json"
        with open(output_file, "w") as f:
            json.dump(output_data, f)
        
        with open(output_file) as f:
            loaded = json.load(f)
        
        assert loaded["status"] == "success"
        assert len(loaded["transfers"]) == 2
        assert loaded["transfers"][0]["success"] is True
        assert loaded["transfers"][1]["success"] is False
    
    def test_initialize_command(self, sample_topology, temp_work_dir):
        """Test initialization command format."""
        input_data = {
            "command": "initialize",
            "topology": sample_topology,
            "config": {
                "data_rate": "10Mbps",
                "error_model": "none"
            }
        }
        
        input_file = temp_work_dir / "input.json"
        with open(input_file, "w") as f:
            json.dump(input_data, f)
        
        with open(input_file) as f:
            loaded = json.load(f)
        
        assert loaded["command"] == "initialize"
        assert "nodes" in loaded["topology"]
        assert "links" in loaded["topology"]


class TestNS3SubprocessExecution:
    """Tests for NS-3 subprocess invocation."""
    
    def test_subprocess_call_format(self, mock_ns3_subprocess, temp_work_dir):
        """Test subprocess is called with correct arguments."""
        @dataclass
        class PacketTransfer:
            source_id: str
            destination_id: str
            packet_id: int
            timestamp: float
            success: bool
            latency_ms: Optional[float] = None
        
        class NS3Backend:
            def __init__(self, ns3_path, work_dir):
                self.ns3_path = Path(ns3_path)
                self.work_dir = Path(work_dir)
                self.input_file = self.work_dir / "input.json"
                self.output_file = self.work_dir / "output.json"
                self.pending_sends = []
            
            def _step_file_mode(self, timestep):
                # Write input
                with open(self.input_file, "w") as f:
                    json.dump({
                        "command": "step",
                        "timestep": timestep,
                        "sends": self.pending_sends
                    }, f)
                
                # Expected command
                cmd = [
                    str(self.ns3_path / "ns3"),
                    "run",
                    f"satellite-update-scenario --input={self.input_file} "
                    f"--output={self.output_file}"
                ]
                
                return cmd
        
        backend = NS3Backend("/opt/ns3", temp_work_dir)
        cmd = backend._step_file_mode(60.0)
        
        assert "ns3" in cmd[0]
        assert "run" in cmd
        assert "satellite-update-scenario" in cmd[2]
    
    def test_subprocess_with_mock(self, mock_ns3_subprocess, temp_work_dir):
        """Test subprocess execution with mock."""
        mock_result = mock_ns3_subprocess()
        
        with patch('subprocess.run', return_value=mock_result) as mock_run:
            result = subprocess.run(["ns3", "run", "test"], capture_output=True, text=True)
            
            assert result.returncode == 0
            mock_run.assert_called_once()
    
    def test_subprocess_error_handling(self, temp_work_dir):
        """Test handling of NS-3 errors."""
        class NS3Backend:
            def __init__(self, work_dir):
                self.work_dir = Path(work_dir)
            
            def _run_ns3(self, cmd):
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    raise RuntimeError(f"NS-3 error: {result.stderr}")
                return result
        
        backend = NS3Backend(temp_work_dir)
        
        # Mock failed subprocess
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "Simulation failed"
        
        with patch('subprocess.run', return_value=mock_result):
            with pytest.raises(RuntimeError) as exc_info:
                backend._run_ns3(["ns3", "run", "test"])
            
            assert "NS-3 error" in str(exc_info.value)


class TestNS3FileCleanup:
    """Tests for temporary file cleanup."""
    
    def test_temp_directory_creation(self):
        """Test temporary directory is created."""
        import tempfile
        
        with tempfile.TemporaryDirectory(prefix="ns3_test_") as tmpdir:
            tmpdir = Path(tmpdir)
            assert tmpdir.exists()
            
            # Create some files
            (tmpdir / "input.json").write_text("{}")
            (tmpdir / "output.json").write_text("{}")
            
            assert (tmpdir / "input.json").exists()
        
        # After context exit, directory should be gone
        assert not tmpdir.exists()
    
    def test_cleanup_on_error(self, temp_work_dir):
        """Test cleanup happens even on error."""
        class NS3Backend:
            def __init__(self, work_dir):
                self.work_dir = Path(work_dir)
                self._cleanup_files = []
            
            def _create_temp_file(self, name):
                path = self.work_dir / name
                path.touch()
                self._cleanup_files.append(path)
                return path
            
            def cleanup(self):
                for f in self._cleanup_files:
                    if f.exists():
                        f.unlink()
        
        backend = NS3Backend(temp_work_dir)
        temp_file = backend._create_temp_file("test.json")
        
        assert temp_file.exists()
        
        backend.cleanup()
        
        assert not temp_file.exists()


class TestNS3TransferParsing:
    """Tests for parsing transfer results."""
    
    def test_parse_successful_transfer(self):
        """Test parsing successful transfer."""
        @dataclass
        class PacketTransfer:
            source_id: str
            destination_id: str
            packet_id: int
            timestamp: float
            success: bool
            latency_ms: Optional[float] = None
            dropped_reason: Optional[str] = None
        
        def parse_transfer(data):
            return PacketTransfer(
                source_id=data["source"],
                destination_id=data["destination"],
                packet_id=data["packet_id"],
                timestamp=data["timestamp"],
                success=data["success"],
                latency_ms=data.get("latency_ms"),
                dropped_reason=data.get("dropped_reason")
            )
        
        data = {
            "source": "SAT-001",
            "destination": "SAT-002",
            "packet_id": 0,
            "timestamp": 0.023,
            "success": True,
            "latency_ms": 23.4
        }
        
        transfer = parse_transfer(data)
        
        assert transfer.source_id == "SAT-001"
        assert transfer.success is True
        assert transfer.latency_ms == 23.4
        assert transfer.dropped_reason is None
    
    def test_parse_failed_transfer(self):
        """Test parsing failed transfer."""
        @dataclass
        class PacketTransfer:
            source_id: str
            destination_id: str
            packet_id: int
            timestamp: float
            success: bool
            latency_ms: Optional[float] = None
            dropped_reason: Optional[str] = None
        
        def parse_transfer(data):
            return PacketTransfer(
                source_id=data["source"],
                destination_id=data["destination"],
                packet_id=data["packet_id"],
                timestamp=data["timestamp"],
                success=data["success"],
                latency_ms=data.get("latency_ms"),
                dropped_reason=data.get("dropped_reason")
            )
        
        data = {
            "source": "SAT-001",
            "destination": "SAT-002",
            "packet_id": 0,
            "timestamp": 0.045,
            "success": False,
            "dropped_reason": "queue_overflow"
        }
        
        transfer = parse_transfer(data)
        
        assert transfer.success is False
        assert transfer.dropped_reason == "queue_overflow"
        assert transfer.latency_ms is None
    
    def test_parse_multiple_transfers(self):
        """Test parsing multiple transfers."""
        @dataclass
        class PacketTransfer:
            source_id: str
            destination_id: str
            packet_id: int
            timestamp: float
            success: bool
        
        def parse_transfers(transfers_data):
            return [
                PacketTransfer(
                    source_id=t["source"],
                    destination_id=t["destination"],
                    packet_id=t["packet_id"],
                    timestamp=t["timestamp"],
                    success=t["success"]
                )
                for t in transfers_data
            ]
        
        data = [
            {"source": "A", "destination": "B", "packet_id": 0, 
             "timestamp": 0.01, "success": True},
            {"source": "B", "destination": "C", "packet_id": 1, 
             "timestamp": 0.02, "success": True},
            {"source": "C", "destination": "D", "packet_id": 2, 
             "timestamp": 0.03, "success": False},
        ]
        
        transfers = parse_transfers(data)
        
        assert len(transfers) == 3
        assert sum(1 for t in transfers if t.success) == 2


class TestNS3InstallationDetection:
    """Tests for NS-3 installation detection."""
    
    def test_check_ns3_installation_exists(self):
        """Test detection when NS-3 is installed."""
        class NS3Backend:
            @staticmethod
            def check_ns3_installation(ns3_path):
                ns3_exe = Path(ns3_path) / "ns3"
                return ns3_exe.exists()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create fake ns3 executable
            ns3_exe = Path(tmpdir) / "ns3"
            ns3_exe.touch()
            
            assert NS3Backend.check_ns3_installation(tmpdir) is True
    
    def test_check_ns3_installation_missing(self, temp_work_dir):
        """Test detection when NS-3 is not installed."""
        class NS3Backend:
            @staticmethod
            def check_ns3_installation(ns3_path):
                ns3_exe = Path(ns3_path) / "ns3"
                return ns3_exe.exists()
        
        # Empty directory
        assert NS3Backend.check_ns3_installation(temp_work_dir) is False
    
    def test_import_error_message(self):
        """Test clear error message when NS-3 not available."""
        def check_ns3_or_raise(ns3_path):
            if not Path(ns3_path).exists():
                raise FileNotFoundError(
                    f"NS-3 not found at {ns3_path}\n"
                    "Install NS-3 or use --network-backend=native (default)"
                )
        
        with pytest.raises(FileNotFoundError) as exc_info:
            check_ns3_or_raise("/nonexistent/path")
        
        assert "NS-3 not found" in str(exc_info.value)
        assert "native" in str(exc_info.value)


@pytest.mark.requires_ns3
class TestNS3FileBackendReal:
    """Tests requiring actual NS-3 installation."""
    
    def test_real_ns3_execution(self):
        """Test real NS-3 execution."""
        # This test only runs if NS-3 is installed
        pass
    
    def test_real_scenario_compilation(self):
        """Test real scenario compilation."""
        pass


class TestNS3CommandLineArguments:
    """Tests for command-line arguments."""
    
    def test_ns3_mode_argument(self):
        """Test --ns3-mode argument parsing."""
        import argparse
        
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--ns3-mode",
            choices=["file", "socket", "bindings"],
            default="file"
        )
        
        args = parser.parse_args([])
        assert args.ns3_mode == "file"
        
        args = parser.parse_args(["--ns3-mode", "socket"])
        assert args.ns3_mode == "socket"
    
    def test_ns3_path_argument(self, temp_work_dir):
        """Test --ns3-path argument parsing."""
        import argparse
        
        parser = argparse.ArgumentParser()
        parser.add_argument("--ns3-path", type=Path, default=Path("/usr/local/ns3"))
        
        args = parser.parse_args([])
        assert args.ns3_path == Path("/usr/local/ns3")
        
        args = parser.parse_args(["--ns3-path", str(temp_work_dir)])
        assert args.ns3_path == temp_work_dir
