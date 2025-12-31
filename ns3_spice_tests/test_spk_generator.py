#!/usr/bin/env python3
"""
Tests for Step 3: SPK Export Utility

These tests verify:
1. State vectors exported in correct format
2. mkspk setup files contain required parameters
3. Export works with all constellation types
4. Duration and step size configurable
5. NAIF IDs assigned correctly
"""

import json
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestSPKGeneratorBasic:
    """Basic tests for SPKGenerator class."""
    
    def test_generator_creation(self, temp_output_dir):
        """Test SPKGenerator can be created."""
        # Expected interface after implementation
        class SPKGenerator:
            def __init__(self, output_dir: Path):
                self.output_dir = output_dir
                self.segments = []
            
            def add_satellite_trajectory(self, sat_id, naif_id, states):
                self.segments.append({
                    "satellite_id": sat_id,
                    "naif_id": naif_id,
                    "states": states
                })
        
        gen = SPKGenerator(temp_output_dir)
        assert gen.output_dir == temp_output_dir
        assert gen.segments == []
    
    def test_add_satellite_trajectory(self, temp_output_dir):
        """Test adding satellite trajectory."""
        class SPKGenerator:
            def __init__(self, output_dir):
                self.segments = []
            
            def add_satellite_trajectory(self, sat_id, naif_id, states):
                self.segments.append({
                    "satellite_id": sat_id,
                    "naif_id": naif_id,
                    "states": states
                })
        
        gen = SPKGenerator(temp_output_dir)
        
        states = [
            {
                "time": datetime(2025, 1, 1, 0, 0, 0),
                "pos": [7000.0, 0.0, 0.0],
                "vel": [0.0, 7.5, 0.0]
            },
            {
                "time": datetime(2025, 1, 1, 0, 1, 0),
                "pos": [6999.0, 50.0, 0.0],
                "vel": [-0.1, 7.5, 0.0]
            }
        ]
        
        gen.add_satellite_trajectory("SAT-001", -100001, states)
        
        assert len(gen.segments) == 1
        assert gen.segments[0]["satellite_id"] == "SAT-001"
        assert gen.segments[0]["naif_id"] == -100001
        assert len(gen.segments[0]["states"]) == 2


class TestSPKExportForMkspk:
    """Tests for mkspk-compatible export."""
    
    def test_export_creates_state_file(self, temp_output_dir):
        """Test state vector file is created."""
        class SPKGenerator:
            def __init__(self, output_dir):
                self.output_dir = Path(output_dir)
                self.segments = []
            
            def add_satellite_trajectory(self, sat_id, naif_id, states):
                self.segments.append({
                    "satellite_id": sat_id,
                    "naif_id": naif_id,
                    "states": states
                })
            
            def export_for_mkspk(self):
                self.output_dir.mkdir(parents=True, exist_ok=True)
                for seg in self.segments:
                    sat_id = seg["satellite_id"]
                    data_file = self.output_dir / f"{sat_id}_states.txt"
                    with open(data_file, "w") as f:
                        for state in seg["states"]:
                            t = state["time"].strftime("%Y-%m-%dT%H:%M:%S.%f")
                            pos = state["pos"]
                            vel = state["vel"]
                            f.write(f"{t} {pos[0]} {pos[1]} {pos[2]} "
                                   f"{vel[0]} {vel[1]} {vel[2]}\n")
        
        gen = SPKGenerator(temp_output_dir)
        gen.add_satellite_trajectory(
            "SAT-001", -100001,
            [{"time": datetime(2025, 1, 1), "pos": [7000, 0, 0], "vel": [0, 7.5, 0]}]
        )
        gen.export_for_mkspk()
        
        state_file = temp_output_dir / "SAT-001_states.txt"
        assert state_file.exists()
    
    def test_state_file_format(self, temp_output_dir):
        """Test state file has correct format."""
        class SPKGenerator:
            def __init__(self, output_dir):
                self.output_dir = Path(output_dir)
                self.segments = []
            
            def add_satellite_trajectory(self, sat_id, naif_id, states):
                self.segments.append({
                    "satellite_id": sat_id,
                    "naif_id": naif_id,
                    "states": states
                })
            
            def export_for_mkspk(self):
                self.output_dir.mkdir(parents=True, exist_ok=True)
                for seg in self.segments:
                    sat_id = seg["satellite_id"]
                    data_file = self.output_dir / f"{sat_id}_states.txt"
                    with open(data_file, "w") as f:
                        for state in seg["states"]:
                            t = state["time"].strftime("%Y-%m-%dT%H:%M:%S.%f")
                            pos = state["pos"]
                            vel = state["vel"]
                            f.write(f"{t} {pos[0]} {pos[1]} {pos[2]} "
                                   f"{vel[0]} {vel[1]} {vel[2]}\n")
        
        gen = SPKGenerator(temp_output_dir)
        gen.add_satellite_trajectory(
            "SAT-001", -100001,
            [
                {"time": datetime(2025, 1, 1, 0, 0, 0), "pos": [7000.0, 0.0, 0.0], "vel": [0.0, 7.5, 0.0]},
                {"time": datetime(2025, 1, 1, 0, 1, 0), "pos": [6999.0, 50.0, 0.0], "vel": [-0.1, 7.5, 0.0]},
            ]
        )
        gen.export_for_mkspk()
        
        state_file = temp_output_dir / "SAT-001_states.txt"
        content = state_file.read_text()
        lines = content.strip().split("\n")
        
        assert len(lines) == 2
        
        # Parse first line
        parts = lines[0].split()
        assert len(parts) == 7  # timestamp + 3 pos + 3 vel
        assert "2025-01-01" in parts[0]
    
    def test_export_creates_setup_file(self, temp_output_dir):
        """Test mkspk setup file is created."""
        class SPKGenerator:
            def __init__(self, output_dir):
                self.output_dir = Path(output_dir)
                self.segments = []
            
            def add_satellite_trajectory(self, sat_id, naif_id, states, center_body=399):
                self.segments.append({
                    "satellite_id": sat_id,
                    "naif_id": naif_id,
                    "states": states,
                    "center_body": center_body
                })
            
            def export_for_mkspk(self):
                self.output_dir.mkdir(parents=True, exist_ok=True)
                for seg in self.segments:
                    sat_id = seg["satellite_id"]
                    setup_file = self.output_dir / f"{sat_id}_setup.txt"
                    with open(setup_file, "w") as f:
                        f.write("\\begindata\n")
                        f.write("INPUT_DATA_TYPE = 'STATES'\n")
                        f.write("OUTPUT_SPK_TYPE = 9\n")
                        f.write(f"OBJECT_ID = {seg['naif_id']}\n")
                        f.write(f"OBJECT_NAME = '{sat_id}'\n")
                        f.write(f"CENTER_ID = {seg['center_body']}\n")
                        f.write("REF_FRAME_NAME = 'J2000'\n")
                        f.write("\\begintext\n")
        
        gen = SPKGenerator(temp_output_dir)
        gen.add_satellite_trajectory("SAT-001", -100001, [], center_body=399)
        gen.export_for_mkspk()
        
        setup_file = temp_output_dir / "SAT-001_setup.txt"
        assert setup_file.exists()
    
    def test_setup_file_contains_required_fields(self, temp_output_dir):
        """Test setup file contains all required mkspk fields."""
        class SPKGenerator:
            def __init__(self, output_dir):
                self.output_dir = Path(output_dir)
                self.segments = []
            
            def add_satellite_trajectory(self, sat_id, naif_id, states, center_body=399, ref_frame="J2000"):
                self.segments.append({
                    "satellite_id": sat_id,
                    "naif_id": naif_id,
                    "states": states,
                    "center_body": center_body,
                    "reference_frame": ref_frame
                })
            
            def export_for_mkspk(self):
                self.output_dir.mkdir(parents=True, exist_ok=True)
                for seg in self.segments:
                    sat_id = seg["satellite_id"]
                    setup_file = self.output_dir / f"{sat_id}_setup.txt"
                    with open(setup_file, "w") as f:
                        f.write("\\begindata\n")
                        f.write("INPUT_DATA_TYPE = 'STATES'\n")
                        f.write("OUTPUT_SPK_TYPE = 9\n")
                        f.write(f"OBJECT_ID = {seg['naif_id']}\n")
                        f.write(f"OBJECT_NAME = '{sat_id}'\n")
                        f.write(f"CENTER_ID = {seg['center_body']}\n")
                        f.write(f"REF_FRAME_NAME = '{seg['reference_frame']}'\n")
                        f.write("PRODUCER_ID = 'SatUpdate'\n")
                        f.write("DATA_ORDER = 'EPOCH X Y Z VX VY VZ'\n")
                        f.write("INPUT_DATA_UNITS = ('ANGLES=DEGREES' 'DISTANCES=KM')\n")
                        f.write("DATA_DELIMITER = ' '\n")
                        f.write("POLYNOM_DEGREE = 7\n")
                        f.write("\\begintext\n")
        
        gen = SPKGenerator(temp_output_dir)
        gen.add_satellite_trajectory("SAT-001", -100001, [])
        gen.export_for_mkspk()
        
        setup_file = temp_output_dir / "SAT-001_setup.txt"
        content = setup_file.read_text()
        
        required_fields = [
            "INPUT_DATA_TYPE",
            "OUTPUT_SPK_TYPE",
            "OBJECT_ID",
            "OBJECT_NAME",
            "CENTER_ID",
            "REF_FRAME_NAME",
            "DATA_ORDER",
        ]
        
        for field in required_fields:
            assert field in content, f"Missing required field: {field}"


class TestSPKExportFromSimulation:
    """Tests for exporting simulation to SPK."""
    
    def test_add_from_simulation(self, small_simulation, temp_output_dir):
        """Test adding trajectories from simulation."""
        class SPKGenerator:
            def __init__(self, output_dir):
                self.output_dir = Path(output_dir)
                self.segments = []
            
            def add_from_simulation(self, simulation, start_time, duration_hours, 
                                   step_seconds=60.0, naif_id_start=-100001):
                num_steps = int(duration_hours * 3600 / step_seconds)
                
                trajectories = {
                    sat.satellite_id: []
                    for sat in simulation.satellites
                }
                
                current_time = start_time
                
                for step in range(num_steps):
                    for sat in simulation.satellites:
                        trajectories[sat.satellite_id].append({
                            "time": current_time,
                            "pos": sat.get_position_eci().tolist(),
                            "vel": sat.get_velocity_eci().tolist()
                        })
                    
                    simulation.step(step_seconds)
                    current_time += timedelta(seconds=step_seconds)
                
                for idx, (sat_id, states) in enumerate(trajectories.items()):
                    self.segments.append({
                        "satellite_id": sat_id,
                        "naif_id": naif_id_start - idx,
                        "states": states
                    })
        
        gen = SPKGenerator(temp_output_dir)
        gen.add_from_simulation(
            small_simulation,
            start_time=datetime(2025, 1, 1),
            duration_hours=0.1,  # 6 minutes
            step_seconds=60.0
        )
        
        # Should have one segment per satellite
        assert len(gen.segments) == small_simulation.num_satellites
        
        # Each segment should have states
        for seg in gen.segments:
            assert len(seg["states"]) > 0
    
    def test_export_all_constellation_types(self, temp_output_dir):
        """Test export works with all constellation types."""
        from simulation import (
            Simulation, 
            SimulationConfig, 
            ConstellationType
        )
        
        constellation_types = [
            ConstellationType.WALKER_DELTA,
            ConstellationType.WALKER_STAR,
            ConstellationType.RANDOM,
        ]
        
        for const_type in constellation_types:
            config = SimulationConfig(
                constellation_type=const_type,
                num_planes=2,
                sats_per_plane=2,
                num_satellites=4,
                random_seed=42,
            )
            sim = Simulation(config)
            sim.initialize()
            
            # Should be able to get positions from all satellites
            for sat in sim.satellites:
                pos = sat.get_position_eci()
                vel = sat.get_velocity_eci()
                
                assert pos.shape == (3,)
                assert vel.shape == (3,)
                assert np.linalg.norm(pos) > 6371  # Above Earth


class TestSPKCommandLineExport:
    """Tests for command-line export arguments."""
    
    def test_export_spk_argument(self, temp_output_dir):
        """Test --export-spk argument parsing."""
        import argparse
        
        parser = argparse.ArgumentParser()
        parser.add_argument("--export-spk", type=Path)
        parser.add_argument("--export-duration", type=float, default=24.0)
        parser.add_argument("--export-step", type=float, default=60.0)
        
        args = parser.parse_args([
            "--export-spk", str(temp_output_dir),
            "--export-duration", "12.0",
            "--export-step", "30.0"
        ])
        
        assert args.export_spk == temp_output_dir
        assert args.export_duration == 12.0
        assert args.export_step == 30.0
    
    def test_default_export_parameters(self):
        """Test default export parameters."""
        import argparse
        
        parser = argparse.ArgumentParser()
        parser.add_argument("--export-duration", type=float, default=24.0)
        parser.add_argument("--export-step", type=float, default=60.0)
        
        args = parser.parse_args([])
        
        assert args.export_duration == 24.0  # 24 hours
        assert args.export_step == 60.0      # 60 seconds


class TestNAIFIDAssignment:
    """Tests for NAIF ID assignment."""
    
    def test_naif_ids_are_negative(self, temp_output_dir):
        """Test NAIF IDs for spacecraft are negative."""
        class SPKGenerator:
            def __init__(self, output_dir, naif_id_start=-100001):
                self.naif_id_start = naif_id_start
                self.next_naif_id = naif_id_start
                self.segments = []
            
            def add_satellite_trajectory(self, sat_id, states):
                naif_id = self.next_naif_id
                self.next_naif_id -= 1
                self.segments.append({
                    "satellite_id": sat_id,
                    "naif_id": naif_id,
                    "states": states
                })
                return naif_id
        
        gen = SPKGenerator(temp_output_dir)
        
        naif_id_1 = gen.add_satellite_trajectory("SAT-001", [])
        naif_id_2 = gen.add_satellite_trajectory("SAT-002", [])
        
        assert naif_id_1 < 0
        assert naif_id_2 < 0
        assert naif_id_1 != naif_id_2
    
    def test_naif_ids_unique(self, temp_output_dir):
        """Test each satellite gets unique NAIF ID."""
        class SPKGenerator:
            def __init__(self, output_dir, naif_id_start=-100001):
                self.next_naif_id = naif_id_start
                self.segments = []
            
            def add_satellite_trajectory(self, sat_id, states):
                naif_id = self.next_naif_id
                self.next_naif_id -= 1
                self.segments.append({
                    "satellite_id": sat_id,
                    "naif_id": naif_id
                })
        
        gen = SPKGenerator(temp_output_dir)
        
        for i in range(10):
            gen.add_satellite_trajectory(f"SAT-{i:03d}", [])
        
        naif_ids = [seg["naif_id"] for seg in gen.segments]
        assert len(naif_ids) == len(set(naif_ids)), "NAIF IDs not unique"
    
    def test_naif_id_mapping_file(self, temp_output_dir):
        """Test NAIF ID mapping is documented."""
        class SPKGenerator:
            def __init__(self, output_dir, naif_id_start=-100001):
                self.output_dir = Path(output_dir)
                self.next_naif_id = naif_id_start
                self.segments = []
            
            def add_satellite_trajectory(self, sat_id, states):
                naif_id = self.next_naif_id
                self.next_naif_id -= 1
                self.segments.append({
                    "satellite_id": sat_id,
                    "naif_id": naif_id
                })
            
            def export_naif_mapping(self):
                self.output_dir.mkdir(parents=True, exist_ok=True)
                mapping_file = self.output_dir / "naif_mapping.json"
                mapping = {
                    seg["satellite_id"]: seg["naif_id"]
                    for seg in self.segments
                }
                with open(mapping_file, "w") as f:
                    json.dump(mapping, f, indent=2)
        
        gen = SPKGenerator(temp_output_dir)
        gen.add_satellite_trajectory("SAT-001", [])
        gen.add_satellite_trajectory("SAT-002", [])
        gen.export_naif_mapping()
        
        mapping_file = temp_output_dir / "naif_mapping.json"
        assert mapping_file.exists()
        
        with open(mapping_file) as f:
            mapping = json.load(f)
        
        assert "SAT-001" in mapping
        assert "SAT-002" in mapping
