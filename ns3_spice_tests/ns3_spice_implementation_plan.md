# NS-3 and SPICE Integration Implementation Plan

## Executive Summary

This document outlines the implementation plan for adding optional NS-3 network simulation and SPICE ephemeris support to the SatUpdate satellite constellation simulator. These features are designed as **opt-in extensions** that preserve all existing functionality while enabling more sophisticated simulation capabilities.

### Design Principles

1. **Backward Compatibility**: All existing functionality remains unchanged. New features are activated only through explicit configuration.
2. **Graceful Degradation**: If optional dependencies (SpiceyPy, NS-3) are not installed, the system falls back to native implementations.
3. **Pluggable Architecture**: Abstract interfaces allow swapping implementations without changing core simulation logic.
4. **Incremental Adoption**: Each feature can be enabled independently.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         SatUpdate Core                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────────┐    ┌──────────────────────┐                   │
│  │  TrajectoryProvider  │    │   NetworkBackend     │                   │
│  │     (Abstract)       │    │     (Abstract)       │                   │
│  └──────────┬───────────┘    └──────────┬───────────┘                   │
│             │                           │                                │
│      ┌──────┴──────┐             ┌──────┴──────┐                        │
│      │             │             │             │                        │
│  ┌───▼────┐   ┌────▼────┐   ┌────▼────┐   ┌───▼────┐                   │
│  │Keplerian│  │  SPICE  │   │ Native  │   │  NS-3  │                   │
│  │Provider │  │ Provider│   │ Backend │   │ Backend│                   │
│  │(default)│  │(opt-in) │   │(default)│   │(opt-in)│                   │
│  └─────────┘  └─────────┘   └─────────┘   └────────┘                   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### New Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--trajectory-provider` | Trajectory source: `keplerian`, `spice` | `keplerian` |
| `--spice-kernels-dir` | Directory containing SPICE kernels | None |
| `--spice-config` | Path to SPICE constellation config JSON | None |
| `--network-backend` | Network simulator: `native`, `ns3` | `native` |
| `--ns3-mode` | NS-3 communication: `file`, `socket`, `bindings` | `file` |
| `--ns3-path` | Path to NS-3 installation | `/usr/local/ns3` |
| `--ns3-host` | NS-3 server host (socket mode) | `localhost` |
| `--ns3-port` | NS-3 server port (socket mode) | `5555` |
| `--export-spk` | Export constellation to SPK format | None |

---

## Implementation Steps

### Step 1: TrajectoryProvider Interface

**Goal**: Create an abstract interface for satellite position computation that decouples trajectory calculation from the core simulation.

**Files to Create**:
- `simulation/trajectory.py`

**Changes to Existing Files**:
- `simulation/simulation.py` - Add optional trajectory provider parameter
- `simulation/__init__.py` - Export new classes

#### Detailed Design

```python
# simulation/trajectory.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple, Optional
import numpy as np
```

The `TrajectoryProvider` abstract base class defines the interface:

| Method | Description |
|--------|-------------|
| `get_state(satellite_id, time)` | Returns full state vector (position + velocity) |
| `get_position_eci(satellite_id, time)` | Returns position only (more efficient) |
| `get_satellite_ids()` | Lists available satellites |
| `get_time_bounds(satellite_id)` | Returns valid time range |

The `KeplerianProvider` class wraps existing `Satellite` objects to implement this interface, ensuring zero behavioral change when using the default provider.

#### Acceptance Criteria

- [ ] `TrajectoryProvider` ABC defined with all required methods
- [ ] `TrajectoryState` dataclass for state vectors
- [ ] `KeplerianProvider` wraps existing satellite functionality
- [ ] Simulation works identically with `KeplerianProvider` as with direct satellite access
- [ ] All existing tests pass without modification

---

### Step 2: SPICE Provider Implementation

**Goal**: Implement a trajectory provider that reads satellite positions from SPICE ephemeris kernels.

**Files to Create**:
- `simulation/spice_provider.py`

**Dependencies**:
- `spiceypy>=5.0.0` (optional)

#### Detailed Design

The SPICE provider supports multiple kernel types:

| Kernel Type | Extension | Purpose |
|-------------|-----------|---------|
| LSK | `.tls` | Leap seconds |
| SPK | `.bsp` | Spacecraft/planetary ephemerides |
| FK | `.tf` | Frame definitions |
| PCK | `.tpc` | Planetary constants |

**SpiceKernelSet** dataclass groups required kernels:

```python
@dataclass
class SpiceKernelSet:
    leapseconds: Path          # Required
    planetary: List[Path]      # Optional (for Earth position)
    spacecraft: List[Path]     # Required (satellite ephemerides)
    frame: Optional[Path]      # Optional
    planetary_constants: Optional[Path]  # Optional
```

**SpiceProvider** features:
- Automatic time bounds detection from kernel coverage
- Efficient caching of computed positions
- Thread-safe kernel management
- Clean unload on destruction

**SpiceDatasetLoader** utility class provides factory methods:
- `from_horizons_export()` - Load NASA HORIZONS SPK exports
- `from_constellation_definition()` - Load from JSON config file
- `from_tle_file()` - Convert TLE to SPICE (future)

#### Configuration File Format

```json
{
  "name": "MyConstellation",
  "epoch": "2025-01-01T00:00:00Z",
  "leapseconds": "naif0012.tls",
  "planetary": ["de440.bsp"],
  "spacecraft_kernels": ["constellation_v1.bsp"],
  "satellites": {
    "SAT-001": -100001,
    "SAT-002": -100002,
    "SAT-003": -100003
  },
  "reference_frame": "J2000",
  "observer": "EARTH"
}
```

#### Graceful Degradation

```python
try:
    import spiceypy as spice
    SPICE_AVAILABLE = True
except ImportError:
    SPICE_AVAILABLE = False
    
class SpiceProvider(TrajectoryProvider):
    def __init__(self, ...):
        if not SPICE_AVAILABLE:
            raise ImportError(
                "SpiceyPy not installed. Install with: pip install spiceypy\n"
                "Or use --trajectory-provider=keplerian (default)"
            )
```

#### Acceptance Criteria

- [ ] `SpiceProvider` implements `TrajectoryProvider` interface
- [ ] Kernels loaded/unloaded correctly
- [ ] Time bounds computed from kernel coverage
- [ ] Positions match expected values for known ephemerides
- [ ] Clear error message when SpiceyPy not installed
- [ ] Simulation runs correctly with SPICE provider
- [ ] Memory properly cleaned up on provider destruction

---

### Step 3: SPK Export Utility

**Goal**: Enable exporting SatUpdate constellation definitions to SPICE SPK format for interoperability with other tools.

**Files to Create**:
- `tools/generate_spk.py`
- `tools/__init__.py`

#### Detailed Design

Since SpiceyPy has limited SPK writing support, we provide two export paths:

**Path 1: Direct Export (Limited)**
- Uses `spice.spkw09` for Type 9 (Lagrange interpolation) segments
- Suitable for short-duration, high-accuracy needs

**Path 2: mkspk Export (Recommended)**
- Exports state vectors and setup files compatible with NAIF's `mkspk` tool
- More flexible and produces standard-compliant SPK files

**SPKGenerator** class:

```python
class SPKGenerator:
    def add_satellite_trajectory(satellite_id, naif_id, states, ...)
    def add_from_simulation(simulation, start_time, duration, ...)
    def export_for_mkspk(output_dir)  # Recommended
    def write_direct(output_file)      # Limited support
```

**Output Structure for mkspk**:

```
output_dir/
├── SAT-001_states.txt      # State vectors
├── SAT-001_setup.txt       # mkspk configuration
├── SAT-002_states.txt
├── SAT-002_setup.txt
└── generate_all.sh         # Script to run mkspk for all satellites
```

#### Command-Line Integration

```bash
# Export current constellation to SPK
python main.py --type walker_delta --planes 4 --sats-per-plane 6 \
    --export-spk ./spk_output --export-duration 24 --export-step 60

# This creates mkspk-compatible files in ./spk_output/
```

#### Acceptance Criteria

- [ ] State vectors exported in correct format for mkspk
- [ ] Setup files contain all required mkspk parameters
- [ ] Exported data round-trips correctly (export → mkspk → load → compare)
- [ ] Works with all constellation types (Walker-Delta, Walker-Star, Random)
- [ ] Duration and step size configurable
- [ ] NAIF IDs assigned correctly and documented

---

### Step 4: NetworkBackend Interface

**Goal**: Create an abstract interface for network simulation that allows plugging in different network models.

**Files to Create**:
- `simulation/network_backend.py`

**Changes to Existing Files**:
- `simulation/simulation.py` - Refactor agent protocol to use network backend

#### Detailed Design

**Core Abstractions**:

```python
@dataclass
class PacketTransfer:
    source_id: str
    destination_id: str
    packet_id: int
    timestamp: float
    success: bool
    latency_ms: Optional[float] = None
    dropped_reason: Optional[str] = None

class NetworkBackend(ABC):
    @abstractmethod
    def initialize(self, topology: Dict) -> None
    
    @abstractmethod
    def update_topology(self, active_links: Set[Tuple[str, str]]) -> None
    
    @abstractmethod
    def send_packet(self, source, destination, packet_id, size) -> bool
    
    @abstractmethod
    def step(self, timestep: float) -> List[PacketTransfer]
    
    @abstractmethod
    def get_statistics(self) -> NetworkStatistics
```

**NativeNetworkBackend** preserves current behavior:
- Instant packet delivery (zero latency)
- Perfect reliability (no drops)
- Unlimited bandwidth
- Topology-aware (respects line-of-sight and range)

**Key Design Decision**: The native backend must produce **identical results** to the current implementation when used as default. This is verified by running the entire existing test suite.

#### Integration with Agent Protocol

The existing 4-phase agent protocol remains unchanged. The network backend is used only for the actual packet transfer step:

```python
# Phase 4: Transfer (modified)
for transfer in self._network_backend.step(timestep):
    if transfer.success:
        receiver_agent.receive_packet(transfer.packet_id)
```

#### Acceptance Criteria

- [ ] `NetworkBackend` ABC with complete interface
- [ ] `NativeNetworkBackend` produces identical results to current implementation
- [ ] `NetworkStatistics` captures relevant metrics
- [ ] Topology updates handled correctly
- [ ] All existing tests pass with `NativeNetworkBackend`
- [ ] Agent protocol works correctly with backend abstraction

---

### Step 5: NS-3 Backend - File Mode

**Goal**: Implement NS-3 integration using file-based communication for batch processing.

**Files to Create**:
- `simulation/ns3_backend.py`
- `ns3_scenarios/satellite_update_scenario.cc`
- `ns3_scenarios/CMakeLists.txt`

#### Detailed Design

**Communication Protocol**:

1. Python writes JSON input file with topology and pending packets
2. Python invokes NS-3 scenario via subprocess
3. NS-3 runs simulation and writes JSON output file
4. Python reads results and updates agent state

**Input Format**:

```json
{
  "command": "step",
  "timestep": 60.0,
  "topology": {
    "nodes": [
      {"id": "SAT-001", "type": "satellite", "position": [7000, 0, 0]},
      {"id": "BASE-1", "type": "ground", "position": [6371, 0, 0]}
    ],
    "links": [["SAT-001", "SAT-002"], ["SAT-002", "SAT-003"]]
  },
  "sends": [
    {"source": "BASE-1", "destination": "SAT-001", "packet_id": 0, "size": 1024},
    {"source": "SAT-001", "destination": "SAT-002", "packet_id": 1, "size": 1024}
  ],
  "config": {
    "data_rate": "10Mbps",
    "propagation_model": "constant_speed",
    "error_model": "none"
  }
}
```

**Output Format**:

```json
{
  "status": "success",
  "simulation_time": 60.0,
  "transfers": [
    {
      "source": "BASE-1",
      "destination": "SAT-001", 
      "packet_id": 0,
      "timestamp": 0.023,
      "success": true,
      "latency_ms": 23.4
    },
    {
      "source": "SAT-001",
      "destination": "SAT-002",
      "packet_id": 1,
      "timestamp": 0.045,
      "success": false,
      "dropped_reason": "queue_overflow"
    }
  ],
  "statistics": {
    "total_packets_sent": 2,
    "total_packets_received": 1,
    "average_latency_ms": 23.4,
    "link_utilization": {
      "SAT-001-SAT-002": 0.85
    }
  }
}
```

**NS-3 Scenario Script** (C++):

The scenario script handles:
- Node creation and positioning
- Point-to-point link setup with satellite characteristics
- Internet stack installation
- Packet scheduling and transmission
- Result collection via trace callbacks

#### NS-3 Installation Detection

```python
class NS3Backend:
    @staticmethod
    def check_ns3_installation(ns3_path: Path) -> bool:
        """Verify NS-3 is installed and configured."""
        ns3_exe = ns3_path / "ns3"
        if not ns3_exe.exists():
            return False
        # Try running ns3 --version
        result = subprocess.run([str(ns3_exe), "--version"], 
                                capture_output=True)
        return result.returncode == 0
```

#### Acceptance Criteria

- [ ] JSON protocol fully specified and documented
- [ ] NS-3 scenario compiles and runs standalone
- [ ] File-based communication works reliably
- [ ] Temporary files cleaned up properly
- [ ] Error handling for NS-3 failures
- [ ] Latency values realistic for satellite links
- [ ] Works without NS-3 installed (clear error message)

---

### Step 6: NS-3 Backend - Socket Mode

**Goal**: Enable real-time communication with NS-3 for interactive simulations.

**Files to Modify**:
- `simulation/ns3_backend.py` - Add socket mode
- `ns3_scenarios/satellite_update_scenario.cc` - Add socket server

#### Detailed Design

**Why Socket Mode?**

| Aspect | File Mode | Socket Mode |
|--------|-----------|-------------|
| Latency | High (process spawn) | Low (persistent connection) |
| Use Case | Batch processing | Interactive/visualization |
| Complexity | Simple | Moderate |
| State | Stateless | Stateful |

**Protocol**:
- TCP socket with JSON-line protocol (newline-delimited JSON)
- Persistent NS-3 process runs as server
- Python client connects and sends commands
- Asynchronous response handling via background thread

**Connection Lifecycle**:

```
Python                          NS-3 Server
   |                                 |
   |------ connect ---------------->|
   |<----- ack --------------------|
   |                                 |
   |------ initialize topology ---->|
   |<----- ack --------------------|
   |                                 |
   |------ step (sends=[...]) ----->|
   |<----- transfers=[...] --------|
   |                                 |
   |------ step (sends=[...]) ----->|
   |<----- transfers=[...] --------|
   |         ...                     |
   |                                 |
   |------ shutdown --------------->|
   |<----- ack --------------------|
   |------ close ----------------->|
```

**Thread Safety**:

```python
class NS3Backend:
    def __init__(self, ...):
        self._socket: Optional[socket.socket] = None
        self._response_queue: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._receiver_thread: Optional[threading.Thread] = None
```

**Timeout Handling**:

```python
def step(self, timestep: float) -> List[PacketTransfer]:
    # Send command
    self._send_command({"command": "step", ...})
    
    # Wait for response with timeout
    try:
        response = self._response_queue.get(timeout=30.0)
    except queue.Empty:
        raise TimeoutError("NS-3 did not respond within 30 seconds")
    
    return self._parse_transfers(response)
```

#### Acceptance Criteria

- [ ] Socket connection established reliably
- [ ] Background receiver thread handles responses
- [ ] Proper cleanup on disconnect/error
- [ ] Timeout handling prevents hangs
- [ ] Thread-safe command sending
- [ ] Reconnection logic for dropped connections
- [ ] Performance improvement over file mode demonstrated

---

### Step 7: NS-3 Backend - Python Bindings Mode

**Goal**: Direct integration with NS-3 via Python bindings for maximum performance and flexibility.

**Files to Modify**:
- `simulation/ns3_backend.py` - Add bindings mode

**Dependencies**:
- NS-3 compiled with Python bindings
- Optionally: SNS3 (Satellite Network Simulator 3)

#### Detailed Design

**Why Bindings Mode?**

| Aspect | File/Socket Mode | Bindings Mode |
|--------|------------------|---------------|
| Performance | Moderate | Best |
| Flexibility | Limited to protocol | Full NS-3 API |
| Setup Complexity | Low | High |
| Debugging | Separate processes | Single process |

**NS-3 Python Bindings Setup**:

```python
def _init_bindings_mode(self):
    try:
        # Core NS-3 modules
        import ns.core
        import ns.network
        import ns.internet
        import ns.point_to_point
        import ns.applications
        import ns.mobility
        
        # Optional: SNS3 satellite module
        try:
            import ns.satellite
            self._has_sns3 = True
        except ImportError:
            self._has_sns3 = False
        
        self._ns = {
            "core": ns.core,
            "network": ns.network,
            # ...
        }
        
    except ImportError as e:
        raise ImportError(
            f"NS-3 Python bindings not available: {e}\n"
            "Build NS-3 with: ./ns3 configure --enable-python-bindings\n"
            "Or use --ns3-mode=file or --ns3-mode=socket"
        )
```

**Scenario Setup**:

```python
def _setup_ns3_scenario(self):
    ns = self._ns
    
    # Create node container
    self._nodes = ns["network"].NodeContainer()
    self._nodes.Create(len(self.topology["nodes"]))
    
    # Setup mobility (positions)
    mobility = ns["mobility"].MobilityHelper()
    positions = ns["mobility"].ListPositionAllocator()
    for node in self.topology["nodes"]:
        pos = node["position"]
        positions.Add(ns["core"].Vector(pos[0], pos[1], pos[2]))
    mobility.SetPositionAllocator(positions)
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel")
    mobility.Install(self._nodes)
    
    # Install internet stack
    internet = ns["internet"].InternetStackHelper()
    internet.Install(self._nodes)
    
    # Setup channels and devices
    # ...
```

**Event-Driven Integration**:

```python
def step(self, timestep: float) -> List[PacketTransfer]:
    ns = self._ns
    
    # Schedule pending sends
    for send in self._pending_sends:
        # Create application to send packet
        self._schedule_packet_send(send)
    self._pending_sends.clear()
    
    # Clear transfer results
    self._completed_transfers.clear()
    
    # Run NS-3 for timestep duration
    ns["core"].Simulator.Stop(ns["core"].Seconds(timestep))
    ns["core"].Simulator.Run()
    
    # Results collected via trace callbacks
    return list(self._completed_transfers)
```

**Trace Callbacks**:

```python
def _setup_trace_callbacks(self):
    # Callback when packet received
    def rx_callback(context, packet, address):
        # Extract packet info and record transfer
        self._completed_transfers.append(PacketTransfer(...))
    
    # Connect to trace sources
    # ...
```

#### SNS3 Integration (Optional)

If SNS3 is available, use its satellite-specific models:

```python
if self._has_sns3:
    # Use SNS3's satellite channel model
    channel = ns["satellite"].SatelliteChannel()
    channel.SetPropagationDelayModel(
        ns["satellite"].SatellitePropagationDelayModel()
    )
else:
    # Fallback to point-to-point with manual delay calculation
    channel = ns["point_to_point"].PointToPointChannel()
```

#### Acceptance Criteria

- [ ] NS-3 Python bindings detected correctly
- [ ] Nodes created with correct positions
- [ ] Internet stack installed properly
- [ ] Packets sent and received correctly
- [ ] Trace callbacks capture all events
- [ ] Results match file/socket mode for same scenario
- [ ] SNS3 used when available
- [ ] Clean fallback when bindings unavailable

---

## Testing Strategy

### Test Categories

1. **Unit Tests**: Test individual components in isolation
2. **Integration Tests**: Test component interactions
3. **Regression Tests**: Ensure existing functionality preserved
4. **Mock Tests**: Test NS-3/SPICE integration without dependencies
5. **End-to-End Tests**: Full simulation with optional backends

### Test File Structure

```
tests/
├── test_trajectory_provider.py     # Step 1
├── test_spice_provider.py          # Step 2
├── test_spk_generator.py           # Step 3
├── test_network_backend.py         # Step 4
├── test_ns3_file_backend.py        # Step 5
├── test_ns3_socket_backend.py      # Step 6
├── test_ns3_bindings_backend.py    # Step 7
├── test_backward_compatibility.py  # Regression tests
├── test_integration.py             # End-to-end tests
├── conftest.py                     # Shared fixtures
└── mocks/
    ├── mock_spice.py
    └── mock_ns3.py
```

### CI/CD Considerations

```yaml
# .github/workflows/test.yml
jobs:
  test-core:
    # Runs on every PR - no optional dependencies
    steps:
      - run: pytest tests/ -m "not requires_spice and not requires_ns3"
  
  test-spice:
    # Runs on main branch - requires spiceypy
    steps:
      - run: pip install spiceypy
      - run: pytest tests/ -m "requires_spice"
  
  test-ns3:
    # Manual trigger - requires NS-3 installation
    steps:
      - run: pytest tests/ -m "requires_ns3"
```

---

## Migration Guide

### For Existing Users

**No action required.** All existing functionality works identically. New features are opt-in.

### Enabling SPICE Support

1. Install SpiceyPy: `pip install spiceypy`
2. Obtain SPICE kernels (leapseconds + spacecraft ephemeris)
3. Create configuration file or use `SpiceDatasetLoader`
4. Run with `--trajectory-provider=spice --spice-config=config.json`

### Enabling NS-3 Support

1. Install NS-3 with satellite module
2. Build the SatUpdate NS-3 scenario
3. Run with `--network-backend=ns3 --ns3-mode=file --ns3-path=/path/to/ns3`

---

## Timeline Estimate

| Step | Description | Estimated Effort |
|------|-------------|------------------|
| 1 | TrajectoryProvider Interface | 2-3 days |
| 2 | SPICE Provider | 3-4 days |
| 3 | SPK Export | 2-3 days |
| 4 | NetworkBackend Interface | 2-3 days |
| 5 | NS-3 File Mode | 4-5 days |
| 6 | NS-3 Socket Mode | 2-3 days |
| 7 | NS-3 Bindings Mode | 3-4 days |
| - | Testing & Documentation | 3-4 days |
| **Total** | | **21-29 days** |

Steps can be parallelized: Steps 1-3 (trajectory) are independent of Steps 4-7 (network).

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| SpiceyPy API changes | Medium | Pin version, add compatibility layer |
| NS-3 version incompatibility | High | Support multiple versions, document requirements |
| Performance regression | Medium | Benchmark suite, profiling |
| Complex debugging with NS-3 | Medium | Comprehensive logging, file mode for debugging |
| SNS3 availability | Low | Core features work without SNS3 |

---

## References

- [SPICE Toolkit Documentation](https://naif.jpl.nasa.gov/naif/documentation.html)
- [SpiceyPy Documentation](https://spiceypy.readthedocs.io/)
- [NS-3 Manual](https://www.nsnam.org/docs/manual/html/)
- [SNS3 Documentation](https://sns3.io/documentation/)
- [NASA HORIZONS System](https://ssd.jpl.nasa.gov/horizons/)
