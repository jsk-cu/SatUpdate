# NS-3 and SPICE Integration Implementation Plan

## Executive Summary

This document outlines the implementation plan for adding optional NS-3 network simulation and SPICE ephemeris support to the SatUpdate satellite constellation simulator. These features are designed as **opt-in extensions** that preserve all existing functionality while enabling more sophisticated simulation capabilities.

### Current Status

| Step | Component | Status | Tests |
|------|-----------|--------|-------|
| 1 | TrajectoryProvider Interface | ✅ **COMPLETE** | 23 |
| 2 | SPICE Provider | ✅ **COMPLETE** | 31 |
| 3 | SPK Export Utility | ✅ **COMPLETE** | 41 |
| 4 | NetworkBackend Interface | ✅ **COMPLETE** | 41 |
| 5 | NS-3 File Mode | ✅ **COMPLETE** | 46 |
| 6 | NS-3 Socket Mode | ✅ **COMPLETE** | 46 |
| 7 | NS-3 Bindings Mode | ✅ **COMPLETE** | 37 |
| **Total** | | **7/7 Complete** | **265 tests** |

---

## Design Principles

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
│      ┌──────┴──────┐             ┌──────┴──────────────┐                │
│      │             │             │          │          │                │
│  ┌───▼────┐   ┌────▼────┐   ┌────▼────┐ ┌──▼───┐ ┌────▼────┐           │
│  │Keplerian│  │  SPICE  │   │ Native  │ │Delayed│ │  NS-3   │           │
│  │Provider │  │ Provider│   │ Backend │ │Backend│ │ Backend │           │
│  │(default)│  │(opt-in) │   │(default)│ │(test) │ │(opt-in) │           │
│  └─────────┘  └─────────┘   └─────────┘ └──────┘ └─────────┘           │
│       ✅           ✅            ✅         ✅         ✅                │
│                                                                          │
│                              NS-3 Backend Modes                          │
│                    ┌────────────┬────────────┬────────────┐             │
│                    │    File    │   Socket   │  Bindings  │             │
│                    │    Mode    │    Mode    │    Mode    │             │
│                    │     ✅     │     ✅     │     ✅     │             │
│                    └────────────┴────────────┴────────────┘             │
└─────────────────────────────────────────────────────────────────────────┘
```

### Command-Line Arguments

| Argument | Description | Default | Status |
|----------|-------------|---------|--------|
| `--trajectory-provider` | Trajectory source: `keplerian`, `spice` | `keplerian` | ✅ Ready |
| `--spice-kernels-dir` | Directory containing SPICE kernels | None | ✅ Ready |
| `--spice-config` | Path to SPICE constellation config JSON | None | ✅ Ready |
| `--network-backend` | Network simulator: `native`, `ns3` | `native` | ✅ Ready |
| `--ns3-mode` | NS-3 communication: `file`, `socket`, `bindings` | `file` | ✅ Ready |
| `--ns3-path` | Path to NS-3 installation | `/usr/local/ns3` | ✅ Ready |
| `--ns3-host` | NS-3 server host (socket mode) | `localhost` | ✅ Ready |
| `--ns3-port` | NS-3 server port (socket mode) | `5555` | ✅ Ready |
| `--export-spk` | Export constellation to SPK format | None | ✅ Ready |

---

## Completed Implementation Steps

### Step 1: TrajectoryProvider Interface ✅ COMPLETE

**Goal**: Create an abstract interface for satellite position computation that decouples trajectory calculation from the core simulation.

**Tests**: 23 passed

---

### Step 2: SPICE Provider ✅ COMPLETE

**Goal**: Implement a TrajectoryProvider that uses NASA's SPICE toolkit for high-fidelity ephemeris.

**Tests**: 31 passed

---

### Step 3: SPK Export Utility ✅ COMPLETE

**Goal**: Enable export of simulation state to SPICE SPK format for interoperability with external tools.

**Tests**: 41 passed

---

### Step 4: NetworkBackend Interface ✅ COMPLETE

**Goal**: Create an abstract interface for network simulation that allows plugging in different network models.

**Tests**: 41 passed

---

### Step 5: NS-3 Backend - File Mode ✅ COMPLETE

**Goal**: Implement NS-3 integration using file-based communication (JSON input/output).

**Tests**: 46 passed

**Key Components**:
- `NS3Config` - Network configuration parameters
- `NS3Mode` - Communication mode enumeration
- `NS3ErrorModel` - Error model types
- `NS3PropagationModel` - Propagation models
- `NS3Node` - Node specification for topology
- `NS3SendCommand` - Packet send command
- `NS3Backend` - Full NetworkBackend implementation

---

### Step 6: NS-3 Backend - Socket Mode ✅ COMPLETE

**Goal**: Enable real-time communication with NS-3 for interactive simulations.

**Tests**: 46 passed

**Key Components**:
- `NS3SocketClient` - Thread-safe TCP socket client
- `SocketConnectionError` - Connection failure exception
- `SocketTimeoutError` - Timeout exception

**Features Implemented**:
- TCP socket with JSON-line protocol (newline-delimited JSON)
- Persistent NS-3 process runs as server
- Background receiver thread for async responses
- Automatic reconnection on disconnect
- Thread-safe command sending
- Configurable timeouts

**Protocol**:
```
Python                          NS-3 Server
   |                                 |
   |------ connect ---------------->|
   |<----- ack --------------------|
   |------ step (sends=[...]) ----->|
   |<----- transfers=[...] --------|
   |------ shutdown --------------->|
   |------ close ----------------->|
```

**Performance Characteristics**:
| Aspect | File Mode | Socket Mode |
|--------|-----------|-------------|
| Latency | High (process spawn) | Low (persistent connection) |
| Use Case | Batch processing | Interactive/visualization |
| Complexity | Simple | Moderate |
| State | Stateless | Stateful |

**Acceptance Criteria**: ✅ All met
- [x] Socket connection established reliably
- [x] Background receiver thread handles responses
- [x] Proper cleanup on disconnect/error
- [x] Timeout handling prevents hangs
- [x] Thread-safe command sending
- [x] Reconnection logic for dropped connections
- [x] Graceful fallback to mock mode

---

### Step 7: NS-3 Backend - Python Bindings Mode ✅ COMPLETE

**Goal**: Direct integration with NS-3 via Python bindings for maximum performance.

**Tests**: 37 passed (2 skipped without NS-3 bindings)

**Key Components**:
- `NS3BindingsWrapper` - Wrapper for NS-3 Python bindings
- `NS3BindingsError` - Exception for bindings errors
- `check_ns3_bindings()` - Detect NS-3 Python bindings
- `check_sns3_bindings()` - Detect SNS3 satellite extensions

**Features Implemented**:
- Dynamic NS-3 module loading (ns.core, ns.network, ns.internet, etc.)
- Node container creation and management
- Internet stack installation
- Point-to-point link configuration
- Position-based mobility model (ConstantPositionMobilityModel)
- IP address assignment
- Packet tracking (pending and completed)
- SNS3 satellite extensions support (when available)

**Performance Characteristics**:
| Aspect | File/Socket Mode | Bindings Mode |
|--------|-----------------|---------------|
| Performance | Process overhead | Native speed |
| Flexibility | Fixed protocol | Full NS-3 API |
| Debugging | Separate process | Integrated |
| Dependencies | NS-3 installation | NS-3 + Python bindings |

**Acceptance Criteria**: ✅ All met
- [x] NS-3 Python bindings detected correctly
- [x] Nodes created with correct positions
- [x] Internet stack installed properly
- [x] Packets sent and received correctly
- [x] Results match file/socket mode for same scenario
- [x] SNS3 used when available
- [x] Clean fallback when bindings unavailable

---

## File Structure Summary

### Implementation Files

```
simulation/
├── __init__.py                 # Updated with all exports
├── trajectory.py               # Step 1: TrajectoryProvider interface
├── spice_provider.py           # Step 2: SPICE Provider
├── network_backend.py          # Step 4: NetworkBackend interface
└── ns3_backend.py              # Steps 5-7: NS-3 Backend (all modes)

tools/
├── __init__.py                 # Step 3: Exports
└── generate_spk.py             # Step 3: SPK Export Utility

ns3_scenarios/
├── satellite-update-scenario.cc  # Step 5: NS-3 C++ scenario
├── CMakeLists.txt                # Step 5: Build config
└── README.md                     # Step 5: Documentation
```

### Test Files

```
ns3_spice_tests/
├── conftest.py                   # Shared fixtures and markers
├── test_trajectory_provider.py   # Step 1 tests (23)
├── test_spice_provider.py        # Step 2 tests (31)
├── test_spk_generator.py         # Step 3 tests (41)
├── test_network_backend.py       # Step 4 tests (41)
├── test_ns3_file_backend.py      # Step 5 tests (46)
├── test_ns3_socket_mode.py       # Step 6 tests (46)
└── test_ns3_bindings_mode.py     # Step 7 tests (37)
```

---

## Exports Available

```python
from simulation import (
    # Step 1: TrajectoryProvider
    TrajectoryProvider, TrajectoryState, KeplerianProvider,
    create_keplerian_provider,
    
    # Step 2: SPICE Provider
    SpiceProvider, SpiceKernelSet, SpiceConstellationConfig,
    SpiceDatasetLoader, is_spice_available, SPICE_AVAILABLE,
    
    # Step 4: NetworkBackend
    NetworkBackend, NativeNetworkBackend, DelayedNetworkBackend,
    PacketTransfer, NetworkStatistics, DropReason, PendingTransfer,
    create_native_backend, create_delayed_backend,
    
    # Steps 5-7: NS-3 Backend
    NS3Backend, NS3Config, NS3Mode, NS3ErrorModel, NS3PropagationModel,
    NS3Node, NS3SendCommand,
    NS3SocketClient, SocketConnectionError, SocketTimeoutError,  # Step 6
    NS3BindingsWrapper, NS3BindingsError,                        # Step 7
    check_ns3_bindings, check_sns3_bindings,                     # Step 7
    create_ns3_backend, check_ns3_available, is_ns3_available,
)

from tools import (
    # Step 3: SPK Export
    SPKGenerator, StateVector, SPKSegment, NAIFIDManager,
    create_spk_from_simulation,
)
```

---

## Testing Strategy

### Test Categories

| Category | Purpose | Count |
|----------|---------|-------|
| Unit Tests | Individual components | ~200 |
| Integration Tests | Component interactions | ~40 |
| Mock Tests | Test without dependencies | ~60 |
| Regression Tests | Backward compatibility | ~15 |

### Conditional Skipping

Tests requiring optional dependencies are automatically skipped:
- `@pytest.mark.requires_spice` - Skipped without SpiceyPy
- `@pytest.mark.requires_ns3` - Skipped without NS-3 installation
- `@pytest.mark.requires_ns3_bindings` - Skipped without NS-3 Python bindings

### Running Tests

```bash
# Run all tests (skips unavailable dependencies)
pytest ns3_spice_tests/ -v

# Run only core tests (no optional dependencies)
pytest ns3_spice_tests/ -v -m "not requires_spice and not requires_ns3"

# Run with SPICE tests (requires: pip install spiceypy)
pytest ns3_spice_tests/ -v -m "requires_spice"

# Run with NS-3 tests (requires NS-3 installation)
pytest ns3_spice_tests/ -v -m "requires_ns3"

# Run specific step tests
pytest ns3_spice_tests/test_ns3_socket_mode.py -v      # Step 6
pytest ns3_spice_tests/test_ns3_bindings_mode.py -v    # Step 7
```

---

## Usage Examples

### NS-3 Backend Modes

```python
from simulation import NS3Backend, NS3Config, create_ns3_backend

# File Mode (default) - Best for batch processing
backend = create_ns3_backend(mode="file", ns3_path="/opt/ns3")

# Socket Mode - Best for interactive/real-time simulation
backend = create_ns3_backend(
    mode="socket",
    host="localhost",
    port=5555
)

# Bindings Mode - Best performance (requires NS-3 Python bindings)
backend = create_ns3_backend(mode="bindings")

# Mock Mode - For testing without NS-3
backend = create_ns3_backend(mode="mock")

# Usage pattern (same for all modes)
backend.initialize(topology)
backend.send_packet("SAT-001", "SAT-002", packet_id=1)
transfers = backend.step(60.0)
stats = backend.get_statistics()
backend.shutdown()
```

### Mode Selection Guide

| Use Case | Recommended Mode |
|----------|------------------|
| Unit testing | `mock` |
| Batch simulations | `file` |
| Interactive visualization | `socket` |
| High-performance research | `bindings` |
| CI/CD pipelines | `mock` or `file` |

---

## Migration Guide

### For Existing Users

**No action required.** All existing functionality works identically. New features are opt-in.

### Enabling SPICE Support

1. Install SpiceyPy: `pip install spiceypy`
2. Obtain SPICE kernels (leapseconds + spacecraft ephemeris)
3. Use in code:
   ```python
   from simulation import SpiceProvider, SpiceKernelSet
   
   kernels = SpiceKernelSet(
       leapseconds=Path("naif0012.tls"),
       spacecraft=[Path("constellation.bsp")]
   )
   provider = SpiceProvider(kernels, config)
   ```

### Enabling NS-3 Support

1. Install NS-3 (version 3.36+)
2. Copy scenario to scratch directory: `cp ns3_scenarios/*.cc /path/to/ns3/scratch/`
3. Build: `cd /path/to/ns3 && ./ns3 build`
4. Use in code:
   ```python
   from simulation import NS3Backend
   
   backend = NS3Backend(mode="file", ns3_path="/path/to/ns3")
   ```

### Enabling NS-3 Python Bindings

1. Build NS-3 with Python bindings:
   ```bash
   cd /path/to/ns3
   ./ns3 configure --enable-python-bindings
   ./ns3 build
   ```
2. Use bindings mode:
   ```python
   from simulation import NS3Backend, check_ns3_bindings
   
   if check_ns3_bindings():
       backend = NS3Backend(mode="bindings")
   else:
       backend = NS3Backend(mode="file")  # Fallback
   ```

---

## Timeline Summary

| Step | Description | Estimated | Status |
|------|-------------|-----------|--------|
| 1 | TrajectoryProvider Interface | 2-3 days | ✅ Complete |
| 2 | SPICE Provider | 3-4 days | ✅ Complete |
| 3 | SPK Export | 2-3 days | ✅ Complete |
| 4 | NetworkBackend Interface | 2-3 days | ✅ Complete |
| 5 | NS-3 File Mode | 4-5 days | ✅ Complete |
| 6 | NS-3 Socket Mode | 2-3 days | ✅ Complete |
| 7 | NS-3 Bindings Mode | 3-4 days | ✅ Complete |

**Total Implementation**: Complete ✅

---

## Risk Assessment

| Risk | Impact | Mitigation | Status |
|------|--------|------------|--------|
| SpiceyPy API changes | Medium | Pin version, graceful degradation | ✅ Mitigated |
| NS-3 version incompatibility | High | `show version` detection, multiple paths | ✅ Mitigated |
| Performance regression | Medium | Benchmark suite, native backend default | ✅ Mitigated |
| Complex debugging with NS-3 | Medium | Mock mode, comprehensive logging | ✅ Mitigated |
| SNS3 availability | Low | Core features work without SNS3 | ✅ Mitigated |
| Socket connection failures | Medium | Auto-reconnection, fallback to mock | ✅ Mitigated |
| Bindings not available | Low | Graceful fallback to file/socket mode | ✅ Mitigated |

---

## References

- [SPICE Toolkit Documentation](https://naif.jpl.nasa.gov/naif/documentation.html)
- [SpiceyPy Documentation](https://spiceypy.readthedocs.io/)
- [NS-3 Manual](https://www.nsnam.org/docs/manual/html/)
- [NS-3 Python Bindings](https://www.nsnam.org/docs/manual/html/python.html)
- [SNS3 Documentation](https://sns3.io/documentation/)
- [NASA HORIZONS System](https://ssd.jpl.nasa.gov/horizons/)