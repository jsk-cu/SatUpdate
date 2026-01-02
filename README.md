# SatUpdate - Satellite Constellation Simulator

A high-fidelity satellite constellation simulator with support for multi-packet update dissemination algorithms, optional NS-3 network simulation, and SPICE ephemeris integration.

## Overview

SatUpdate simulates homogeneous satellite constellations with the goal of experimenting with algorithms for disseminating multi-packet updates to the entire constellation. The simulator provides:

- **Trajectory Simulation**: Keplerian orbit propagation or NASA SPICE ephemeris
- **Network Simulation**: Native instant delivery, delayed propagation, or NS-3 high-fidelity simulation
- **Pluggable Architecture**: Abstract interfaces allow swapping implementations without code changes
- **Graceful Degradation**: Falls back to built-in implementations when optional dependencies are unavailable

## Installation

### Basic Installation

```bash
git clone <repository>
cd satupdate
pip install -r requirements.txt
```

### Optional Dependencies

```bash
# For SPICE ephemeris support
pip install spiceypy

# For NS-3 network simulation
# See NS-3 Installation section below
```

## Quick Start

```python
from simulation import (
    NS3Backend,
    create_ns3_backend,
    NativeNetworkBackend,
)

# Simple network backend (instant delivery)
backend = NativeNetworkBackend()
backend.initialize({
    "nodes": [
        {"id": "SAT-001", "type": "satellite", "position": [7000000, 0, 0]},
        {"id": "SAT-002", "type": "satellite", "position": [0, 7000000, 0]},
    ],
    "links": [("SAT-001", "SAT-002")]
})

# Send packets
backend.send_packet("SAT-001", "SAT-002", packet_id=1)

# Advance simulation
transfers = backend.step(60.0)  # 60 second timestep

# Check results
for transfer in transfers:
    print(f"{transfer.source_id} -> {transfer.destination_id}: {'OK' if transfer.success else 'FAILED'}")
```

## Network Backend Options

SatUpdate provides multiple network simulation backends:

| Backend | Description | Use Case |
|---------|-------------|----------|
| `NativeNetworkBackend` | Instant, perfect delivery | Fast prototyping |
| `DelayedNetworkBackend` | Propagation delay simulation | Latency testing |
| `NS3Backend (mock)` | Mock NS-3 with realistic latency | Testing without NS-3 |
| `NS3Backend (file)` | Full NS-3 via file I/O | Batch processing |
| `NS3Backend (socket)` | Full NS-3 via TCP socket | Interactive simulation |
| `NS3Backend (bindings)` | Direct NS-3 Python API | Maximum performance |

### Using NS-3 Backend

```python
from simulation import NS3Backend, NS3Config, check_ns3_bindings

# Auto-detect best available mode
if check_ns3_bindings():
    backend = NS3Backend(mode="bindings")
elif is_ns3_available():
    backend = NS3Backend(mode="file", ns3_path="/opt/ns3")
else:
    backend = NS3Backend(mode="mock")

# Configure network parameters
config = NS3Config(
    data_rate="10Mbps",
    propagation_model="constant_speed",
    error_model="none",
)

backend = NS3Backend(mode="mock", config=config)
```

## NS-3 Installation

### Ubuntu/Debian

```bash
# Install dependencies
sudo apt install g++ python3 cmake ninja-build git

# Clone NS-3
git clone https://gitlab.com/nsnam/ns-3-dev.git /opt/ns3
cd /opt/ns3

# Configure and build
./ns3 configure --enable-examples --enable-tests
./ns3 build

# Optional: Enable Python bindings
./ns3 configure --enable-python-bindings
./ns3 build
```

### Using NS-3 Socket Mode

Start the NS-3 server (requires the satellite-update-scenario):

```bash
cd /opt/ns3
cp /path/to/satupdate/ns3_scenarios/satellite-update-scenario.cc scratch/
./ns3 build
./ns3 run "satellite-update-scenario --server --port=5555"
```

Then connect from Python:

```python
backend = NS3Backend(mode="socket", host="localhost", port=5555)
```

## SPICE Ephemeris Support

For high-fidelity orbital mechanics using NASA's SPICE toolkit:

```python
from simulation import SpiceProvider, SpiceKernelSet

# Load SPICE kernels
kernels = SpiceKernelSet(
    leapseconds=Path("naif0012.tls"),
    spacecraft=[Path("constellation.bsp")]
)

# Create provider
provider = SpiceProvider(kernels, config)

# Get satellite state
state = provider.get_state("SAT-001", epoch)
print(f"Position: {state.position}")
print(f"Velocity: {state.velocity}")
```

## Project Structure

```
satupdate/
├── simulation/
│   ├── __init__.py           # Module exports
│   ├── trajectory.py         # TrajectoryProvider interface
│   ├── spice_provider.py     # SPICE ephemeris provider
│   ├── network_backend.py    # NetworkBackend interface
│   └── ns3_backend.py        # NS-3 integration (all modes)
│
├── tools/
│   ├── __init__.py           # Tool exports
│   └── generate_spk.py       # SPK export utility
│
├── ns3_scenarios/
│   ├── satellite-update-scenario.cc  # NS-3 C++ scenario
│   └── CMakeLists.txt               # Build configuration
│
├── ns3_spice_tests/
│   ├── conftest.py                   # Test fixtures
│   ├── test_trajectory_provider.py   # Step 1 tests
│   ├── test_spice_provider.py        # Step 2 tests
│   ├── test_spk_generator.py         # Step 3 tests
│   ├── test_network_backend.py       # Step 4 tests
│   ├── test_ns3_file_backend.py      # Step 5 tests
│   ├── test_ns3_socket_mode.py       # Step 6 tests
│   ├── test_ns3_bindings_mode.py     # Step 7 tests
│   └── ns3_spice_implementation_plan.md
│
└── README.md
```

## Running Tests

```bash
# Run all tests
pytest ns3_spice_tests/ -v

# Run without optional dependencies
pytest ns3_spice_tests/ -v -m "not requires_spice and not requires_ns3"

# Run specific test modules
pytest ns3_spice_tests/test_network_backend.py -v
pytest ns3_spice_tests/test_ns3_socket_mode.py -v

# Run with coverage
pytest ns3_spice_tests/ --cov=simulation --cov-report=html
```

### Test Requirements

All tests must pass for the implementation to be complete:

| Test File | Tests | Description |
|-----------|-------|-------------|
| `test_trajectory_provider.py` | 23 | TrajectoryProvider interface |
| `test_spice_provider.py` | 31 | SPICE ephemeris (skipped without SpiceyPy) |
| `test_spk_generator.py` | 41 | SPK export utility |
| `test_network_backend.py` | 41 | NetworkBackend interface |
| `test_ns3_file_backend.py` | 46 | NS-3 file mode |
| `test_ns3_socket_mode.py` | 46 | NS-3 socket mode |
| `test_ns3_bindings_mode.py` | 37 | NS-3 bindings mode |
| **Total** | **265** | |

## API Reference

### NetworkBackend Interface

```python
class NetworkBackend(ABC):
    def initialize(self, topology: Dict[str, Any]) -> None: ...
    def update_topology(self, active_links: Set[Tuple[str, str]]) -> None: ...
    def send_packet(self, source: str, destination: str, packet_id: int, 
                    size_bytes: int = 1024, metadata: Dict = None) -> bool: ...
    def step(self, timestep: float) -> List[PacketTransfer]: ...
    def get_statistics(self) -> NetworkStatistics: ...
    def shutdown(self) -> None: ...
```

### NS3Backend Modes

```python
from simulation import NS3Mode

NS3Mode.FILE      # JSON file-based communication
NS3Mode.SOCKET    # TCP socket communication
NS3Mode.BINDINGS  # Direct Python bindings
NS3Mode.MOCK      # Mock mode for testing
```

### Detection Functions

```python
from simulation import (
    check_ns3_available,    # Check NS-3 installation
    check_ns3_bindings,     # Check Python bindings
    check_sns3_bindings,    # Check SNS3 extensions
    is_spice_available,     # Check SpiceyPy
)
```

## Configuration

### NS3Config Options

```python
from simulation import NS3Config, NS3ErrorModel, NS3PropagationModel

config = NS3Config(
    data_rate="10Mbps",                              # Link bandwidth
    propagation_model=NS3PropagationModel.CONSTANT_SPEED,  # Delay model
    propagation_speed=299792458.0,                   # m/s (speed of light)
    error_model=NS3ErrorModel.NONE,                  # Error model
    error_rate=0.0,                                  # For RATE model
    queue_size=100,                                  # Packets
    mtu=1500,                                        # Bytes
    fixed_delay_ms=0.0,                              # For FIXED model
)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Ensure all tests pass: `pytest ns3_spice_tests/ -v`
4. Submit a pull request

## License

[License information here]

## Acknowledgments

- NASA NAIF for the SPICE toolkit
- NS-3 development team
- SNS3 satellite network simulator team