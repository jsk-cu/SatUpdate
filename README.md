# SatUpdate - Satellite Constellation Simulator

A Python-based satellite constellation simulator for experimenting with orbital mechanics and multi-satellite software update distribution algorithms.

## Overview

SatUpdate provides a complete framework for simulating satellite constellations and experimenting with distributed packet dissemination protocols. The simulator includes:

- **Keplerian orbital mechanics** for accurate satellite positioning
- **Multiple constellation patterns** (Walker-Delta, Walker-Star, random)
- **Agent-based packet distribution** protocol for software update simulation
- **Ground station** modeling with configurable position and range
- **Real-time 3D visualization** using Pygame
- **Headless mode** for batch simulations and analysis

## Project Structure

```
SatUpdate/
├── simulation/              # Core simulation engine
│   ├── __init__.py
│   ├── orbit.py             # Keplerian orbital mechanics (EllipticalOrbit)
│   ├── satellite.py         # Satellite class with position tracking
│   ├── constellation.py     # Constellation generation factories
│   ├── base_station.py      # Ground station modeling
│   └── simulation.py        # Main Simulation class with agent protocol
│
├── visualization/           # Pygame-based 3D rendering
│   ├── __init__.py
│   ├── camera.py            # Spherical coordinate camera
│   ├── renderer.py          # Earth, orbits, satellites rendering
│   └── visualizer.py        # Main visualization loop
│
├── agents/                  # Packet distribution agents
│   ├── __init__.py          # Agent registry and utilities
│   ├── base_agent.py        # Dummy agent (no requests)
│   └── min_agent.py         # Minimum-first strategy
│
├── examples/                # Example scripts
│   └── run_simulation.py
│
├── main.py                  # Command-line interface
└── README.md
```

## Installation

### Requirements

- Python 3.8+
- NumPy
- Pygame (for visualization only)

```bash
# Install dependencies
pip install numpy pygame

# Run from the SatUpdate directory
cd SatUpdate
python main.py --help
```

## Quick Start

### Command Line

```bash
# Default Walker-Delta constellation (3 planes × 4 satellites)
python main.py

# Walker-Star polar constellation
python main.py --type walker_star --planes 6 --sats-per-plane 6

# Random constellation with 15 satellites
python main.py --type random --num 15

# Custom communication ranges
python main.py --comm-range 3000 --bs-range 5000

# Use different agent controllers
python main.py --agent-controller min   # Orders by completion (default)
python main.py --agent-controller base  # Dummy agent (no distribution)

# Headless simulation for 2 hours
python main.py --headless --duration 7200 --timestep 60

# Full help
python main.py --help
```

### Python API

```python
from simulation import Simulation, SimulationConfig, ConstellationType
import math

# Create configuration
config = SimulationConfig(
    constellation_type=ConstellationType.WALKER_DELTA,
    num_planes=4,
    sats_per_plane=6,
    altitude=550,                    # km
    inclination=math.radians(53),
    num_packets=100,                 # packets in software update
    communication_range=5000,        # inter-satellite range (km)
    base_station_latitude=40.7,      # New York
    base_station_longitude=-74.0,
    base_station_range=8000,         # ground station range (km)
)

# Create and initialize simulation
sim = Simulation(config)
sim.initialize()

# Run simulation until update complete
while not sim.is_update_complete():
    sim.step(60)  # 60 second timestep
    stats = sim.state.agent_statistics
    print(f"Time: {sim.simulation_time/60:.0f} min, "
          f"Completion: {stats.average_completion:.1f}%")
```

## Command-Line Arguments

### Constellation Type
| Argument | Description |
|----------|-------------|
| `--type`, `-t` | Constellation type: `walker_delta`, `walker_star`, `random` |

### Walker Constellation Parameters
| Argument | Default | Description |
|----------|---------|-------------|
| `--planes`, `-p` | 3 | Number of orbital planes |
| `--sats-per-plane`, `-s` | 4 | Satellites per plane |
| `--phasing`, `-f` | 1 | Walker phasing parameter F |

### Random Constellation
| Argument | Default | Description |
|----------|---------|-------------|
| `--num`, `-n` | 10 | Number of satellites |

### Orbital Parameters
| Argument | Default | Description |
|----------|---------|-------------|
| `--altitude`, `-a` | 550 | Orbital altitude (km) |
| `--inclination`, `-i` | 53 | Inclination (degrees) |

### Communication Parameters
| Argument | Default | Description |
|----------|---------|-------------|
| `--comm-range` | unlimited | Inter-satellite communication range (km) |
| `--num-packets` | 100 | Packets in software update |

### Agent Controller
| Argument | Default | Description |
|----------|---------|-------------|
| `--agent-controller` | min | Agent type: `base` (dummy), `min` (completion-ordered) |

### Base Station Parameters
| Argument | Default | Description |
|----------|---------|-------------|
| `--bs-latitude` | 0 | Base station latitude (degrees) |
| `--bs-longitude` | 0 | Base station longitude (degrees) |
| `--bs-altitude` | 0 | Base station altitude (km) |
| `--bs-range` | 10000 | Base station communication range (km) |

### Simulation Control
| Argument | Default | Description |
|----------|---------|-------------|
| `--time-scale` | 60 | Simulation seconds per real second |
| `--seed` | random | Random seed for reproducibility |
| `--paused` | false | Start simulation paused |

### Headless Mode
| Argument | Default | Description |
|----------|---------|-------------|
| `--headless` | false | Run without visualization |
| `--duration` | 3600 | Simulation duration (seconds) |
| `--timestep` | 60 | Simulation timestep (seconds) |

### Window Settings
| Argument | Default | Description |
|----------|---------|-------------|
| `--width` | 1000 | Window width (pixels) |
| `--height` | 800 | Window height (pixels) |

## Visualization Controls

| Key | Action |
|-----|--------|
| Arrow keys | Rotate camera |
| `+` / `-` | Zoom in/out |
| `[` / `]` | Decrease/increase time scale |
| `SPACE` | Pause/Resume |
| `R` | Regenerate constellation |
| `ESC` | Quit |

### Satellite Colors

Satellites are colored based on their software update completion status:
- **Red**: 0% packets received
- **Yellow**: 50% packets received  
- **Green**: 100% packets received (fully updated)

## Constellation Types

### Walker-Delta

Standard Walker constellation with orbital planes evenly distributed over 360° of RAAN (Right Ascension of Ascending Node). Used by LEO constellations like Starlink.

```bash
python main.py --type walker_delta --planes 6 --sats-per-plane 8
```

### Walker-Star

Similar to Walker-Delta but planes are distributed over 180° of RAAN, creating a "star" pattern when viewed from above the pole. Used for polar orbit constellations like Iridium.

```bash
python main.py --type walker_star --planes 6 --sats-per-plane 6 --inclination 86
```

### Random

Randomized orbits for testing and experimentation. Each satellite gets independent orbital parameters.

```bash
python main.py --type random --num 15 --seed 42
```

## Software Update Distribution Protocol

The simulation includes an agent-based packet distribution protocol:

1. **Base Station** starts with all packets (complete software update)
2. **Satellites** start with no packets
3. Each timestep, a 4-phase protocol runs:
   - **Phase 1: Broadcast** - Agents announce what packets they have
   - **Phase 2: Request** - Agents request packets from neighbors
   - **Phase 3: Respond** - Agents decide which requests to fulfill
   - **Phase 4: Transfer** - Packets are transferred between agents

### Agent Types

| Agent | Description |
|-------|-------------|
| `base` | **Dummy agent**: Makes no requests. No packet distribution occurs. Useful as a control case. |
| `min` | **Minimum-first**: Orders neighbors by completion percentage (lowest first), then requests the lowest-indexed missing packets from each. |

### Communication Requirements

For two satellites to communicate:
1. **Line of sight** - Not blocked by Earth
2. **Within range** - Distance ≤ `communication_range` (if set)

For satellite-to-ground communication:
1. **Line of sight** - Satellite visible from ground station
2. **Within range** - Distance ≤ `base_station_range`
3. **Above horizon** - Satellite elevation ≥ minimum elevation angle

## API Reference

### SimulationConfig

```python
@dataclass
class SimulationConfig:
    constellation_type: ConstellationType  # WALKER_DELTA, WALKER_STAR, RANDOM
    num_planes: int = 3                    # Orbital planes (Walker)
    sats_per_plane: int = 4                # Satellites per plane (Walker)
    num_satellites: int = 12               # Total satellites (random)
    altitude: float = 550.0                # Orbital altitude (km)
    inclination: float = 0.925             # Inclination (radians)
    phasing_parameter: int = 1             # Walker phasing F
    random_seed: Optional[int] = None      # For reproducibility
    communication_range: Optional[float] = None  # km, None = unlimited
    num_packets: int = 100                 # Packets in update
    base_station_latitude: float = 0.0     # degrees
    base_station_longitude: float = 0.0    # degrees
    base_station_altitude: float = 0.0     # km
    base_station_range: float = 10000.0    # km
```

### Simulation Class

```python
class Simulation:
    def initialize(self) -> None
    def step(self, timestep: float) -> SimulationState
    def run(self, duration: float, timestep: float) -> List[SimulationState]
    def reset(self) -> None
    def is_update_complete(self) -> bool
    def get_inter_satellite_distances(self) -> Dict[Tuple[str, str], float]
    def get_line_of_sight_matrix(self) -> Dict[Tuple[str, str], bool]
```

### Satellite Class

```python
class Satellite:
    def step(self, timestep: float) -> None
    def get_position_eci(self) -> np.ndarray      # [x, y, z] in km
    def get_velocity_eci(self) -> np.ndarray      # [vx, vy, vz] in km/s
    def get_geospatial_position() -> GeospatialPosition
    def distance_to(self, other: Satellite) -> float
    def has_line_of_sight(self, other: Satellite) -> bool
```

### EllipticalOrbit Class

```python
class EllipticalOrbit:
    # Primary parameters
    apoapsis: float              # km from Earth center
    periapsis: float             # km from Earth center
    inclination: float           # radians
    longitude_of_ascending_node: float  # RAAN, radians
    argument_of_periapsis: float        # radians
    
    # Derived properties
    semi_major_axis: float       # km
    eccentricity: float          # 0 to <1
    period: float                # seconds
```

## Extending the Agent Protocol

Agents use a class hierarchy where `BaseAgent` is the base class and custom
agents subclass it, overriding `make_requests()` to implement their strategy.

### Class Hierarchy

```
BaseAgent (base class)
├── Provides 4-phase protocol interface
├── Default make_requests() returns {} (no requests)
└── Useful as control case

MinAgent(BaseAgent)
├── Overrides make_requests()
└── Orders neighbors by completion, requests lowest packets
```

### Creating a Custom Agent

```python
from agents import BaseAgent, register_agent

class MyAgent(BaseAgent):
    name = "my_agent"
    description = "My custom distribution strategy"

    def make_requests(self, neighbor_broadcasts):
        """Override to implement custom request logic."""
        requests = {}
        missing = self.get_missing_packets()

        for neighbor_id, broadcast in neighbor_broadcasts.items():
            available = missing & broadcast.get("packets", set())
            if available:
                # Request the lowest-indexed available packet
                requests[neighbor_id] = min(available)
                missing.discard(requests[neighbor_id])

        return requests

# Register so it can be used via --agent-controller my_agent
register_agent("my_agent", MyAgent)
```

### Available Methods from BaseAgent

When subclassing, you have access to these utility methods:

| Method | Description |
|--------|-------------|
| `self.packets` | Set of packet indices this agent has |
| `self.get_missing_packets()` | Returns set of missing packet indices |
| `self.has_all_packets()` | True if agent has all packets |
| `self.get_completion_percentage()` | Returns 0-100 completion % |
| `self.is_base_station` | True if this is the ground station |

### Protocol Methods

| Method | When Called | Default Behavior |
|--------|-------------|------------------|
| `broadcast_state()` | Phase 1 | Returns packets, completion, etc. |
| `make_requests(broadcasts)` | Phase 2 | Returns `{}` (override this!) |
| `receive_requests_and_update(requests)` | Phase 3 | Grants all valid requests |
| `receive_packets_and_update(received)` | Phase 4 | Adds received packets |

## Example: Batch Analysis

```python
from simulation import Simulation, SimulationConfig, ConstellationType
import math

results = []
for num_planes in [3, 4, 6, 8]:
    config = SimulationConfig(
        constellation_type=ConstellationType.WALKER_DELTA,
        num_planes=num_planes,
        sats_per_plane=6,
        altitude=550,
        inclination=math.radians(53),
        num_packets=50,
        random_seed=42,
    )
    
    sim = Simulation(config)
    sim.initialize()
    
    steps = 0
    while not sim.is_update_complete() and steps < 1000:
        sim.step(60)
        steps += 1
    
    results.append({
        'planes': num_planes,
        'satellites': sim.num_satellites,
        'time_to_complete': sim.simulation_time,
        'steps': steps,
    })

for r in results:
    print(f"{r['planes']} planes, {r['satellites']} sats: "
          f"{r['time_to_complete']/60:.0f} min ({r['steps']} steps)")
```

## Units

- **Distance**: kilometers (km)
- **Angles**: radians (internal), degrees (CLI)
- **Time**: seconds
- **Velocity**: km/s

## License

MIT License