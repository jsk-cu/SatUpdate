# SatUpdate - Satellite Constellation Simulator

A Python-based satellite constellation simulator for experimenting with orbital mechanics and multi-satellite communication algorithms.

## Project Structure

```
SatUpdate/
├── simulation/          # Numerical simulation (no visualization dependencies)
│   ├── __init__.py
│   ├── orbit.py         # EllipticalOrbit class (Keplerian orbital mechanics)
│   ├── satellite.py     # Satellite class with position tracking
│   ├── constellation.py # Factory functions for constellation generation
│   └── simulation.py    # Main Simulation class
│
├── visualization/       # Pygame-based 3D visualization (optional)
│   ├── __init__.py
│   ├── camera.py        # 3D camera with spherical coordinates
│   ├── renderer.py      # Rendering functions for Earth, orbits, satellites
│   └── visualizer.py    # Main Visualizer class
│
├── examples/            # Example scripts
│   └── run_simulation.py
│
├── main.py              # Command-line entry point
└── README.md
```

## Features

- **Keplerian Orbital Mechanics**: Full support for elliptical orbits with all classical orbital elements
- **Constellation Types**:
  - Walker-Delta: Evenly distributed planes over 360° RAAN
  - Walker-Star: Polar/near-polar with planes over 180° RAAN
  - Random: Randomized orbits for testing
- **Simulation Independence**: Run simulations without visualization for batch processing
- **Interactive 3D Visualization**: Pygame-based renderer with camera controls
- **Inter-satellite Analysis**: Distance calculations and line-of-sight checking

## Installation

```bash
# Install dependencies
pip install numpy pygame

# Run from the parent directory of SatUpdate
cd /path/to/parent
python -m SatUpdate.main --help
```

## Quick Start

### Command Line

```bash
# Default Walker-Delta constellation (3 planes × 4 satellites)
python main.py

# Walker-Star polar constellation
python main.py --type walker_star --planes 6 --sats-per-plane 6

# Random constellation
python main.py --type random --num 15

# Headless simulation (no visualization)
python main.py --headless --duration 7200 --timestep 60

# Full options
python main.py --help
```

### Python API

#### Using the Simulation Class

```python
from SatUpdate.simulation import Simulation, SimulationConfig, ConstellationType
import math

# Create configuration
config = SimulationConfig(
    constellation_type=ConstellationType.WALKER_DELTA,
    num_planes=4,
    sats_per_plane=6,
    altitude=550,  # km
    inclination=math.radians(53),
)

# Create and initialize simulation
sim = Simulation(config)
sim.initialize()

# Run simulation
for _ in range(100):
    sim.step(60)  # 60 second timestep
    print(f"Time: {sim.simulation_time} seconds")

# Get inter-satellite data
distances = sim.get_inter_satellite_distances()
los_matrix = sim.get_line_of_sight_matrix()
```

#### Direct Constellation Creation

```python
from SatUpdate.simulation import (
    create_walker_delta_constellation,
    create_walker_star_constellation,
    create_random_constellation,
)

# Walker-Delta constellation
orbits, satellites = create_walker_delta_constellation(
    num_planes=6,
    sats_per_plane=8,
    altitude=550,
    inclination=math.radians(53),
)

# Advance simulation manually
timestep = 60  # seconds
for sat in satellites:
    sat.step(timestep)
```

#### With Visualization

```python
from SatUpdate.visualization import Visualizer

# Create visualizer
visualizer = Visualizer(width=1000, height=800)

# Create and start simulation
visualizer.create_simulation(
    constellation_type="walker_delta",
    num_planes=4,
    sats_per_plane=6,
)

# Run interactive visualization
visualizer.run()
```

## Controls (Visualization)

| Key | Action |
|-----|--------|
| Arrow keys | Rotate camera |
| +/- | Zoom in/out |
| [ ] | Decrease/increase time scale |
| SPACE | Pause/Resume |
| R | Regenerate constellation |
| ESC | Quit |

## Constellation Types

### Walker-Delta
Standard Walker constellation with orbital planes evenly distributed over 360° of RAAN. Commonly used for LEO constellations like Starlink.

```python
# Walker notation: i:T/P/F
# i = inclination, T = total satellites, P = planes, F = phasing parameter
config = SimulationConfig(
    constellation_type=ConstellationType.WALKER_DELTA,
    num_planes=6,      # P
    sats_per_plane=8,  # T/P
    phasing_parameter=1, # F
)
```

### Walker-Star
Similar to Walker-Delta but planes are distributed over 180° of RAAN, creating a "star" pattern when viewed from above the pole. Used for polar orbit constellations like Iridium.

```python
config = SimulationConfig(
    constellation_type=ConstellationType.WALKER_STAR,
    inclination=math.radians(86.4),  # Near-polar
)
```

### Random
Randomly generated orbits for testing and experimentation.

```python
config = SimulationConfig(
    constellation_type=ConstellationType.RANDOM,
    num_satellites=20,
    random_seed=42,  # For reproducibility
)
```

## API Reference

### Simulation Class

```python
class Simulation:
    def __init__(self, config: SimulationConfig)
    def initialize(self) -> None
    def step(self, timestep: float) -> SimulationState
    def run(self, duration: float, timestep: float) -> List[SimulationState]
    def reset(self) -> None
    def regenerate(self, new_seed: Optional[int] = None) -> None
    def get_inter_satellite_distances(self) -> Dict[Tuple[str, str], float]
    def get_line_of_sight_matrix(self) -> Dict[Tuple[str, str], bool]
```

### Satellite Class

```python
class Satellite:
    def __init__(self, orbit: EllipticalOrbit, initial_position: float, satellite_id: str)
    def step(self, timestep: float) -> None
    def get_position_eci(self) -> np.ndarray  # [x, y, z] in km
    def get_velocity_eci(self) -> np.ndarray  # [vx, vy, vz] in km/s
    def get_geospatial_position(self) -> GeospatialPosition  # lat, lon, alt
    def distance_to(self, other: Satellite) -> float
    def has_line_of_sight(self, other: Satellite) -> bool
```

### EllipticalOrbit Class

```python
class EllipticalOrbit:
    def __init__(
        self,
        apoapsis: float,      # km from Earth center
        periapsis: float,     # km from Earth center
        inclination: float,   # radians
        longitude_of_ascending_node: float,  # radians (RAAN)
        argument_of_periapsis: float,  # radians
        earth_radius: float = EARTH_RADIUS_KM,
        earth_mass: float = EARTH_MASS_KG,
    )
    
    # Properties
    semi_major_axis: float
    eccentricity: float
    period: float  # seconds
    apoapsis_altitude: float  # km above surface
    periapsis_altitude: float  # km above surface
```

## License

MIT License
