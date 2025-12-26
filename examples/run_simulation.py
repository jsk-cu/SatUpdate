#!/usr/bin/env python3
"""
Example: Running Satellite Constellation Simulations

Demonstrates how to run simulations programmatically without visualization.
Useful for:
- Batch processing and data collection
- Testing distribution algorithms
- Running on headless servers
- Performance analysis
"""

import math
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation import (
    Simulation,
    SimulationConfig,
    ConstellationType,
    EllipticalOrbit,
    Satellite,
    create_walker_delta_constellation,
    create_walker_star_constellation,
    EARTH_RADIUS_KM,
)


def example_basic_simulation():
    """
    Basic example using the Simulation class.
    """
    print("=" * 70)
    print("Example 1: Basic Simulation")
    print("=" * 70)

    config = SimulationConfig(
        constellation_type=ConstellationType.WALKER_DELTA,
        num_planes=3,
        sats_per_plane=4,
        altitude=550,
        inclination=math.radians(53),
        num_packets=50,
    )

    sim = Simulation(config)
    sim.initialize()

    print(f"\nCreated constellation:")
    print(f"  Satellites: {sim.num_satellites}")
    print(f"  Orbital planes: {sim.num_orbits}")
    print(f"  Packets: {config.num_packets}")

    # Run until complete or timeout
    print("\nRunning simulation...")
    max_steps = 100
    for step in range(max_steps):
        sim.step(60)

        if (step + 1) % 20 == 0:
            stats = sim.state.agent_statistics
            print(
                f"  Step {step+1}: {stats.average_completion:.1f}% avg, "
                f"{stats.fully_updated_count}/{sim.num_satellites} complete"
            )

        if sim.is_update_complete():
            print(f"\n  Update complete at step {step+1}!")
            break

    stats = sim.state.agent_statistics
    print(f"\nFinal state:")
    print(f"  Time: {sim.simulation_time/60:.1f} minutes")
    print(f"  Average completion: {stats.average_completion:.1f}%")
    print(f"  Fully updated: {stats.fully_updated_count}/{sim.num_satellites}")


def example_custom_base_station():
    """
    Example with custom base station location.
    """
    print("\n" + "=" * 70)
    print("Example 2: Custom Base Station Location")
    print("=" * 70)

    # Base station in New York
    config = SimulationConfig(
        constellation_type=ConstellationType.WALKER_DELTA,
        num_planes=4,
        sats_per_plane=4,
        altitude=550,
        inclination=math.radians(53),
        num_packets=30,
        base_station_latitude=40.7128,
        base_station_longitude=-74.0060,
        base_station_range=5000,
    )

    sim = Simulation(config)
    sim.initialize()

    print(f"\nBase station: New York ({config.base_station_latitude}°, "
          f"{config.base_station_longitude}°)")
    print(f"Range: {config.base_station_range} km")

    # Run for a few steps and check base station connectivity
    for _ in range(10):
        sim.step(60)

    bs_links = sim.state.base_station_links
    print(f"\nAfter 10 minutes:")
    print(f"  Satellites in range of base station: {len(bs_links)}")

    if bs_links:
        print("  Connected satellites:")
        for bs_name, sat_id in list(bs_links)[:5]:
            geo = sim.state.satellite_positions.get(sat_id)
            if geo:
                print(f"    {sat_id}: lat={geo.latitude_deg:+.1f}°, "
                      f"lon={geo.longitude_deg:+.1f}°")


def example_communication_range():
    """
    Example comparing unlimited vs limited communication range.
    """
    print("\n" + "=" * 70)
    print("Example 3: Communication Range Comparison")
    print("=" * 70)

    base_config = dict(
        constellation_type=ConstellationType.WALKER_DELTA,
        num_planes=3,
        sats_per_plane=4,
        altitude=550,
        inclination=math.radians(53),
        num_packets=30,
        random_seed=42,
    )

    # Run with unlimited range
    config_unlimited = SimulationConfig(**base_config, communication_range=None)
    sim_unlimited = Simulation(config_unlimited)
    sim_unlimited.initialize()

    # Run with limited range
    config_limited = SimulationConfig(**base_config, communication_range=2000)
    sim_limited = Simulation(config_limited)
    sim_limited.initialize()

    print("\nComparing unlimited vs 2000 km communication range:")

    for step in range(50):
        sim_unlimited.step(60)
        sim_limited.step(60)

        if (step + 1) % 10 == 0:
            stats_u = sim_unlimited.state.agent_statistics
            stats_l = sim_limited.state.agent_statistics
            links_u = len(sim_unlimited.state.active_links)
            links_l = len(sim_limited.state.active_links)

            print(f"\n  Step {step+1}:")
            print(f"    Unlimited: {stats_u.average_completion:.1f}% avg, "
                  f"{links_u} active links")
            print(f"    Limited:   {stats_l.average_completion:.1f}% avg, "
                  f"{links_l} active links")

        if sim_unlimited.is_update_complete() and sim_limited.is_update_complete():
            break


def example_walker_star():
    """
    Example with Walker-Star (polar) constellation.
    """
    print("\n" + "=" * 70)
    print("Example 4: Walker-Star Polar Constellation")
    print("=" * 70)

    orbits, satellites = create_walker_star_constellation(
        num_planes=6,
        sats_per_plane=6,
        altitude=780,
        inclination=math.radians(86.4),
        phasing_parameter=1,
    )

    print(f"\nCreated polar constellation:")
    print(f"  Satellites: {len(satellites)}")
    print(f"  Altitude: 780 km")
    print(f"  Inclination: 86.4°")
    print(f"  Orbital period: {satellites[0].orbit.period/60:.1f} minutes")

    # Show initial positions
    print("\nSample satellite positions:")
    for sat in satellites[:6]:
        geo = sat.get_geospatial_position()
        print(f"  {sat.satellite_id}: lat={geo.latitude_deg:+6.1f}°, "
              f"lon={geo.longitude_deg:+7.1f}°")


def example_elliptical_orbit():
    """
    Example with highly elliptical (Molniya-type) orbit.
    """
    print("\n" + "=" * 70)
    print("Example 5: Highly Elliptical (Molniya) Orbit")
    print("=" * 70)

    molniya = EllipticalOrbit(
        apoapsis=EARTH_RADIUS_KM + 39873,
        periapsis=EARTH_RADIUS_KM + 500,
        inclination=math.radians(63.4),
        longitude_of_ascending_node=math.radians(45),
        argument_of_periapsis=math.radians(270),
    )

    satellite = Satellite(molniya, initial_position=0.0, satellite_id="MOLNIYA-1")

    print(f"\nMolniya orbit:")
    print(f"  Periapsis altitude: {molniya.periapsis_altitude:.0f} km")
    print(f"  Apoapsis altitude: {molniya.apoapsis_altitude:.0f} km")
    print(f"  Eccentricity: {molniya.eccentricity:.4f}")
    print(f"  Period: {molniya.period/3600:.1f} hours")

    print("\nPosition throughout orbit:")
    print(f"{'Position':^10} {'Altitude (km)':^15} {'Speed (km/s)':^12} {'Lat':^10}")
    print("-" * 50)

    for pos in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8]:
        satellite.position = pos
        geo = satellite.get_geospatial_position()
        speed = satellite.get_speed()
        print(f"{pos:^10.1f} {geo.altitude:^15,.0f} {speed:^12.2f} "
              f"{geo.latitude_deg:^+10.1f}")


def example_batch_comparison():
    """
    Example batch analysis comparing different configurations.
    """
    print("\n" + "=" * 70)
    print("Example 6: Batch Configuration Comparison")
    print("=" * 70)

    configurations = [
        {"num_planes": 2, "sats_per_plane": 4},
        {"num_planes": 3, "sats_per_plane": 4},
        {"num_planes": 4, "sats_per_plane": 4},
        {"num_planes": 4, "sats_per_plane": 6},
    ]

    print("\nComparing update completion time:")
    print(f"{'Config':^15} {'Satellites':^12} {'Steps':^10} {'Time (min)':^12}")
    print("-" * 55)

    for cfg in configurations:
        config = SimulationConfig(
            constellation_type=ConstellationType.WALKER_DELTA,
            altitude=550,
            inclination=math.radians(53),
            num_packets=30,
            random_seed=42,
            **cfg,
        )

        sim = Simulation(config)
        sim.initialize()

        steps = 0
        while not sim.is_update_complete() and steps < 500:
            sim.step(60)
            steps += 1

        config_str = f"{cfg['num_planes']}×{cfg['sats_per_plane']}"
        print(f"{config_str:^15} {sim.num_satellites:^12} {steps:^10} "
              f"{sim.simulation_time/60:^12.1f}")


def main():
    """Run all examples."""
    example_basic_simulation()
    example_custom_base_station()
    example_communication_range()
    example_walker_star()
    example_elliptical_orbit()
    example_batch_comparison()

    print("\n" + "=" * 70)
    print("All examples complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()