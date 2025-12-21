#!/usr/bin/env python3
"""
Example: Running Satellite Constellation Simulations

This script demonstrates how to run simulations without visualization,
which is useful for:
- Running simulations faster than real-time
- Batch processing and data collection
- Testing communication algorithms
- Running on headless servers
"""

import math
import sys
import os

# Add parent of SatUpdate to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from SatUpdate.simulation import (
    Simulation,
    SimulationConfig,
    ConstellationType,
    EllipticalOrbit,
    Satellite,
    create_walker_delta_constellation,
    create_walker_star_constellation,
    EARTH_RADIUS_KM,
)


def example_using_simulation_class():
    """
    Example using the high-level Simulation class.
    """
    print("=" * 70)
    print("Example 1: Using the Simulation Class")
    print("=" * 70)
    
    # Create a simulation configuration
    config = SimulationConfig(
        constellation_type=ConstellationType.WALKER_DELTA,
        num_planes=4,
        sats_per_plane=6,
        altitude=550,
        inclination=math.radians(53),
        phasing_parameter=1,
    )
    
    # Create and initialize the simulation
    sim = Simulation(config)
    sim.initialize()
    
    print(f"\nCreated simulation: {sim}")
    print(f"Total satellites: {sim.num_satellites}")
    print(f"Orbital planes: {sim.num_orbits}")
    
    # Run simulation for 30 minutes with 1-minute timesteps
    print("\nRunning simulation for 30 minutes...")
    
    for minute in range(30):
        sim.step(60)  # 60 second timestep
        
        if (minute + 1) % 10 == 0:
            print(f"\n  After {minute + 1} minutes:")
            # Show first 3 satellites
            for sat_id, geo in list(sim.state.satellite_positions.items())[:3]:
                print(f"    {sat_id}: alt={geo.altitude:.0f} km, "
                      f"lat={geo.latitude_deg:+.1f}°, lon={geo.longitude_deg:+.1f}°")
    
    # Get final statistics
    print(f"\n  Final state:")
    print(f"    Simulation time: {sim.simulation_time:.0f} seconds")
    print(f"    Steps executed: {sim.state.step_count}")
    
    # Calculate line of sight statistics
    los_matrix = sim.get_line_of_sight_matrix()
    visible = sum(1 for v in los_matrix.values() if v)
    total = len(los_matrix)
    print(f"    LOS pairs: {visible}/{total} ({visible/total*100:.1f}%)")
    
    return sim


def example_walker_delta_constellation():
    """
    Example directly creating a Walker-Delta constellation.
    """
    print("\n" + "=" * 70)
    print("Example 2: Walker-Delta Constellation (Direct Creation)")
    print("=" * 70)
    
    # Create a Walker-Delta constellation: 6 planes, 8 satellites per plane
    orbits, satellites = create_walker_delta_constellation(
        num_planes=6,
        sats_per_plane=8,
        altitude=550,  # km
        inclination=math.radians(53),
        phasing_parameter=1
    )
    
    print(f"\nCreated constellation with {len(satellites)} satellites")
    print(f"Orbital altitude: 550 km")
    print(f"Inclination: 53°")
    
    # Get orbital period from first satellite
    period = satellites[0].orbit.period
    print(f"Orbital period: {period/60:.1f} minutes")
    
    # Simulate for one full orbit
    print("\nSimulating one complete orbit...")
    
    timestep = 60.0  # 1 minute timestep
    sim_time = 0.0
    
    # Track visibility statistics
    visibility_samples = []
    
    while sim_time < period:
        # Count line-of-sight connections
        visible_pairs = 0
        total_pairs = 0
        
        for i, sat_a in enumerate(satellites):
            for sat_b in satellites[i+1:]:
                total_pairs += 1
                if sat_a.has_line_of_sight(sat_b):
                    visible_pairs += 1
        
        visibility_samples.append(visible_pairs / total_pairs * 100)
        
        # Advance all satellites
        for sat in satellites:
            sat.step(timestep)
        
        sim_time += timestep
        
        # Print progress every 15 minutes
        if int(sim_time) % 900 == 0:
            print(f"  t={sim_time/60:5.1f} min: "
                  f"{visible_pairs}/{total_pairs} pairs visible "
                  f"({visibility_samples[-1]:.1f}%)")
    
    # Summary statistics
    avg_visibility = sum(visibility_samples) / len(visibility_samples)
    min_visibility = min(visibility_samples)
    max_visibility = max(visibility_samples)
    
    print(f"\nVisibility Statistics:")
    print(f"  Average visibility: {avg_visibility:.1f}%")
    print(f"  Minimum visibility: {min_visibility:.1f}%")
    print(f"  Maximum visibility: {max_visibility:.1f}%")


def example_walker_star_constellation():
    """
    Example creating a Walker-Star (polar) constellation.
    """
    print("\n" + "=" * 70)
    print("Example 3: Walker-Star (Polar) Constellation")
    print("=" * 70)
    
    # Create a Walker-Star constellation for polar coverage
    orbits, satellites = create_walker_star_constellation(
        num_planes=6,
        sats_per_plane=6,
        altitude=780,  # km (similar to Iridium)
        inclination=math.radians(86.4),  # Near-polar
        phasing_parameter=1
    )
    
    print(f"\nCreated polar constellation with {len(satellites)} satellites")
    print(f"Orbital altitude: 780 km")
    print(f"Inclination: 86.4°")
    print(f"Orbital period: {satellites[0].orbit.period/60:.1f} minutes")
    
    # Show initial satellite distribution
    print("\nInitial satellite positions:")
    for i, sat in enumerate(satellites[:6]):
        geo = sat.get_geospatial_position()
        print(f"  {sat.satellite_id}: lat={geo.latitude_deg:+.1f}°, lon={geo.longitude_deg:+.1f}°")
    print(f"  ... and {len(satellites) - 6} more")


def example_elliptical_orbit():
    """
    Example creating a highly elliptical (Molniya-type) orbit.
    """
    print("\n" + "=" * 70)
    print("Example 4: Highly Elliptical (Molniya) Orbit")
    print("=" * 70)
    
    # Create a Molniya-type orbit
    molniya_orbit = EllipticalOrbit(
        apoapsis=EARTH_RADIUS_KM + 39873,  # ~40,000 km altitude
        periapsis=EARTH_RADIUS_KM + 500,    # ~500 km altitude
        inclination=math.radians(63.4),     # Critical inclination
        longitude_of_ascending_node=math.radians(45),
        argument_of_periapsis=math.radians(270),  # Apoapsis over northern hemisphere
    )
    
    satellite = Satellite(molniya_orbit, initial_position=0.0, satellite_id="MOLNIYA-1")
    
    print(f"\nMolniya orbit parameters:")
    print(f"  Periapsis altitude: {molniya_orbit.periapsis_altitude:.0f} km")
    print(f"  Apoapsis altitude: {molniya_orbit.apoapsis_altitude:.0f} km")
    print(f"  Eccentricity: {molniya_orbit.eccentricity:.4f}")
    print(f"  Period: {molniya_orbit.period/3600:.1f} hours")
    
    # Show position and velocity throughout orbit
    print("\nPosition and velocity throughout orbit:")
    print(f"{'Position':^10} {'Altitude (km)':^15} {'Speed (km/s)':^15} {'Lat':^10}")
    print("-" * 60)
    
    for pos_frac in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8]:
        satellite.position = pos_frac
        geo = satellite.get_geospatial_position()
        speed = satellite.get_speed()
        
        print(f"{pos_frac:^10.1f} {geo.altitude:^15,.0f} {speed:^15.2f} "
              f"{geo.latitude_deg:^+10.1f}")


def example_communication_simulation():
    """
    Example simulating communication windows between satellites.
    """
    print("\n" + "=" * 70)
    print("Example 5: Inter-Satellite Communication Simulation")
    print("=" * 70)
    
    # Create two satellites in different orbits
    leo_orbit = EllipticalOrbit(
        apoapsis=EARTH_RADIUS_KM + 550,
        periapsis=EARTH_RADIUS_KM + 550,
        inclination=math.radians(53),
        longitude_of_ascending_node=0,
        argument_of_periapsis=0,
    )
    leo_sat = Satellite(leo_orbit, initial_position=0.0, satellite_id="LEO-1")
    
    geo_orbit = EllipticalOrbit(
        apoapsis=EARTH_RADIUS_KM + 35786,
        periapsis=EARTH_RADIUS_KM + 35786,
        inclination=math.radians(0),
        longitude_of_ascending_node=0,
        argument_of_periapsis=0,
    )
    geo_sat = Satellite(geo_orbit, initial_position=0.0, satellite_id="GEO-1")
    
    print(f"\nSatellite configuration:")
    print(f"  LEO: {leo_sat.satellite_id} at {leo_orbit.periapsis_altitude:.0f} km")
    print(f"  GEO: {geo_sat.satellite_id} at {geo_orbit.periapsis_altitude:.0f} km")
    
    # Track communication windows over one LEO orbit
    print("\nTracking communication windows...")
    
    timestep = 30.0  # 30 second timestep
    sim_time = 0.0
    period = leo_sat.orbit.period
    
    in_contact = False
    contact_start = None
    total_contact_time = 0.0
    contact_count = 0
    
    while sim_time < period:
        los = leo_sat.has_line_of_sight(geo_sat)
        distance = leo_sat.distance_to(geo_sat)
        
        if los and not in_contact:
            in_contact = True
            contact_start = sim_time
            contact_count += 1
            print(f"  Contact {contact_count} START at t={sim_time/60:.1f} min, "
                  f"distance={distance:.0f} km")
        elif not los and in_contact:
            in_contact = False
            contact_duration = sim_time - contact_start
            total_contact_time += contact_duration
            print(f"  Contact {contact_count} END at t={sim_time/60:.1f} min, "
                  f"duration={contact_duration/60:.1f} min")
        
        leo_sat.step(timestep)
        geo_sat.step(timestep)
        sim_time += timestep
    
    # Handle ongoing contact at end
    if in_contact:
        contact_duration = sim_time - contact_start
        total_contact_time += contact_duration
    
    print(f"\nSummary:")
    print(f"  Total contact windows: {contact_count}")
    print(f"  Total contact time: {total_contact_time/60:.1f} min "
          f"({total_contact_time/period*100:.1f}% of LEO orbit)")


def main():
    """Run all examples."""
    example_using_simulation_class()
    example_walker_delta_constellation()
    example_walker_star_constellation()
    example_elliptical_orbit()
    example_communication_simulation()
    
    print("\n" + "=" * 70)
    print("All examples complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()