#!/usr/bin/env python3
"""
SatUpdate - Satellite Constellation Simulator

Main entry point for running satellite constellation simulations with
optional visualization.

Usage:
    python main.py                           # Default Walker-Delta constellation
    python main.py --type walker_star        # Walker-Star (polar) constellation
    python main.py --type random --num 10    # 10 random satellites
    python main.py --headless --duration 3600  # Headless simulation
    python main.py --help                    # Show all options
"""

import argparse
import math
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Satellite Constellation Simulator and Visualizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                              # Default Walker-Delta (3 planes × 4 sats)
  python main.py --type walker_delta -p 6 -s 8  # 6 planes × 8 satellites
  python main.py --type walker_star -p 4 -s 6   # Polar constellation
  python main.py --type random --num 15         # 15 random satellites
  python main.py --headless --duration 7200     # 2-hour headless run

Controls (when visualizing):
  Arrow keys  : Rotate camera
  +/-         : Zoom in/out
  [ ]         : Decrease/increase time scale
  SPACE       : Pause/Resume
  R           : Regenerate constellation
  ESC         : Quit
        """
    )
    
    # Constellation type
    parser.add_argument(
        "--type", "-t",
        type=str,
        choices=["walker_delta", "walker_star", "random"],
        default="walker_delta",
        help="Constellation type (default: walker_delta)"
    )
    
    # Walker constellation parameters
    parser.add_argument(
        "--planes", "-p",
        type=int,
        default=3,
        help="Number of orbital planes for Walker constellations (default: 3)"
    )
    parser.add_argument(
        "--sats-per-plane", "-s",
        type=int,
        default=4,
        help="Satellites per plane for Walker constellations (default: 4)"
    )
    parser.add_argument(
        "--phasing", "-f",
        type=int,
        default=1,
        help="Walker phasing parameter F (default: 1)"
    )
    
    # Random constellation parameters
    parser.add_argument(
        "--num", "-n",
        type=int,
        default=10,
        help="Number of satellites for random constellation (default: 10)"
    )
    
    # Orbital parameters
    parser.add_argument(
        "--altitude", "-a",
        type=float,
        default=550.0,
        help="Orbital altitude in km (default: 550)"
    )
    parser.add_argument(
        "--inclination", "-i",
        type=float,
        default=53.0,
        help="Orbital inclination in degrees (default: 53)"
    )
    
    # Simulation control
    parser.add_argument(
        "--time-scale",
        type=float,
        default=60.0,
        help="Time scale (sim seconds per real second, default: 60)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--paused",
        action="store_true",
        help="Start with simulation paused"
    )
    
    # Headless mode
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run simulation without visualization"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=3600.0,
        help="Simulation duration in seconds for headless mode (default: 3600)"
    )
    parser.add_argument(
        "--timestep",
        type=float,
        default=60.0,
        help="Simulation timestep in seconds for headless mode (default: 60)"
    )
    
    # Window settings
    parser.add_argument(
        "--width",
        type=int,
        default=1000,
        help="Window width in pixels (default: 1000)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=800,
        help="Window height in pixels (default: 800)"
    )
    
    args = parser.parse_args()
    
    # Import simulation components
    from simulation import (
        Simulation,
        SimulationConfig,
        ConstellationType,
    )
    
    # Map string type to enum
    type_map = {
        "walker_delta": ConstellationType.WALKER_DELTA,
        "walker_star": ConstellationType.WALKER_STAR,
        "random": ConstellationType.RANDOM,
    }
    
    # Create simulation configuration
    config = SimulationConfig(
        constellation_type=type_map[args.type],
        num_planes=args.planes,
        sats_per_plane=args.sats_per_plane,
        num_satellites=args.num,
        altitude=args.altitude,
        inclination=math.radians(args.inclination),
        phasing_parameter=args.phasing,
        random_seed=args.seed,
    )
    
    # Create and initialize simulation
    sim = Simulation(config)
    sim.initialize()
    
    # Print constellation info
    print("=" * 60)
    print("SatUpdate - Satellite Constellation Simulator")
    print("=" * 60)
    print(f"\nConstellation Type: {args.type}")
    print(f"Total Satellites: {sim.num_satellites}")
    print(f"Orbital Planes: {sim.num_orbits}")
    print(f"Altitude: {args.altitude} km")
    print(f"Inclination: {args.inclination}°")
    
    if args.type != "random":
        print(f"Phasing Parameter: {args.phasing}")
    
    if sim.satellites:
        period = sim.satellites[0].orbit.period
        print(f"Orbital Period: {period/60:.1f} min")
    
    if args.headless:
        # Run headless simulation
        print(f"\n{'=' * 60}")
        print(f"Running headless simulation for {args.duration:.0f} seconds...")
        print(f"Timestep: {args.timestep:.1f} seconds")
        print(f"{'=' * 60}")
        
        elapsed = 0.0
        report_interval = max(600, args.duration / 10)  # Report at most 10 times
        next_report = report_interval
        
        while elapsed < args.duration:
            sim.step(args.timestep)
            elapsed += args.timestep
            
            if elapsed >= next_report:
                print(f"\nTime: {elapsed/60:.1f} minutes")
                
                # Show sample satellite positions
                for sat_id, geo in list(sim.state.satellite_positions.items())[:3]:
                    print(f"  {sat_id}: alt={geo.altitude:.0f} km, "
                          f"lat={geo.latitude_deg:+.1f}°, lon={geo.longitude_deg:+.1f}°")
                
                if sim.num_satellites > 3:
                    print(f"  ... and {sim.num_satellites - 3} more satellites")
                
                next_report += report_interval
        
        # Final summary
        print(f"\n{'=' * 60}")
        print("Simulation Complete!")
        print(f"{'=' * 60}")
        print(f"Final simulation time: {sim.simulation_time:.0f} seconds")
        print(f"Steps executed: {sim.state.step_count}")
        
        # Calculate inter-satellite statistics
        distances = sim.get_inter_satellite_distances()
        if distances:
            min_dist = min(distances.values())
            max_dist = max(distances.values())
            avg_dist = sum(distances.values()) / len(distances)
            print(f"\nInter-satellite distances:")
            print(f"  Min: {min_dist:.0f} km")
            print(f"  Max: {max_dist:.0f} km")
            print(f"  Avg: {avg_dist:.0f} km")
        
        los_matrix = sim.get_line_of_sight_matrix()
        if los_matrix:
            visible = sum(1 for v in los_matrix.values() if v)
            total = len(los_matrix)
            print(f"\nLine of sight: {visible}/{total} pairs visible ({visible/total*100:.1f}%)")
    
    else:
        # Run with visualization
        try:
            from visualization import Visualizer
        except ImportError as e:
            print(f"\nError: Could not import visualization module: {e}")
            print("Try running with --headless flag for simulation without graphics.")
            sys.exit(1)
        
        print(f"\n{'=' * 60}")
        print("Starting Visualization")
        print(f"{'=' * 60}")
        print("\nControls:")
        print("  Arrow keys  : Rotate camera")
        print("  +/-         : Zoom in/out")
        print("  [ ]         : Decrease/increase time scale")
        print("  SPACE       : Pause/Resume")
        print("  R           : Regenerate constellation")
        print("  ESC         : Quit")
        print()
        
        visualizer = Visualizer(
            width=args.width,
            height=args.height,
            time_scale=args.time_scale,
            paused=args.paused
        )
        
        # Set the simulation we created
        visualizer.set_simulation(sim)
        
        # Run the visualization
        visualizer.run()


if __name__ == "__main__":
    main()