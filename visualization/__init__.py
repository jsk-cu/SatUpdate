#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SatUpdate Visualization Package

This package provides Pygame-based visualization components for
satellite constellation simulations.

The visualization package depends on the simulation package but can be
optionally omitted if only running headless simulations.

Usage:
    from SatUpdate.visualization import Visualizer
    
    visualizer = Visualizer()
    visualizer.create_simulation("walker_delta", num_planes=4, sats_per_plane=6)
    visualizer.run()
"""

from .camera import Camera
from .renderer import Renderer, Colors
from .visualizer import Visualizer, run_visualizer


__all__ = [
    "Camera",
    "Renderer", 
    "Colors",
    "Visualizer",
    "run_visualizer",
]

__version__ = "1.0.0"