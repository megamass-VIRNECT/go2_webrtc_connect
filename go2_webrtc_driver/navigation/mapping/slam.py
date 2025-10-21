"""
SLAM (Simultaneous Localization and Mapping) implementation
Uses occupancy grid mapping with bresenham line algorithm
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from ..utils.map_utils import OccupancyGrid


@dataclass
class SLAMParams:
    """SLAM parameters"""
    map_width: int = 1000           # Map width in cells
    map_height: int = 1000          # Map height in cells
    map_resolution: float = 0.05    # Map resolution (meters/cell)
    map_origin: Tuple[float, float] = (-25.0, -25.0)  # Map origin (meters)

    # Occupancy grid update parameters
    log_odds_occupied: float = 0.7   # Log-odds update for occupied cells
    log_odds_free: float = -0.4      # Log-odds update for free cells

    # Sensor parameters
    max_range: float = 10.0          # Maximum sensor range (meters)
    min_range: float = 0.1           # Minimum sensor range (meters)


class SLAM:
    """
    SLAM implementation using occupancy grid mapping
    """

    def __init__(self, params: SLAMParams = None):
        """
        Initialize SLAM

        Args:
            params: SLAM parameters
        """
        self.params = params if params is not None else SLAMParams()

        # Create occupancy grid
        self.map = OccupancyGrid(
            width=self.params.map_width,
            height=self.params.map_height,
            resolution=self.params.map_resolution,
            origin=self.params.map_origin
        )

        # Robot trajectory
        self.trajectory: List[Tuple[float, float, float]] = []

    def update(self,
               pose: Tuple[float, float, float],
               scan: List[float],
               scan_angles: List[float]):
        """
        Update map with new scan from given pose

        Args:
            pose: Robot pose (x, y, theta)
            scan: List of range measurements
            scan_angles: List of angles for each measurement
        """
        x, y, theta = pose

        # Add pose to trajectory
        self.trajectory.append(pose)

        # Convert robot position to map coordinates
        robot_row, robot_col = self.map.world_to_map(x, y)

        # Process each scan beam
        for range_val, angle in zip(scan, scan_angles):
            # Skip invalid measurements
            if range_val < self.params.min_range or range_val > self.params.max_range:
                continue

            # Calculate endpoint of beam
            beam_angle = theta + angle
            end_x = x + range_val * np.cos(beam_angle)
            end_y = y + range_val * np.sin(beam_angle)

            # Convert to map coordinates
            end_row, end_col = self.map.world_to_map(end_x, end_y)

            # Update map along ray
            self._update_ray(robot_row, robot_col, end_row, end_col, range_val)

    def _update_ray(self, x0: int, y0: int, x1: int, y1: int, range_val: float):
        """
        Update occupancy grid along a ray using Bresenham's line algorithm

        Args:
            x0, y0: Start point (robot position) in map coordinates
            x1, y1: End point (measurement endpoint) in map coordinates
            range_val: Range measurement value
        """
        # Bresenham's line algorithm
        points = self._bresenham_line(x0, y0, x1, y1)

        # Update cells along the ray
        for i, (row, col) in enumerate(points):
            if not self.map.is_valid(row, col):
                continue

            # Last point is the obstacle (if within max range)
            if i == len(points) - 1 and range_val < self.params.max_range * 0.95:
                # Mark as occupied
                self.map.update_occupancy(row, col, self.params.log_odds_occupied)
            else:
                # Mark as free
                self.map.update_occupancy(row, col, self.params.log_odds_free)

    def _bresenham_line(self, x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
        """
        Bresenham's line algorithm for ray tracing

        Args:
            x0, y0: Start point
            x1, y1: End point

        Returns:
            List of (row, col) points along the line
        """
        points = []

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)

        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1

        err = dx - dy

        x, y = x0, y0

        while True:
            points.append((x, y))

            if x == x1 and y == y1:
                break

            e2 = 2 * err

            if e2 > -dy:
                err -= dy
                x += sx

            if e2 < dx:
                err += dx
                y += sy

        return points

    def get_map(self) -> OccupancyGrid:
        """
        Get current occupancy grid map

        Returns:
            OccupancyGrid
        """
        return self.map

    def get_trajectory(self) -> List[Tuple[float, float, float]]:
        """
        Get robot trajectory

        Returns:
            List of poses (x, y, theta)
        """
        return self.trajectory

    def save_map(self, filename: str):
        """
        Save map to file

        Args:
            filename: Output filename
        """
        from ..utils.map_utils import save_map
        save_map(self.map, filename)

    def load_map(self, filename: str):
        """
        Load map from file

        Args:
            filename: Input filename
        """
        from ..utils.map_utils import load_map
        self.map = load_map(filename)

    def clear_map(self):
        """Clear the map"""
        self.map = OccupancyGrid(
            width=self.params.map_width,
            height=self.params.map_height,
            resolution=self.params.map_resolution,
            origin=self.params.map_origin
        )
        self.trajectory = []

    def inflate_obstacles(self, radius: float):
        """
        Inflate obstacles in the map (for robot footprint)

        Args:
            radius: Inflation radius in meters
        """
        # Convert radius to cells
        inflation_cells = int(np.ceil(radius / self.map.resolution))

        # Create copy of map data
        original_data = self.map.data.copy()

        # Inflate obstacles
        for row in range(self.map.height):
            for col in range(self.map.width):
                if original_data[row, col] > 50:  # Occupied cell
                    # Inflate in a square around the obstacle
                    for dr in range(-inflation_cells, inflation_cells + 1):
                        for dc in range(-inflation_cells, inflation_cells + 1):
                            # Check if within circular radius
                            if dr*dr + dc*dc <= inflation_cells*inflation_cells:
                                r, c = row + dr, col + dc
                                if self.map.is_valid(r, c):
                                    # Mark as occupied if not already
                                    if self.map.data[r, c] < 100:
                                        self.map.data[r, c] = min(100, self.map.data[r, c] + 50)

    def get_map_image(self) -> np.ndarray:
        """
        Get map as image (for visualization)

        Returns:
            2D numpy array (0=black/occupied, 255=white/free, 128=gray/unknown)
        """
        image = np.zeros_like(self.map.data, dtype=np.uint8)

        # Unknown cells -> gray (128)
        image[self.map.data < 0] = 128

        # Free cells -> white (255)
        image[self.map.data == 0] = 255

        # Occupied cells -> black (0)
        image[self.map.data > 50] = 0

        # Partially occupied -> gradient
        mask = (self.map.data >= 0) & (self.map.data <= 50)
        image[mask] = (255 - (self.map.data[mask] * 255 / 50)).astype(np.uint8)

        return image
