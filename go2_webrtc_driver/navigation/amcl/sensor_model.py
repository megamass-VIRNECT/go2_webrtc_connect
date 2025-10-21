"""
Sensor model for AMCL
Implements likelihood field model for laser range finder
"""

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
from ..utils.map_utils import OccupancyGrid


@dataclass
class SensorModelParams:
    """Sensor model parameters"""
    z_hit: float = 0.95       # Weight for correct range measurements
    z_rand: float = 0.05      # Weight for random measurements
    sigma_hit: float = 0.2    # Std dev for measurement noise (meters)
    max_range: float = 10.0   # Maximum sensor range (meters)
    min_range: float = 0.1    # Minimum sensor range (meters)
    max_beams: int = 100      # Maximum number of beams to use (subsampling)


class SensorModel:
    """
    Likelihood field sensor model for laser range finder
    Uses the beam_range_finder_model from Probabilistic Robotics
    """

    def __init__(self, params: SensorModelParams = None):
        """
        Initialize sensor model

        Args:
            params: Sensor model parameters
        """
        self.params = params if params is not None else SensorModelParams()
        self._likelihood_field_cache = {}

    def measurement_probability(self,
                                 pose: Tuple[float, float, float],
                                 scan: List[float],
                                 scan_angles: List[float],
                                 occupancy_map: OccupancyGrid) -> float:
        """
        Calculate measurement probability for a given pose

        Args:
            pose: Robot pose (x, y, theta)
            scan: List of range measurements
            scan_angles: List of angles for each measurement (relative to robot)
            occupancy_map: Occupancy grid map

        Returns:
            Measurement probability
        """
        x, y, theta = pose
        q = 1.0

        # Subsample beams if too many
        if len(scan) > self.params.max_beams:
            step = len(scan) // self.params.max_beams
            scan = scan[::step]
            scan_angles = scan_angles[::step]

        for z, angle in zip(scan, scan_angles):
            # Skip invalid measurements
            if z < self.params.min_range or z > self.params.max_range:
                continue

            # Calculate endpoint of beam in world frame
            beam_angle = theta + angle
            x_z = x + z * np.cos(beam_angle)
            y_z = y + z * np.sin(beam_angle)

            # Convert to map coordinates
            row, col = occupancy_map.world_to_map(x_z, y_z)

            # Calculate distance to nearest obstacle
            dist = self._get_nearest_obstacle_distance(row, col, occupancy_map)

            # Calculate probability using likelihood field
            p_hit = self._prob_hit(dist)
            p_rand = 1.0 / self.params.max_range

            # Combine probabilities
            p = self.params.z_hit * p_hit + self.params.z_rand * p_rand

            # Update total probability
            q *= p

        return q

    def _get_nearest_obstacle_distance(self, row: int, col: int, occupancy_map: OccupancyGrid) -> float:
        """
        Get distance to nearest obstacle from a map cell

        Args:
            row, col: Map cell indices
            occupancy_map: Occupancy grid map

        Returns:
            Distance to nearest obstacle in meters
        """
        # Check if already in cache
        cache_key = (row, col, id(occupancy_map))
        if cache_key in self._likelihood_field_cache:
            return self._likelihood_field_cache[cache_key]

        # If out of bounds, return max distance
        if not occupancy_map.is_valid(row, col):
            return self.params.max_range

        # Search in expanding squares for nearest obstacle
        max_search_radius = 50  # cells
        min_dist = self.params.max_range

        for radius in range(max_search_radius):
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    # Only check cells on the border of the square
                    if abs(dr) != radius and abs(dc) != radius:
                        continue

                    r = row + dr
                    c = col + dc

                    if occupancy_map.is_valid(r, c) and occupancy_map.is_occupied(r, c):
                        # Calculate distance
                        dist = np.sqrt(dr**2 + dc**2) * occupancy_map.resolution
                        min_dist = min(min_dist, dist)

            # Early exit if we found a close obstacle
            if min_dist < self.params.sigma_hit * 3:
                break

        # Cache result
        self._likelihood_field_cache[cache_key] = min_dist
        return min_dist

    def _prob_hit(self, dist: float) -> float:
        """
        Calculate probability of hit at given distance from obstacle

        Args:
            dist: Distance to nearest obstacle

        Returns:
            Probability value
        """
        # Gaussian centered at 0
        normalizer = 1.0 / (self.params.sigma_hit * np.sqrt(2 * np.pi))
        return normalizer * np.exp(-0.5 * (dist / self.params.sigma_hit) ** 2)

    def clear_cache(self):
        """Clear likelihood field cache"""
        self._likelihood_field_cache.clear()

    def ray_cast(self,
                 pose: Tuple[float, float, float],
                 angle: float,
                 occupancy_map: OccupancyGrid) -> float:
        """
        Cast a ray from pose at given angle and return range to obstacle

        Args:
            pose: Robot pose (x, y, theta)
            angle: Ray angle relative to robot heading
            occupancy_map: Occupancy grid map

        Returns:
            Range to obstacle in meters
        """
        x, y, theta = pose
        beam_angle = theta + angle

        # Ray casting parameters
        step_size = occupancy_map.resolution / 2.0
        max_steps = int(self.params.max_range / step_size)

        # Cast ray
        for step in range(max_steps):
            distance = step * step_size
            x_ray = x + distance * np.cos(beam_angle)
            y_ray = y + distance * np.sin(beam_angle)

            row, col = occupancy_map.world_to_map(x_ray, y_ray)

            # Check if hit obstacle or out of bounds
            if not occupancy_map.is_valid(row, col) or occupancy_map.is_occupied(row, col):
                return distance

        return self.params.max_range
