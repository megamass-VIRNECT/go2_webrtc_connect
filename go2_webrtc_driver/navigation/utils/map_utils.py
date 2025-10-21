"""
Map utilities for navigation
Handles occupancy grid map creation, loading, and saving
"""

import numpy as np
import json
import pickle
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class OccupancyGrid:
    """
    Occupancy grid map representation

    Attributes:
        data: 2D numpy array where 0=free, 100=occupied, -1=unknown
        resolution: Map resolution in meters/cell
        origin: Map origin (x, y) in meters
        width: Map width in cells
        height: Map height in cells
    """
    data: np.ndarray
    resolution: float
    origin: Tuple[float, float]

    @property
    def width(self) -> int:
        return self.data.shape[1]

    @property
    def height(self) -> int:
        return self.data.shape[0]

    def __init__(self, width: int, height: int, resolution: float = 0.05, origin: Tuple[float, float] = (0.0, 0.0)):
        """
        Initialize an occupancy grid map

        Args:
            width: Map width in cells
            height: Map height in cells
            resolution: Map resolution in meters/cell (default: 0.05m = 5cm)
            origin: Map origin (x, y) in meters
        """
        self.data = np.full((height, width), -1, dtype=np.int8)  # Unknown cells
        self.resolution = resolution
        self.origin = origin

    def world_to_map(self, x: float, y: float) -> Tuple[int, int]:
        """
        Convert world coordinates to map cell indices

        Args:
            x, y: World coordinates in meters

        Returns:
            (row, col) map cell indices
        """
        col = int((x - self.origin[0]) / self.resolution)
        row = int((y - self.origin[1]) / self.resolution)
        return (row, col)

    def map_to_world(self, row: int, col: int) -> Tuple[float, float]:
        """
        Convert map cell indices to world coordinates

        Args:
            row, col: Map cell indices

        Returns:
            (x, y) world coordinates in meters
        """
        x = col * self.resolution + self.origin[0]
        y = row * self.resolution + self.origin[1]
        return (x, y)

    def is_valid(self, row: int, col: int) -> bool:
        """Check if cell indices are within map bounds"""
        return 0 <= row < self.height and 0 <= col < self.width

    def is_occupied(self, row: int, col: int, threshold: int = 50) -> bool:
        """
        Check if a cell is occupied

        Args:
            row, col: Map cell indices
            threshold: Occupancy threshold (default: 50)

        Returns:
            True if occupied, False otherwise
        """
        if not self.is_valid(row, col):
            return True  # Out of bounds is considered occupied
        return self.data[row, col] > threshold

    def is_free(self, row: int, col: int, threshold: int = 50) -> bool:
        """
        Check if a cell is free

        Args:
            row, col: Map cell indices
            threshold: Occupancy threshold (default: 50)

        Returns:
            True if free, False otherwise
        """
        if not self.is_valid(row, col):
            return False
        return self.data[row, col] >= 0 and self.data[row, col] < threshold

    def get_occupancy(self, row: int, col: int) -> int:
        """Get occupancy value at cell"""
        if not self.is_valid(row, col):
            return 100  # Out of bounds
        return self.data[row, col]

    def set_occupied(self, row: int, col: int):
        """Mark cell as occupied"""
        if self.is_valid(row, col):
            self.data[row, col] = 100

    def set_free(self, row: int, col: int):
        """Mark cell as free"""
        if self.is_valid(row, col):
            self.data[row, col] = 0

    def update_occupancy(self, row: int, col: int, log_odds_update: float):
        """
        Update cell occupancy using log-odds

        Args:
            row, col: Map cell indices
            log_odds_update: Log-odds update value
        """
        if not self.is_valid(row, col):
            return

        # Convert current occupancy to log-odds
        if self.data[row, col] < 0:
            current_log_odds = 0.0
        else:
            prob = self.data[row, col] / 100.0
            prob = np.clip(prob, 0.01, 0.99)  # Avoid log(0)
            current_log_odds = np.log(prob / (1 - prob))

        # Update log-odds
        new_log_odds = current_log_odds + log_odds_update

        # Convert back to probability
        new_prob = 1.0 / (1.0 + np.exp(-new_log_odds))

        # Convert to occupancy value (0-100)
        self.data[row, col] = int(np.clip(new_prob * 100, 0, 100))


def save_map(occupancy_grid: OccupancyGrid, filename: str):
    """
    Save occupancy grid map to file

    Args:
        occupancy_grid: OccupancyGrid to save
        filename: Output filename (supports .pkl, .npz, .json)
    """
    if filename.endswith('.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump(occupancy_grid, f)
    elif filename.endswith('.npz'):
        np.savez(filename,
                 data=occupancy_grid.data,
                 resolution=occupancy_grid.resolution,
                 origin=occupancy_grid.origin)
    elif filename.endswith('.json'):
        map_dict = {
            'data': occupancy_grid.data.tolist(),
            'resolution': occupancy_grid.resolution,
            'origin': list(occupancy_grid.origin),
            'width': occupancy_grid.width,
            'height': occupancy_grid.height
        }
        with open(filename, 'w') as f:
            json.dump(map_dict, f)
    else:
        raise ValueError(f"Unsupported file format: {filename}")


def load_map(filename: str) -> OccupancyGrid:
    """
    Load occupancy grid map from file

    Args:
        filename: Input filename (supports .pkl, .npz, .json)

    Returns:
        Loaded OccupancyGrid
    """
    if filename.endswith('.pkl'):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    elif filename.endswith('.npz'):
        data = np.load(filename)
        grid = OccupancyGrid(
            width=data['data'].shape[1],
            height=data['data'].shape[0],
            resolution=float(data['resolution']),
            origin=tuple(data['origin'])
        )
        grid.data = data['data']
        return grid
    elif filename.endswith('.json'):
        with open(filename, 'r') as f:
            map_dict = json.load(f)
        grid = OccupancyGrid(
            width=map_dict['width'],
            height=map_dict['height'],
            resolution=map_dict['resolution'],
            origin=tuple(map_dict['origin'])
        )
        grid.data = np.array(map_dict['data'], dtype=np.int8)
        return grid
    else:
        raise ValueError(f"Unsupported file format: {filename}")
