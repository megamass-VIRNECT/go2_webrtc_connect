"""
Pure Pursuit Controller
Follows a path by tracking a look-ahead point
"""

import numpy as np
from typing import List, Tuple, Optional


class PurePursuitController:
    """
    Pure Pursuit path following controller
    """

    def __init__(self,
                 look_ahead_distance: float = 0.5,
                 max_linear_velocity: float = 0.3,
                 max_angular_velocity: float = 0.8,
                 goal_tolerance: float = 0.2):
        """
        Initialize Pure Pursuit controller

        Args:
            look_ahead_distance: Distance to look ahead on path (meters)
            max_linear_velocity: Maximum forward speed (m/s)
            max_angular_velocity: Maximum rotation speed (rad/s)
            goal_tolerance: Distance to consider goal reached (meters)
        """
        self.look_ahead_distance = look_ahead_distance
        self.max_linear_velocity = max_linear_velocity
        self.max_angular_velocity = max_angular_velocity
        self.goal_tolerance = goal_tolerance

        self.path = None
        self.current_waypoint_index = 0

    def set_path(self, path: List[Tuple[float, float]]):
        """
        Set new path to follow

        Args:
            path: List of waypoints [(x, y), ...]
        """
        self.path = path
        self.current_waypoint_index = 0

    def compute_control(self,
                       current_pose: Tuple[float, float, float]) -> Tuple[float, float, bool]:
        """
        Compute control commands to follow path

        Args:
            current_pose: Current robot pose (x, y, theta)

        Returns:
            Tuple of (linear_velocity, angular_velocity, goal_reached)
        """
        if self.path is None or len(self.path) == 0:
            return 0.0, 0.0, True

        x, y, theta = current_pose

        # Find look-ahead point
        look_ahead_point = self._get_look_ahead_point((x, y))

        if look_ahead_point is None:
            # No more waypoints, check if goal reached
            goal_x, goal_y = self.path[-1]
            distance_to_goal = np.sqrt((x - goal_x)**2 + (y - goal_y)**2)

            if distance_to_goal < self.goal_tolerance:
                return 0.0, 0.0, True  # Goal reached
            else:
                # Use goal as look-ahead point
                look_ahead_point = self.path[-1]

        # Calculate angle to look-ahead point
        dx = look_ahead_point[0] - x
        dy = look_ahead_point[1] - y
        target_angle = np.arctan2(dy, dx)

        # Calculate angle error
        angle_error = self._normalize_angle(target_angle - theta)

        # Pure Pursuit formula
        # Angular velocity proportional to angle error and inversely proportional to look-ahead distance
        distance_to_look_ahead = np.sqrt(dx**2 + dy**2)

        # Linear velocity (reduce when turning)
        linear_velocity = self.max_linear_velocity * (1.0 - abs(angle_error) / np.pi)
        linear_velocity = max(0.1, linear_velocity)  # Minimum speed

        # Angular velocity (Pure Pursuit formula)
        if distance_to_look_ahead > 0.01:
            curvature = 2 * np.sin(angle_error) / distance_to_look_ahead
            angular_velocity = curvature * linear_velocity
        else:
            angular_velocity = 0.0

        # Clamp velocities
        linear_velocity = np.clip(linear_velocity, 0.0, self.max_linear_velocity)
        angular_velocity = np.clip(angular_velocity, -self.max_angular_velocity, self.max_angular_velocity)

        return linear_velocity, angular_velocity, False

    def _get_look_ahead_point(self, current_pos: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """
        Find look-ahead point on path

        Args:
            current_pos: Current position (x, y)

        Returns:
            Look-ahead point (x, y) or None
        """
        x, y = current_pos

        # Find closest waypoint ahead of current position
        min_distance = float('inf')
        closest_index = self.current_waypoint_index

        for i in range(self.current_waypoint_index, len(self.path)):
            wx, wy = self.path[i]
            distance = np.sqrt((x - wx)**2 + (y - wy)**2)

            if distance < min_distance:
                min_distance = distance
                closest_index = i

        # Update current waypoint
        self.current_waypoint_index = closest_index

        # Find look-ahead point
        for i in range(self.current_waypoint_index, len(self.path)):
            wx, wy = self.path[i]
            distance = np.sqrt((x - wx)**2 + (y - wy)**2)

            if distance >= self.look_ahead_distance:
                return (wx, wy)

        # No point found at look-ahead distance, return last point
        if len(self.path) > 0:
            return self.path[-1]

        return None

    def _normalize_angle(self, angle: float) -> float:
        """
        Normalize angle to [-pi, pi]

        Args:
            angle: Angle in radians

        Returns:
            Normalized angle
        """
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def get_progress(self) -> float:
        """
        Get progress along path (0.0 to 1.0)

        Returns:
            Progress percentage
        """
        if self.path is None or len(self.path) == 0:
            return 1.0

        return self.current_waypoint_index / len(self.path)

    def is_goal_reached(self, current_pos: Tuple[float, float]) -> bool:
        """
        Check if goal is reached

        Args:
            current_pos: Current position (x, y)

        Returns:
            True if goal reached
        """
        if self.path is None or len(self.path) == 0:
            return True

        goal_x, goal_y = self.path[-1]
        x, y = current_pos
        distance_to_goal = np.sqrt((x - goal_x)**2 + (y - goal_y)**2)

        return distance_to_goal < self.goal_tolerance
