"""
Transform utilities for navigation
Handles coordinate transformations and conversions
"""

import numpy as np
from typing import Tuple


def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
    """
    Convert Euler angles to quaternion

    Args:
        roll: Roll angle in radians
        pitch: Pitch angle in radians
        yaw: Yaw angle in radians

    Returns:
        Tuple of (x, y, z, w) quaternion components
    """
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return (x, y, z, w)


def quaternion_to_euler(x: float, y: float, z: float, w: float) -> Tuple[float, float, float]:
    """
    Convert quaternion to Euler angles

    Args:
        x, y, z, w: Quaternion components

    Returns:
        Tuple of (roll, pitch, yaw) in radians
    """
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return (roll, pitch, yaw)


def transform_point(point: np.ndarray, pose: Tuple[float, float, float]) -> np.ndarray:
    """
    Transform a point from robot frame to map frame

    Args:
        point: 2D point [x, y] in robot frame
        pose: Robot pose (x, y, theta) in map frame

    Returns:
        Transformed point in map frame
    """
    x, y, theta = pose
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Rotation matrix
    R = np.array([
        [cos_theta, -sin_theta],
        [sin_theta, cos_theta]
    ])

    # Apply rotation and translation
    transformed = R @ point + np.array([x, y])
    return transformed


def inverse_transform_point(point: np.ndarray, pose: Tuple[float, float, float]) -> np.ndarray:
    """
    Transform a point from map frame to robot frame

    Args:
        point: 2D point [x, y] in map frame
        pose: Robot pose (x, y, theta) in map frame

    Returns:
        Transformed point in robot frame
    """
    x, y, theta = pose
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Inverse rotation matrix
    R_inv = np.array([
        [cos_theta, sin_theta],
        [-sin_theta, cos_theta]
    ])

    # Apply inverse translation and rotation
    translated = point - np.array([x, y])
    transformed = R_inv @ translated
    return transformed
