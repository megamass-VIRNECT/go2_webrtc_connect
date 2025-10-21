"""
Motion model for AMCL
Implements odometry-based motion model with noise
"""

import numpy as np
from typing import Tuple
from dataclasses import dataclass


@dataclass
class MotionModelParams:
    """Motion model noise parameters"""
    alpha1: float = 0.1  # Rotation noise from rotation
    alpha2: float = 0.1  # Rotation noise from translation
    alpha3: float = 0.1  # Translation noise from translation
    alpha4: float = 0.1  # Translation noise from rotation


class MotionModel:
    """
    Odometry-based motion model for particle filter
    Uses the sample_motion_model_odometry algorithm from Probabilistic Robotics
    """

    def __init__(self, params: MotionModelParams = None):
        """
        Initialize motion model

        Args:
            params: Motion model noise parameters
        """
        self.params = params if params is not None else MotionModelParams()

    def sample_motion(self,
                      current_pose: Tuple[float, float, float],
                      odom_prev: Tuple[float, float, float],
                      odom_current: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """
        Sample new pose based on odometry motion

        Args:
            current_pose: Current pose (x, y, theta)
            odom_prev: Previous odometry reading (x, y, theta)
            odom_current: Current odometry reading (x, y, theta)

        Returns:
            New sampled pose (x, y, theta)
        """
        x, y, theta = current_pose
        x_prev, y_prev, theta_prev = odom_prev
        x_curr, y_curr, theta_curr = odom_current

        # Calculate odometry deltas
        delta_rot1 = np.arctan2(y_curr - y_prev, x_curr - x_prev) - theta_prev
        delta_trans = np.sqrt((x_curr - x_prev)**2 + (y_curr - y_prev)**2)
        delta_rot2 = theta_curr - theta_prev - delta_rot1

        # Normalize angles
        delta_rot1 = self._normalize_angle(delta_rot1)
        delta_rot2 = self._normalize_angle(delta_rot2)

        # Add noise to odometry
        delta_rot1_noisy = delta_rot1 - self._sample_normal(
            self.params.alpha1 * abs(delta_rot1) + self.params.alpha2 * delta_trans
        )
        delta_trans_noisy = delta_trans - self._sample_normal(
            self.params.alpha3 * delta_trans + self.params.alpha4 * (abs(delta_rot1) + abs(delta_rot2))
        )
        delta_rot2_noisy = delta_rot2 - self._sample_normal(
            self.params.alpha1 * abs(delta_rot2) + self.params.alpha2 * delta_trans
        )

        # Apply motion to current pose
        x_new = x + delta_trans_noisy * np.cos(theta + delta_rot1_noisy)
        y_new = y + delta_trans_noisy * np.sin(theta + delta_rot1_noisy)
        theta_new = theta + delta_rot1_noisy + delta_rot2_noisy

        # Normalize theta
        theta_new = self._normalize_angle(theta_new)

        return (x_new, y_new, theta_new)

    def _sample_normal(self, variance: float) -> float:
        """Sample from normal distribution with given variance"""
        if variance <= 0:
            return 0.0
        return np.random.normal(0, np.sqrt(variance))

    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def motion_likelihood(self,
                          pose: Tuple[float, float, float],
                          odom_prev: Tuple[float, float, float],
                          odom_current: Tuple[float, float, float],
                          actual_pose: Tuple[float, float, float]) -> float:
        """
        Calculate likelihood of reaching actual_pose from pose given odometry

        Args:
            pose: Starting pose (x, y, theta)
            odom_prev: Previous odometry reading (x, y, theta)
            odom_current: Current odometry reading (x, y, theta)
            actual_pose: Actual reached pose (x, y, theta)

        Returns:
            Likelihood value
        """
        x, y, theta = pose
        x_prev, y_prev, theta_prev = odom_prev
        x_curr, y_curr, theta_curr = odom_current
        x_act, y_act, theta_act = actual_pose

        # Calculate expected deltas from odometry
        delta_rot1 = np.arctan2(y_curr - y_prev, x_curr - x_prev) - theta_prev
        delta_trans = np.sqrt((x_curr - x_prev)**2 + (y_curr - y_prev)**2)
        delta_rot2 = theta_curr - theta_prev - delta_rot1

        # Calculate actual deltas
        delta_rot1_act = np.arctan2(y_act - y, x_act - x) - theta
        delta_trans_act = np.sqrt((x_act - x)**2 + (y_act - y)**2)
        delta_rot2_act = theta_act - theta - delta_rot1_act

        # Normalize angles
        delta_rot1 = self._normalize_angle(delta_rot1)
        delta_rot2 = self._normalize_angle(delta_rot2)
        delta_rot1_act = self._normalize_angle(delta_rot1_act)
        delta_rot2_act = self._normalize_angle(delta_rot2_act)

        # Calculate errors
        rot1_error = delta_rot1 - delta_rot1_act
        trans_error = delta_trans - delta_trans_act
        rot2_error = delta_rot2 - delta_rot2_act

        # Calculate likelihood using Gaussian
        var_rot1 = self.params.alpha1 * abs(delta_rot1) + self.params.alpha2 * delta_trans
        var_trans = self.params.alpha3 * delta_trans + self.params.alpha4 * (abs(delta_rot1) + abs(delta_rot2))
        var_rot2 = self.params.alpha1 * abs(delta_rot2) + self.params.alpha2 * delta_trans

        p1 = self._prob_normal(rot1_error, var_rot1)
        p2 = self._prob_normal(trans_error, var_trans)
        p3 = self._prob_normal(rot2_error, var_rot2)

        return p1 * p2 * p3

    def _prob_normal(self, x: float, variance: float) -> float:
        """Calculate Gaussian probability"""
        if variance <= 0:
            return 1.0 if abs(x) < 1e-6 else 0.0
        return (1.0 / np.sqrt(2 * np.pi * variance)) * np.exp(-0.5 * x**2 / variance)
