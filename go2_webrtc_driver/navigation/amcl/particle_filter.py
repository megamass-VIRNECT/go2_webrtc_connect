"""
Particle Filter implementation for AMCL
Implements Adaptive Monte Carlo Localization using particle filtering
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from ..utils.map_utils import OccupancyGrid
from .motion_model import MotionModel, MotionModelParams
from .sensor_model import SensorModel, SensorModelParams


@dataclass
class Particle:
    """Single particle representing a pose hypothesis"""
    x: float
    y: float
    theta: float
    weight: float = 1.0

    def to_tuple(self) -> Tuple[float, float, float]:
        """Convert to tuple (x, y, theta)"""
        return (self.x, self.y, self.theta)


class ParticleFilter:
    """
    Particle filter for robot localization
    """

    def __init__(self,
                 num_particles: int = 500,
                 motion_model: MotionModel = None,
                 sensor_model: SensorModel = None):
        """
        Initialize particle filter

        Args:
            num_particles: Number of particles to use
            motion_model: Motion model (optional, creates default if None)
            sensor_model: Sensor model (optional, creates default if None)
        """
        self.num_particles = num_particles
        self.particles: List[Particle] = []
        self.motion_model = motion_model if motion_model else MotionModel()
        self.sensor_model = sensor_model if sensor_model else SensorModel()

        # Adaptive sampling parameters
        self.alpha_slow = 0.001  # Slow average decay rate
        self.alpha_fast = 0.1    # Fast average decay rate
        self.w_slow = 0.0        # Slow average weight
        self.w_fast = 0.0        # Fast average weight

    def initialize_uniform(self, occupancy_map: OccupancyGrid):
        """
        Initialize particles uniformly in free space

        Args:
            occupancy_map: Occupancy grid map
        """
        self.particles = []
        attempts = 0
        max_attempts = self.num_particles * 100

        while len(self.particles) < self.num_particles and attempts < max_attempts:
            # Random position in map
            x = np.random.uniform(
                occupancy_map.origin[0],
                occupancy_map.origin[0] + occupancy_map.width * occupancy_map.resolution
            )
            y = np.random.uniform(
                occupancy_map.origin[1],
                occupancy_map.origin[1] + occupancy_map.height * occupancy_map.resolution
            )
            theta = np.random.uniform(-np.pi, np.pi)

            # Check if position is in free space
            row, col = occupancy_map.world_to_map(x, y)
            if occupancy_map.is_free(row, col):
                self.particles.append(Particle(x, y, theta, 1.0 / self.num_particles))

            attempts += 1

        # Normalize weights
        self._normalize_weights()

    def initialize_gaussian(self, mean_pose: Tuple[float, float, float],
                           std_xy: float = 0.5, std_theta: float = 0.3):
        """
        Initialize particles with Gaussian distribution around mean pose

        Args:
            mean_pose: Mean pose (x, y, theta)
            std_xy: Standard deviation for x and y (meters)
            std_theta: Standard deviation for theta (radians)
        """
        self.particles = []
        x_mean, y_mean, theta_mean = mean_pose

        for _ in range(self.num_particles):
            x = np.random.normal(x_mean, std_xy)
            y = np.random.normal(y_mean, std_xy)
            theta = np.random.normal(theta_mean, std_theta)
            # Normalize theta to [-pi, pi]
            theta = np.arctan2(np.sin(theta), np.cos(theta))

            self.particles.append(Particle(x, y, theta, 1.0 / self.num_particles))

    def predict(self, odom_prev: Tuple[float, float, float],
                odom_current: Tuple[float, float, float]):
        """
        Prediction step: propagate particles using motion model

        Args:
            odom_prev: Previous odometry reading (x, y, theta)
            odom_current: Current odometry reading (x, y, theta)
        """
        for particle in self.particles:
            new_pose = self.motion_model.sample_motion(
                particle.to_tuple(),
                odom_prev,
                odom_current
            )
            particle.x, particle.y, particle.theta = new_pose

    def update(self, scan: List[float], scan_angles: List[float],
               occupancy_map: OccupancyGrid):
        """
        Update step: update particle weights based on sensor measurements

        Args:
            scan: List of range measurements
            scan_angles: List of angles for each measurement
            occupancy_map: Occupancy grid map
        """
        # Update weights for each particle
        for particle in self.particles:
            weight = self.sensor_model.measurement_probability(
                particle.to_tuple(),
                scan,
                scan_angles,
                occupancy_map
            )
            particle.weight = weight

        # Normalize weights
        self._normalize_weights()

        # Update average weights for adaptive resampling
        w_avg = np.mean([p.weight for p in self.particles])
        self.w_slow = self.w_slow + self.alpha_slow * (w_avg - self.w_slow)
        self.w_fast = self.w_fast + self.alpha_fast * (w_avg - self.w_fast)

    def resample(self):
        """
        Resample particles using low-variance resampling
        Implements adaptive resampling to handle kidnapping problem
        """
        # Calculate random sample probability
        p_random = max(0.0, 1.0 - self.w_fast / self.w_slow) if self.w_slow > 0 else 0.0

        new_particles = []
        weights = [p.weight for p in self.particles]

        # Low variance resampling
        M = len(self.particles)
        r = np.random.uniform(0, 1.0 / M)
        c = weights[0]
        i = 0

        for m in range(M):
            # Adaptive sampling: add random particles
            if np.random.random() < p_random:
                # Add random particle (for handling kidnapping)
                theta = np.random.uniform(-np.pi, np.pi)
                # Use random existing particle position with random orientation
                random_particle = self.particles[np.random.randint(0, M)]
                new_particles.append(Particle(
                    random_particle.x + np.random.normal(0, 0.5),
                    random_particle.y + np.random.normal(0, 0.5),
                    theta,
                    1.0 / M
                ))
            else:
                # Low variance resampling
                U = r + m / M
                while U > c:
                    i = (i + 1) % M
                    c = c + weights[i]

                # Add resampled particle (with small noise to avoid depletion)
                p = self.particles[i]
                new_particles.append(Particle(
                    p.x + np.random.normal(0, 0.02),
                    p.y + np.random.normal(0, 0.02),
                    p.theta + np.random.normal(0, 0.05),
                    1.0 / M
                ))

        self.particles = new_particles

    def get_estimated_pose(self) -> Tuple[float, float, float]:
        """
        Get estimated pose as weighted mean of particles

        Returns:
            Estimated pose (x, y, theta)
        """
        if not self.particles:
            return (0.0, 0.0, 0.0)

        weights = np.array([p.weight for p in self.particles])
        weights = weights / np.sum(weights)  # Normalize

        # Weighted mean for x and y
        x = np.sum([p.x * w for p, w in zip(self.particles, weights)])
        y = np.sum([p.y * w for p, w in zip(self.particles, weights)])

        # Circular mean for theta
        sin_sum = np.sum([np.sin(p.theta) * w for p, w in zip(self.particles, weights)])
        cos_sum = np.sum([np.cos(p.theta) * w for p, w in zip(self.particles, weights)])
        theta = np.arctan2(sin_sum, cos_sum)

        return (x, y, theta)

    def get_pose_covariance(self) -> np.ndarray:
        """
        Calculate pose covariance matrix

        Returns:
            3x3 covariance matrix
        """
        mean_pose = self.get_estimated_pose()
        x_mean, y_mean, theta_mean = mean_pose

        cov = np.zeros((3, 3))
        weights = np.array([p.weight for p in self.particles])
        weights = weights / np.sum(weights)

        for p, w in zip(self.particles, weights):
            dx = p.x - x_mean
            dy = p.y - y_mean
            dtheta = np.arctan2(np.sin(p.theta - theta_mean), np.cos(p.theta - theta_mean))

            diff = np.array([dx, dy, dtheta])
            cov += w * np.outer(diff, diff)

        return cov

    def _normalize_weights(self):
        """Normalize particle weights to sum to 1"""
        total_weight = sum(p.weight for p in self.particles)
        if total_weight > 0:
            for p in self.particles:
                p.weight /= total_weight
        else:
            # All weights are zero, reset to uniform
            for p in self.particles:
                p.weight = 1.0 / len(self.particles)

    def effective_sample_size(self) -> float:
        """
        Calculate effective sample size (ESS)

        Returns:
            Effective sample size
        """
        weights = [p.weight for p in self.particles]
        return 1.0 / np.sum([w**2 for w in weights])


class AMCL:
    """
    Adaptive Monte Carlo Localization
    High-level interface for particle filter-based localization
    """

    def __init__(self,
                 num_particles: int = 500,
                 motion_params: MotionModelParams = None,
                 sensor_params: SensorModelParams = None,
                 resample_threshold: float = 0.5):
        """
        Initialize AMCL

        Args:
            num_particles: Number of particles
            motion_params: Motion model parameters
            sensor_params: Sensor model parameters
            resample_threshold: Resample when ESS drops below this fraction of num_particles
        """
        motion_model = MotionModel(motion_params)
        sensor_model = SensorModel(sensor_params)

        self.particle_filter = ParticleFilter(num_particles, motion_model, sensor_model)
        self.resample_threshold = resample_threshold * num_particles

        self.initialized = False
        self.last_odom: Optional[Tuple[float, float, float]] = None

    def initialize(self, occupancy_map: OccupancyGrid,
                   initial_pose: Optional[Tuple[float, float, float]] = None):
        """
        Initialize AMCL

        Args:
            occupancy_map: Occupancy grid map
            initial_pose: Initial pose (optional, uses uniform if None)
        """
        if initial_pose is not None:
            self.particle_filter.initialize_gaussian(initial_pose)
        else:
            self.particle_filter.initialize_uniform(occupancy_map)

        self.initialized = True

    def update(self,
               odom: Tuple[float, float, float],
               scan: List[float],
               scan_angles: List[float],
               occupancy_map: OccupancyGrid) -> Tuple[float, float, float]:
        """
        Update AMCL with new odometry and laser scan

        Args:
            odom: Current odometry reading (x, y, theta)
            scan: Laser scan ranges
            scan_angles: Laser scan angles
            occupancy_map: Occupancy grid map

        Returns:
            Estimated pose (x, y, theta)
        """
        if not self.initialized:
            raise RuntimeError("AMCL not initialized. Call initialize() first.")

        # Prediction step (if we have previous odometry)
        if self.last_odom is not None:
            self.particle_filter.predict(self.last_odom, odom)

        # Update step
        self.particle_filter.update(scan, scan_angles, occupancy_map)

        # Resample if effective sample size is too low
        ess = self.particle_filter.effective_sample_size()
        if ess < self.resample_threshold:
            self.particle_filter.resample()

        # Store odometry for next iteration
        self.last_odom = odom

        # Return estimated pose
        return self.particle_filter.get_estimated_pose()

    def get_pose(self) -> Tuple[float, float, float]:
        """Get current estimated pose"""
        return self.particle_filter.get_estimated_pose()

    def get_covariance(self) -> np.ndarray:
        """Get current pose covariance"""
        return self.particle_filter.get_pose_covariance()

    def get_particles(self) -> List[Particle]:
        """Get current particles"""
        return self.particle_filter.particles
