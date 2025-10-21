"""
SLAM Mapping Example for Go2 Robot
Creates an occupancy grid map using Lidar data
"""

import asyncio
import logging
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod
from go2_webrtc_driver.load_webrtc_config import load_webrtc_config
from go2_webrtc_driver.navigation import SLAM, SLAMParams

# Enable logging
logging.basicConfig(level=logging.INFO)


class SLAMMapper:
    """SLAM mapper for Go2 robot"""

    def __init__(self, enable_visualization=True):
        # Initialize SLAM
        slam_params = SLAMParams(
            map_width=2000,
            map_height=2000,
            map_resolution=0.1,  # 10cm resolution for faster processing
            map_origin=(-100.0, -100.0),  # 200m x 200m map
            max_range=50.0,  # Reduced for performance
            min_range=0.5
        )
        self.slam = SLAM(slam_params)

        # Current pose estimate (x, y, theta)
        self.current_pose = (0.0, 0.0, 0.0)

        # Previous odometry
        self.prev_odom = None

        # Statistics
        self.scan_count = 0

        # Visualization
        self.enable_visualization = enable_visualization
        self.fig = None
        self.ax = None
        self.map_plot = None
        self.trajectory_plot = None

        if self.enable_visualization:
            self._setup_visualization()

    def lowstate_callback(self, message):
        """Handle lowstate updates (IMU data only - no position)"""
        try:
            if not hasattr(self, 'odom_count'):
                self.odom_count = 0
            self.odom_count += 1

            if self.odom_count % 100 == 0:
                logging.info(f"Received {self.odom_count} lowstate messages")

            data = message.get("data", {})

            # Debug: Print message structure on first call
            if self.prev_odom is None:
                logging.info(f"LowState message keys: {list(message.keys())}")
                logging.info(f"LowState data keys: {list(data.keys())}")

            # Extract IMU state (only yaw is reliable)
            imu_state = data.get("imu_state", {})
            rpy = imu_state.get("rpy", [0.0, 0.0, 0.0])

            # Debug: Print first IMU data
            if self.prev_odom is None:
                logging.info(f"First IMU - rpy: {rpy}")
                logging.warning("WARNING: No position data available in rt/lf/lowstate!")
                logging.warning("SLAM will only track orientation, not robot position.")
                logging.warning("The map will show scans from a fixed location with rotating direction.")

            # Current yaw only (no position data available)
            current_yaw = rpy[2]

            # Update pose estimate
            if self.prev_odom is not None:
                # Only update orientation, position stays at (0, 0)
                dtheta = current_yaw - self.prev_odom

                # Normalize angle difference
                while dtheta > np.pi:
                    dtheta -= 2 * np.pi
                while dtheta < -np.pi:
                    dtheta += 2 * np.pi

                # Debug: Print movement
                if abs(dtheta) > 0.05:  # 约3度
                    logging.info(f"Robot rotated: dtheta={dtheta:.3f} rad ({np.degrees(dtheta):.1f} deg)")

                # Update current pose (only theta changes)
                x, y, theta = self.current_pose
                theta += dtheta

                # Normalize theta
                theta = np.arctan2(np.sin(theta), np.cos(theta))

                self.current_pose = (x, y, theta)

                if abs(dtheta) > 0.05:
                    logging.info(f"New orientation: theta={theta:.3f} rad ({np.degrees(theta):.1f} deg)")

            self.prev_odom = current_yaw

        except Exception as e:
            logging.error(f"Error in lowstate callback: {e}", exc_info=True)

    def sportmodestate_callback(self, message):
        """Handle sport mode state updates (odometry)"""
        try:
            if not hasattr(self, 'odom_count'):
                self.odom_count = 0
            self.odom_count += 1

            if self.odom_count % 100 == 0:
                logging.info(f"Received {self.odom_count} odometry messages")

            data = message.get("data", {})

            # Debug: Print message structure on first call
            if self.prev_odom is None:
                logging.info(f"SportMode message keys: {list(message.keys())}")
                logging.info(f"SportMode data keys: {list(data.keys())}")

            # Extract position and orientation
            position = data.get("position", [0.0, 0.0, 0.0])
            imu_state = data.get("imu_state", {})
            rpy = imu_state.get("rpy", [0.0, 0.0, 0.0])

            # Debug: Print first odometry data
            if self.prev_odom is None:
                logging.info(f"First odometry - position: {position}, rpy: {rpy}")

            # Current odometry (x, y, yaw)
            current_odom = (position[0], position[1], rpy[2])

            # Update pose estimate
            if self.prev_odom is not None:
                # Simple odometry integration
                dx = current_odom[0] - self.prev_odom[0]
                dy = current_odom[1] - self.prev_odom[1]
                dtheta = current_odom[2] - self.prev_odom[2]

                # Debug: Print movement
                if abs(dx) > 0.01 or abs(dy) > 0.01 or abs(dtheta) > 0.01:
                    logging.info(f"Robot moved: dx={dx:.3f}, dy={dy:.3f}, dtheta={dtheta:.3f}")

                # Update current pose
                x, y, theta = self.current_pose
                x += dx * np.cos(theta) - dy * np.sin(theta)
                y += dx * np.sin(theta) + dy * np.cos(theta)
                theta += dtheta

                # Normalize theta
                theta = np.arctan2(np.sin(theta), np.cos(theta))

                self.current_pose = (x, y, theta)

                if abs(dx) > 0.01 or abs(dy) > 0.01:
                    logging.info(f"New pose: x={x:.3f}, y={y:.3f}, theta={theta:.3f}")

            self.prev_odom = current_odom

        except Exception as e:
            logging.error(f"Error in sportmodestate callback: {e}", exc_info=True)

    def _setup_visualization(self):
        """Setup matplotlib visualization"""
        plt.ion()  # Enable interactive mode
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.set_xlabel('Y (meters)')
        self.ax.set_ylabel('X (meters)')
        self.ax.set_title('Real-time SLAM Map')
        self.ax.grid(True, alpha=0.3)

        # Initialize empty plot
        map_image = self.slam.get_map_image()
        extent = [
            self.slam.map.origin[1],
            self.slam.map.origin[1] + self.slam.map.width * self.slam.map.resolution,
            self.slam.map.origin[0],
            self.slam.map.origin[0] + self.slam.map.height * self.slam.map.resolution
        ]
        self.map_plot = self.ax.imshow(map_image, cmap='gray', origin='lower',
                                        extent=extent, vmin=0, vmax=255, alpha=0.8)

        # Initialize trajectory plot
        self.trajectory_plot, = self.ax.plot([], [], 'r-', linewidth=2, label='Trajectory')
        self.robot_plot, = self.ax.plot([], [], 'bo', markersize=10, label='Robot')
        self.ax.legend()

        plt.tight_layout()
        plt.show(block=False)

    def update_visualization(self):
        """Update the visualization with latest map and trajectory"""
        if not self.enable_visualization or self.map_plot is None:
            return

        try:
            # Update map
            map_image = self.slam.get_map_image()
            self.map_plot.set_data(map_image)

            # Update trajectory
            trajectory = self.slam.get_trajectory()
            if trajectory:
                traj_x = [p[0] for p in trajectory]
                traj_y = [p[1] for p in trajectory]
                self.trajectory_plot.set_data(traj_y, traj_x)

            # Update robot position
            x, y, theta = self.current_pose
            self.robot_plot.set_data([y], [x])

            # Update plot
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

        except Exception as e:
            logging.error(f"Error updating visualization: {e}")

    def lidar_callback(self, message):
        """Handle LIDAR scan updates"""
        try:
            data = message.get("data", {})

            # Debug: Print message structure on first call
            if self.scan_count == 0:
                logging.info(f"LIDAR message keys: {list(message.keys())}")
                logging.info(f"LIDAR data keys: {list(data.keys())}")

            # Extract positions from voxel map compressed data
            data_inner = data.get("data", {})

            # Debug: Print inner data structure
            if self.scan_count == 0:
                logging.info(f"Inner data type: {type(data_inner)}")
                if isinstance(data_inner, dict):
                    logging.info(f"Inner data keys: {list(data_inner.keys())}")

            positions = data_inner.get("positions", [])

            # Check if positions is valid
            if positions is None or (hasattr(positions, '__len__') and len(positions) == 0):
                logging.warning(f"No LIDAR points received (scan {self.scan_count})")
                return

            # Convert to numpy array if it isn't already
            if not isinstance(positions, np.ndarray):
                positions = np.array(positions, dtype=np.float32)

            # Convert positions list to 3D points
            points = np.array([positions[i:i+3] for i in range(0, len(positions), 3)], dtype=np.float32)

            # Get map origin to convert to robot-centered coordinates
            origin = data.get("origin", [0.0, 0.0, 0.0])
            width = data.get("width", [128, 128, 38])
            resolution = data.get("resolution", 0.05)

            # Calculate map center (this is roughly where the robot is)
            map_center_x = origin[0] + (width[0] * resolution) / 2.0
            map_center_y = origin[1] + (width[1] * resolution) / 2.0

            # Debug: Print point cloud info
            if self.scan_count == 0:
                logging.info(f"First scan: {len(points)} points")
                logging.info(f"Map origin: {origin}")
                logging.info(f"Map center (robot approx): ({map_center_x:.2f}, {map_center_y:.2f})")
                if len(points) > 0:
                    logging.info(f"First point example (absolute): {points[0]}")
                    logging.info(f"Point range (absolute): min={points.min(axis=0)}, max={points.max(axis=0)}")

            # Convert to robot-centered coordinates
            points[:, 0] -= map_center_x
            points[:, 1] -= map_center_y

            if self.scan_count == 0 and len(points) > 0:
                logging.info(f"First point (robot-centered): {points[0]}")
                logging.info(f"Point range (robot-centered): min={points.min(axis=0)}, max={points.max(axis=0)}")

            # Filter out points that are too far or too close
            ranges_temp = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
            valid_mask = (ranges_temp >= self.slam.params.min_range) & (ranges_temp <= self.slam.params.max_range)

            if self.scan_count % 10 == 0:
                logging.info(f"Valid points: {np.sum(valid_mask)}/{len(points)} (filtered {len(points) - np.sum(valid_mask)})")

            points_filtered = points[valid_mask]

            if len(points_filtered) == 0:
                logging.warning(f"All points filtered out (scan {self.scan_count})")
                return

            # Downsample points aggressively for performance
            downsample_factor = max(1, len(points_filtered) // 500)  # Limit to ~500 points
            points_filtered = points_filtered[::downsample_factor]

            if self.scan_count % 10 == 0:
                logging.info(f"After downsampling: {len(points_filtered)} points (factor: {downsample_factor})")

            # Convert point cloud to polar coordinates (range, angle) - vectorized for speed
            x = points_filtered[:, 0]
            y = points_filtered[:, 1]
            ranges = np.sqrt(x**2 + y**2).tolist()
            angles = np.arctan2(y, x).tolist()

            # Debug: Print range statistics
            if self.scan_count % 10 == 0:
                logging.info(f"Scan {self.scan_count}: {len(ranges)} points, "
                           f"range: [{min(ranges):.2f}, {max(ranges):.2f}]m")

            # Update SLAM map
            if self.scan_count == 0:
                logging.info(f"Starting SLAM update with {len(ranges)} points...")

            self.slam.update(self.current_pose, ranges, angles)

            if self.scan_count == 0:
                logging.info(f"First SLAM update completed!")
                # Debug map state
                map_data = self.slam.map.data
                logging.info(f"Map data range: min={map_data.min()}, max={map_data.max()}, mean={map_data.mean():.2f}")
                logging.info(f"Unknown cells (-1): {np.sum(map_data == -1)}")
                logging.info(f"Free cells (0): {np.sum(map_data == 0)}")
                logging.info(f"Occupied cells (>50): {np.sum(map_data > 50)}")

            self.scan_count += 1

            # Update visualization every 20 scans to reduce overhead
            if self.scan_count % 20 == 0:
                self.update_visualization()
                logging.info(f"Processed {self.scan_count} scans. Current pose: "
                           f"x={self.current_pose[0]:.2f}, y={self.current_pose[1]:.2f}, "
                           f"theta={self.current_pose[2]:.2f}")

                # Debug map state periodically
                map_data = self.slam.map.data
                logging.info(f"Map stats: unknown={np.sum(map_data == -1)}, "
                           f"free={np.sum(map_data < 50) - np.sum(map_data == -1)}, "
                           f"occupied={np.sum(map_data > 50)}")

            # Save map every 50 scans for safety
            if self.scan_count % 50 == 0 and self.scan_count > 0:
                map_filename = f"go2_map_backup_{self.scan_count}.pkl"
                self.slam.save_map(map_filename)
                logging.info(f"Backup map saved to {map_filename}")

        except Exception as e:
            logging.error(f"Error in lidar callback: {e}", exc_info=True)


async def main():
    try:
        # Connection setup
        # Uncomment the appropriate connection method:
        conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip="192.168.1.7")
        # config = load_webrtc_config()
        # conn = Go2WebRTCConnection(WebRTCConnectionMethod.Remote,
        #                           serialNumber=config["sn"],
        #                           username=config["user"]["email"],
        #                           password=config["user"]["password"])

        # Connect to the WebRTC service
        logging.info("Connecting to Go2...")
        await conn.connect()
        logging.info("Connected successfully!")

        # Disable traffic saving mode
        await conn.datachannel.disableTrafficSaving(True)

        # Set decoder type
        conn.datachannel.set_decoder(decoder_type='libvoxel')

        # Initialize SLAM mapper (disable visualization for debugging)
        mapper = SLAMMapper(enable_visualization=False)

        # Subscribe to sportmodestate (for position and orientation data)
        conn.datachannel.pub_sub.subscribe("rt/lf/sportmodestate", mapper.sportmodestate_callback)
        logging.info("Subscribed to rt/lf/sportmodestate for odometry data")

        # Add a counter to check if odometry is being received
        mapper.odom_count = 0

        # Turn on LIDAR
        conn.datachannel.pub_sub.publish_without_callback("rt/utlidar/switch", "on")
        logging.info("LIDAR turned on")

        # Subscribe to LIDAR data
        conn.datachannel.pub_sub.subscribe("rt/utlidar/voxel_map_compressed", mapper.lidar_callback)
        logging.info("Subscribed to LIDAR data")

        # Run for specified duration
        duration = 60  # 1 minute
        logging.info(f"Mapping for {duration} seconds... (Press Ctrl+C to stop early)")

        try:
            await asyncio.sleep(duration)
        except KeyboardInterrupt:
            logging.info("Mapping interrupted by user")

        # Save the map
        map_filename = "go2_map.pkl"
        mapper.slam.save_map(map_filename)
        logging.info(f"Map saved to {map_filename}")

        # Print statistics
        logging.info(f"\nMapping complete!")
        logging.info(f"Total scans processed: {mapper.scan_count}")
        logging.info(f"Final pose: x={mapper.current_pose[0]:.2f}, "
                    f"y={mapper.current_pose[1]:.2f}, "
                    f"theta={mapper.current_pose[2]:.2f}")
        logging.info(f"Trajectory points: {len(mapper.slam.get_trajectory())}")

        # Save final visualization
        if mapper.enable_visualization and mapper.fig is not None:
            mapper.fig.savefig('slam_map_final.png', dpi=150, bbox_inches='tight')
            logging.info("Final map visualization saved to slam_map_final.png")

            # Keep window open
            logging.info("Close the visualization window to exit")
            plt.ioff()
            plt.show()

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
        sys.exit(0)
