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
            map_width=800,
            map_height=800,
            map_resolution=0.05,  # 5cm resolution for finer detail
            map_origin=(-20.0, -20.0),  # 40m x 40m map (800 cells × 0.05m = 40m)
            max_range=3.0,  # Maximum lidar range (limited to avoid voxel map boundary artifacts)
            min_range=0.5,
            log_odds_occupied=1.2,  # Balanced: strong enough to detect obstacles, not too aggressive
            log_odds_free=-0.5  # Balanced: clears noise but doesn't erase real obstacles
        )
        self.slam = SLAM(slam_params)

        # Current pose estimate (x, y, theta)
        self.current_pose = (0.0, 0.0, 0.0)

        # Previous odometry
        self.prev_odom = None

        # Odometry origin (first odometry position becomes our origin)
        self.odom_origin = None

        # Statistics
        self.scan_count = 0

        # Visualization
        self.enable_visualization = enable_visualization
        self.fig = None
        self.ax = None
        self.map_plot = None
        self.trajectory_plot = None
        # Limit LIDAR data to ±60° (120° forward field of view)
        self.front_angle_limit = np.radians(60.0)

        if self.enable_visualization:
            self._setup_visualization()

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

            # Current odometry (x, y, yaw) - position is in world frame
            current_odom = (position[0], position[1], rpy[2])

            # Initialize odometry origin on first message
            if self.odom_origin is None:
                self.odom_origin = current_odom
                self.current_pose = (0.0, 0.0, 0.0)
                logging.info(f"Odometry origin set: {self.odom_origin}")
                logging.info(f"SLAM initialized at origin (0, 0, 0)")

            # Update pose estimate using relative movement
            if self.prev_odom is not None:
                # Calculate movement in world frame
                dx_world = current_odom[0] - self.prev_odom[0]
                dy_world = current_odom[1] - self.prev_odom[1]
                dtheta = current_odom[2] - self.prev_odom[2]

                # Normalize angle difference
                while dtheta > np.pi:
                    dtheta -= 2 * np.pi
                while dtheta < -np.pi:
                    dtheta += 2 * np.pi

                # Debug: Print movement
                if abs(dx_world) > 0.01 or abs(dy_world) > 0.01 or abs(dtheta) > 0.01:
                    logging.info(f"Robot moved: dx={dx_world:.3f}, dy={dy_world:.3f}, dtheta={dtheta:.3f}")

                # Update current pose directly in world frame
                x, y, theta = self.current_pose
                x += dx_world
                y += dy_world
                theta += dtheta

                # Normalize theta
                theta = np.arctan2(np.sin(theta), np.cos(theta))

                self.current_pose = (x, y, theta)

                if abs(dx_world) > 0.01 or abs(dy_world) > 0.01:
                    logging.info(f"New pose: x={x:.3f}, y={y:.3f}, theta={theta:.3f}")

            self.prev_odom = current_odom

        except Exception as e:
            logging.error(f"Error in sportmodestate callback: {e}", exc_info=True)

    def _setup_visualization(self):
        """Setup matplotlib visualization"""
        plt.ion()  # Enable interactive mode
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.set_xlabel('X (meters)')
        self.ax.set_ylabel('Y (meters)')
        self.ax.set_title('Real-time SLAM Map')
        self.ax.grid(True, alpha=0.3)

        # Initialize empty plot
        map_image = self.slam.get_map_image()
        extent = [
            self.slam.map.origin[0],
            self.slam.map.origin[0] + self.slam.map.width * self.slam.map.resolution,
            self.slam.map.origin[1],
            self.slam.map.origin[1] + self.slam.map.height * self.slam.map.resolution
        ]
        self.map_plot = self.ax.imshow(map_image, cmap='gray', origin='lower',
                                        extent=extent, vmin=0, vmax=255, alpha=0.8)

        # Initialize robot position and direction plots
        self.robot_plot, = self.ax.plot([], [], 'bo', markersize=10, label='Robot')
        self.robot_arrow = self.ax.arrow(0, 0, 0, 0, head_width=0.3, head_length=0.5, fc='red', ec='red')

        # Add status text
        self.status_text = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes,
                                        verticalalignment='top', bbox=dict(boxstyle='round',
                                        facecolor='wheat', alpha=0.8), fontsize=9)

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

            # Update robot position
            x, y, theta = self.current_pose
            self.robot_plot.set_data([x], [y])

            # Update robot direction arrow
            if self.robot_arrow is not None:
                self.robot_arrow.remove()
            arrow_length = 1.0  # 1 meter arrow
            dx = arrow_length * np.cos(theta)
            dy = arrow_length * np.sin(theta)
            self.robot_arrow = self.ax.arrow(x, y, dx, dy,
                                            head_width=0.3, head_length=0.5,
                                            fc='red', ec='red', alpha=0.8)

            # Update status text
            status_str = f'Robot Pose:\n'
            status_str += f'  X: {x:.2f}m\n'
            status_str += f'  Y: {y:.2f}m\n'
            status_str += f'  θ: {np.degrees(theta):.1f}°'
            self.status_text.set_text(status_str)

            # Zoom to robot-centered view (±10m around robot)
            zoom_range = 10.0  # meters
            self.ax.set_xlim(x - zoom_range, x + zoom_range)
            self.ax.set_ylim(y - zoom_range, y + zoom_range)

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

            positions_raw = data_inner.get("positions", [])

            # Check if positions is valid
            if positions_raw is None or (hasattr(positions_raw, '__len__') and len(positions_raw) == 0):
                logging.warning(f"No LIDAR points received (scan {self.scan_count})")
                return

            # Convert to numpy array and normalize shape
            positions_np = np.asarray(positions_raw, dtype=np.float32)
            if positions_np.size == 0:
                logging.warning(f"No LIDAR points received after conversion (scan {self.scan_count})")
                return

            if positions_np.ndim == 1:
                if positions_np.size % 3 != 0:
                    logging.error(f"Unexpected LIDAR positions length ({positions_np.size}); cannot reshape into XYZ triplets")
                    return
                voxel_indices = positions_np.reshape(-1, 3)
            elif positions_np.ndim == 2 and positions_np.shape[1] == 3:
                voxel_indices = positions_np
            else:
                logging.error(f"Unexpected LIDAR positions shape {positions_np.shape}; expected (N*3,) or (N,3)")
                return

            # Get voxel map parameters
            origin = data.get("origin", [0.0, 0.0, 0.0])
            width = data.get("width", [128, 128, 38])
            resolution = data.get("resolution", 0.05)

            # Debug: Print voxel map info
            if self.scan_count == 0:
                logging.info(f"Voxel map - Origin: {origin}, Width: {width}, Resolution: {resolution}")

            # CRITICAL FIX: positions are voxel grid INDICES, not world coordinates!
            # Convert voxel indices to 3D points in world frame
            origin_np = np.asarray(origin, dtype=np.float32)

            # Filter out voxel map boundary points to avoid artifact edges
            # Boundary points (index 0 or max) represent the physical limit of the voxel map
            # and should not be treated as real obstacles
            boundary_margin = 2  # cells
            voxel_mask = (
                (voxel_indices[:, 0] > boundary_margin) & (voxel_indices[:, 0] < width[0] - boundary_margin) &
                (voxel_indices[:, 1] > boundary_margin) & (voxel_indices[:, 1] < width[1] - boundary_margin)
            )
            voxel_indices = voxel_indices[voxel_mask]

            if len(voxel_indices) == 0:
                logging.warning(f"All voxels filtered out by boundary filter (scan {self.scan_count})")
                return

            points = origin_np + (voxel_indices * resolution)

            if self.scan_count == 0 and len(voxel_indices) > 0:
                logging.info(f"First scan: {len(voxel_indices)} voxels")
                logging.info(f"Voxel indices range: X=[{voxel_indices[:,0].min()}, {voxel_indices[:,0].max()}], "
                           f"Y=[{voxel_indices[:,1].min()}, {voxel_indices[:,1].max()}], "
                           f"Z=[{voxel_indices[:,2].min()}, {voxel_indices[:,2].max()}]")

            if self.scan_count == 0 and len(points) > 0:
                logging.info(f"World coords range: X=[{points[:,0].min():.2f}, {points[:,0].max():.2f}], "
                           f"Y=[{points[:,1].min():.2f}, {points[:,1].max():.2f}], "
                           f"Z=[{points[:,2].min():.2f}, {points[:,2].max():.2f}]")

            # Wait for odometry to be initialized
            if self.odom_origin is None or self.prev_odom is None:
                if self.scan_count == 0:
                    logging.warning("Waiting for odometry data before processing LIDAR...")
                return

            # Get current robot position and orientation from odometry (world frame)
            robot_x_world = self.prev_odom[0]
            robot_y_world = self.prev_odom[1]
            robot_yaw_world = self.prev_odom[2]

            if self.scan_count == 0:
                logging.info(f"Robot pose (world frame): ({robot_x_world:.2f}, {robot_y_world:.2f}, yaw={np.degrees(robot_yaw_world):.1f}°)")

            # Convert to robot-centered coordinates using actual odometry position
            points[:, 0] -= robot_x_world
            points[:, 1] -= robot_y_world

            # CRITICAL: Rotate points from world frame to robot body frame
            # World frame lidar data needs to be rotated by -yaw to get robot body frame
            cos_yaw = np.cos(-robot_yaw_world)
            sin_yaw = np.sin(-robot_yaw_world)
            x_rotated = points[:, 0] * cos_yaw - points[:, 1] * sin_yaw
            y_rotated = points[:, 0] * sin_yaw + points[:, 1] * cos_yaw
            points[:, 0] = x_rotated
            points[:, 1] = y_rotated

            if self.scan_count == 0 and len(points) > 0:
                logging.info(f"Robot-centered coords range: X=[{points[:,0].min():.2f}, {points[:,0].max():.2f}], "
                           f"Y=[{points[:,1].min():.2f}, {points[:,1].max():.2f}]")

            # Debug: Log Z coordinate range
            if self.scan_count % 10 == 0 and len(points) > 0:
                logging.info(f"Z coords range: [{points[:,2].min():.2f}, {points[:,2].max():.2f}]m")

            # Filter by height (Z axis) - take points near ground level for 2D mapping
            height_min = -0.2  # 센서 아래 20cm만 포함해 지면 반사 억제
            height_max = 1.0   # 센서 위 1m (테이블/사람 포함)
            height_mask = (points[:, 2] >= height_min) & (points[:, 2] <= height_max)

            # Filter out points that are too far or too close
            ranges_temp = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
            range_mask = (ranges_temp >= self.slam.params.min_range) & (ranges_temp <= self.slam.params.max_range)

            # Combine height and range filters
            valid_mask = height_mask & range_mask

            if self.scan_count % 10 == 0:
                logging.info(f"Valid points: {np.sum(valid_mask)}/{len(points)} "
                           f"(height filtered: {len(points) - np.sum(height_mask)}, "
                           f"range filtered: {len(points) - np.sum(range_mask)})")

            points_filtered = points[valid_mask]

            if len(points_filtered) == 0:
                logging.warning(f"All points filtered out (scan {self.scan_count})")
                return

            # Downsample points for performance (500 points for good quality)
            downsample_factor = max(1, len(points_filtered) // 500)  # Limit to ~500 points
            points_filtered = points_filtered[::downsample_factor]

            if self.scan_count % 10 == 0:
                logging.info(f"After downsampling: {len(points_filtered)} points (factor: {downsample_factor})")

            # Convert point cloud to polar coordinates (range, angle)
            # Points are already in robot frame (x-forward, y-left)
            # Restrict to the robot's forward 120° field of view
            angles_rad = np.arctan2(points_filtered[:, 1], points_filtered[:, 0])
            front_mask = np.abs(angles_rad) <= self.front_angle_limit
            points_filtered = points_filtered[front_mask]
            angles_rad = angles_rad[front_mask]

            if len(points_filtered) == 0:
                logging.warning(f"No LIDAR points within ±{np.degrees(self.front_angle_limit):.0f}° (scan {self.scan_count})")
                return

            ranges = np.linalg.norm(points_filtered[:, :2], axis=1).tolist()
            angles = angles_rad.tolist()

            # Debug: Print range statistics and front obstacle distance
            if self.scan_count % 10 == 0:
                logging.info(f"Scan {self.scan_count}: {len(ranges)} points, "
                           f"range: [{min(ranges):.2f}, {max(ranges):.2f}]m")

                # Find obstacles directly in front (angle close to 0)
                front_obstacles = []
                for r, a in zip(ranges, angles):
                    # Check if angle is within ±15 degrees of front (0 radians)
                    if abs(a) < np.radians(15):
                        front_obstacles.append(r)

                if front_obstacles:
                    min_front_dist = min(front_obstacles)
                    logging.info(f">>> FRONT OBSTACLE: {min_front_dist:.2f}m ahead (±15°)")

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

            # Update visualization every 5 scans for faster real-time feedback
            if self.scan_count % 5 == 0:
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

        # Initialize SLAM mapper with real-time visualization
        mapper = SLAMMapper(enable_visualization=True)

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

        # Keep-alive task to prevent WebRTC consent timeout
        async def keep_alive():
            """Periodically request data to keep WebRTC connection alive"""
            from go2_webrtc_driver.constants import RTC_TOPIC
            while True:
                try:
                    await asyncio.sleep(10)  # Every 10 seconds
                    # Request motion mode status to keep connection active
                    await conn.datachannel.pub_sub.publish_request_new(
                        RTC_TOPIC["MOTION_SWITCHER"],
                        {"api_id": 1001}
                    )
                except Exception as e:
                    logging.debug(f"Keep-alive error (expected on shutdown): {e}")
                    break

        # Start keep-alive task
        keep_alive_task = asyncio.create_task(keep_alive())

        # Run for specified duration
        duration = 600  # 10 minutes (increase as needed)
        logging.info(f"Mapping for {duration} seconds ({duration//60} minutes)... (Press Ctrl+C to stop early)")

        try:
            await asyncio.sleep(duration)
        except (KeyboardInterrupt, asyncio.CancelledError):
            logging.info("\nMapping interrupted by user")
        finally:
            # Cancel keep-alive task
            keep_alive_task.cancel()
            try:
                await keep_alive_task
            except asyncio.CancelledError:
                pass

        # Save the map (always save, even if interrupted)
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

    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully - save map before exiting
        logging.info("\n\nKeyboard interrupt detected - saving map before exit...")
        try:
            map_filename = "go2_map.pkl"
            mapper.slam.save_map(map_filename)
            logging.info(f"Map saved to {map_filename}")
            logging.info(f"Total scans processed: {mapper.scan_count}")
        except Exception as e:
            logging.error(f"Error saving map: {e}")
    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram interrupted by user - map should be saved")
        sys.exit(0)
